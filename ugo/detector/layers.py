# Box decoding, detection MLP, SAM2 mask refinement, prototype attention, prompt encoding, positional encoding, and upsampling adapted from GECO2 commit ef6954a59d22620b9a3052f8500530a799c0379b:
# https://github.com/jerpelhan/GECO2/tree/ef6954a59d22620b9a3052f8500530a799c0379b/models
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# SAM2-derived portions are licensed under third_party/sam2/LICENSE.

import math
from typing import Any, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder as SamPromptEncoder
from sam2.modeling.sam.transformer import TwoWayTransformer


def boxes_with_scores(density_map, tlrb, sort=False, validate=False):
    batch_size, channels, _, _ = density_map.shape
    pooled = F.max_pool2d(density_map, 3, 1, 1)
    if validate:
        batch_thresh = torch.max(density_map.reshape(batch_size, -1), dim=-1).values.view(
            batch_size, channels, 1, 1) / 8
    else:
        batch_thresh = torch.median(density_map.reshape(batch_size, -1),
                                    dim=-1).values.view(batch_size, channels, 1, 1)
    mask = (pooled == density_map) & (density_map > batch_thresh)

    outputs = []
    reference_points_batch = []
    for batch_index in range(batch_size):
        box_scores = density_map[batch_index, mask[batch_index]]
        reference_points = mask[batch_index].nonzero()[:, -2:]
        box_centers = reference_points / torch.tensor(mask.shape[2:], device=mask.device)
        offsets = tlrb[batch_index].permute(1, 2, 0)
        box_offsets = offsets[mask[batch_index].permute(1, 2, 0).expand_as(offsets)].reshape(-1, 4)
        sign = torch.tensor([-1, -1, 1, 1], device=mask.device)
        boxes = box_centers.flip(-1).repeat(1, 2) + sign * box_offsets

        if sort:
            order = torch.argsort(box_scores, descending=True)
            box_scores = box_scores[order]
            boxes = boxes[order]
            reference_points = reference_points[order]

        outputs.append({
            "pred_boxes": boxes.unsqueeze(0),
            "box_v": box_scores.unsqueeze(0),
        })
        reference_points_batch.append(reference_points.T)

    return outputs, reference_points_batch


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        hidden_dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(input_size, output_size) for input_size, output_size in zip(
                [input_dim] + hidden_dims,
                hidden_dims + [output_dim],
            ))

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
        return x


class UpsamplingLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.reset_parameters()

    def forward(self, x):
        return self.layer(x)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class PrototypeAttentionBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.cross_attention = Attention(embedding_dim, num_heads)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, image_f, prototypes):
        image_f = image_f + self.cross_attention(
            q=image_f,
            k=prototypes,
            v=prototypes,
        )
        return self.norm(image_f)


class Attention(nn.Module):

    def __init__(self, embedding_dim, num_heads, downsample_rate=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x, num_heads):
        batch_size, token_count, channels = x.shape
        x = x.reshape(
            batch_size,
            token_count,
            num_heads,
            channels // num_heads,
        )
        return x.transpose(1, 2)

    def _recombine_heads(self, x):
        batch_size, num_heads, token_count, channels_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(
            batch_size,
            token_count,
            num_heads * channels_per_head,
        )

    def forward(self, q, k, v):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        channels_per_head = q.shape[-1]
        attention = q @ k.permute(0, 1, 3, 2)
        attention = attention / math.sqrt(channels_per_head)
        attention = torch.softmax(attention, dim=-1)
        output = attention @ v
        output = self._recombine_heads(output)
        return self.out_proj(output)


class PositionEmbeddingRandom(nn.Module):

    def __init__(self, num_pos_feats=64, scale: Optional[float] = None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords):
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]):
        height, width = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones(
            (height, width),
            device=device,
            dtype=torch.float32,
        )
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / height
        x_embed = x_embed / width
        positional_encoding = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_encoding.permute(2, 0, 1)


class PromptEncoder(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

    def get_dense_pe(self):
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


class MaskProcessor(nn.Module):

    def __init__(
        self,
        hidden_dim,
        image_size,
        reduction,
        sam_checkpoint_path,
    ):
        super().__init__()
        self.sam_prompt_embed_dim = hidden_dim
        self.reduction = reduction
        self.sam_image_embedding_size = image_size // reduction
        self.image_size = image_size
        self.prompt_encoder_sam = SamPromptEncoder(
            embed_dim=hidden_dim,
            image_embedding_size=(
                self.sam_image_embedding_size,
                self.sam_image_embedding_size,
            ),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=hidden_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=hidden_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=True,
            iou_prediction_use_sigmoid=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
        )
        self.num_feature_levels = 3
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        checkpoint = torch.load(sam_checkpoint_path, map_location="cpu")
        sam_state = checkpoint["model"] if "model" in checkpoint else checkpoint
        mask_decoder_state = {
            key.replace("mask_decoder.", "").replace("sam_", ""): value
            for key, value in sam_state.items() if "mask_decoder" in key
        }
        self.mask_decoder.load_state_dict(mask_decoder_state)
        prompt_encoder_state = {
            key.replace("prompt_encoder.", "").replace("sam_", ""): value
            for key, value in sam_state.items() if "prompt_encoder" in key
        }
        self.prompt_encoder_sam.load_state_dict(prompt_encoder_state)
        no_memory_state = {key: value for key, value in sam_state.items() if "no_mem_embed" in key}
        self.load_state_dict(no_memory_state, strict=False)

    def _prepare_backbone_features(self, backbone_out):
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
        feat_sizes = [(embedding.shape[-2], embedding.shape[-1]) for embedding in vision_pos_embeds]
        vision_feats = [feature_map.flatten(2).permute(2, 0, 1) for feature_map in feature_maps]
        vision_pos_embeds = [
            embedding.flatten(2).permute(2, 0, 1) for embedding in vision_pos_embeds
        ]
        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def forward_feats(self, feats):
        feats["backbone_fpn"][0] = self.mask_decoder.conv_s0(feats["backbone_fpn"][0])
        feats["backbone_fpn"][1] = self.mask_decoder.conv_s1(feats["backbone_fpn"][1])
        _, vision_feats, _, _ = self._prepare_backbone_features(feats)
        vision_feats[-1] = vision_feats[-1] + self.no_mem_embed
        batch_size = vision_feats[0].shape[1]
        return [
            feature.permute(1, 2, 0).view(
                batch_size,
                -1,
                *feature_size,
            ) for feature, feature_size in zip(
                vision_feats[::-1],
                self._bb_feat_sizes[::-1],
            )
        ][::-1]

    def forward(self, features_orig, outputs, sequence):
        batch_masks = []
        batch_iou = []
        batch_bboxes = []

        for image_index in range(len(outputs)):
            features = {
                "vision_features":
                features_orig["vision_features"][image_index].unsqueeze(0),
                "vision_pos_enc":
                [feature[image_index].unsqueeze(0) for feature in features_orig["vision_pos_enc"]],
                "backbone_fpn":
                [feature[image_index].unsqueeze(0) for feature in features_orig["backbone_fpn"]],
            }
            features = self.forward_feats(features)
            step = 100
            iou_predictions = []
            corrected_box_batches = []
            mask_batches = []

            for box_index in range(
                    step,
                    len(outputs[image_index]["pred_boxes"][0]) + step,
                    step,
            ):
                box = outputs[image_index]["pred_boxes"][0][(box_index -
                                                             step):box_index] * self.image_size
                box_coords = box.reshape(-1, 2, 2)
                box_labels = torch.tensor(
                    [[2, 3]],
                    dtype=torch.int,
                    device=box.device,
                )
                box_labels = box_labels.repeat(box.size(0), 1)
                sparse_embeddings, dense_embeddings = self.prompt_encoder_sam(
                    points=(box_coords, box_labels),
                    boxes=None,
                    masks=None,
                )
                low_res_masks, iou_prediction, _, _ = self.mask_decoder(
                    image_embeddings=features[-1],
                    image_pe=self.prompt_encoder_sam.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=True,
                    high_res_features=features[:-1],
                )
                best_iou_indices = torch.argmax(iou_prediction, dim=-1)
                batch_indices = torch.arange(
                    low_res_masks.shape[0],
                    device=iou_prediction.device,
                )
                low_res_masks = low_res_masks[
                    batch_indices,
                    best_iou_indices,
                ].unsqueeze(1)
                iou_prediction = iou_prediction[
                    batch_indices,
                    best_iou_indices,
                ]
                iou_predictions.append(iou_prediction)

                masks = F.interpolate(
                    low_res_masks,
                    sequence["img_size"],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                if len(masks.shape) == 2:
                    masks = masks.unsqueeze(0)

                corrected_boxes = torch.zeros(
                    (masks.shape[0], 4),
                    dtype=torch.float,
                    device=masks.device,
                )
                for mask_index, mask in enumerate(masks):
                    y, x = torch.where(mask > 0)
                    if y.shape[0] > 0 and x.shape[0] > 0:
                        corrected_boxes[mask_index, 0] = torch.min(x)
                        corrected_boxes[mask_index, 1] = torch.min(y)
                        corrected_boxes[mask_index, 2] = torch.max(x)
                        corrected_boxes[mask_index, 3] = torch.max(y)

                mask_batches.append(masks)
                corrected_box_batches.append(corrected_boxes)

            if corrected_box_batches:
                batch_masks.append(mask_batches)
                batch_bboxes.append(torch.cat(corrected_box_batches))
                batch_iou.append(torch.cat(iou_predictions).unsqueeze(0))
            else:
                batch_masks.append([])
                batch_bboxes.append(torch.tensor([]).to(features[0].device))
                batch_iou.append(torch.tensor([]).to(features[0].device))

        batch_masks = [
            torch.cat(masks) if masks else torch.zeros((1, 1024, 1024)) for masks in batch_masks
        ]
        return batch_masks, batch_iou, batch_bboxes
