# Detection heads and prototype extraction adapted from GECO2 commit ef6954a59d22620b9a3052f8500530a799c0379b:
# https://github.com/jerpelhan/GECO2/blob/ef6954a59d22620b9a3052f8500530a799c0379b/models/counter.py

import torch
from torch import nn
from torch.cuda.amp import autocast
from torchvision import ops
from torchvision.ops import roi_align

from .box_correction import BoxCorrection
from .layers import MLP, PromptEncoder, boxes_with_scores
from .prototype_attention import PrototypeAttention


class GeCo2Detector(nn.Module):

    def __init__(
        self,
        sam_checkpoint_path,
        image_size=1024,
        emb_dim=256,
        reduction=16,
    ):
        super().__init__()
        self.validate = True
        self.emb_dim = emb_dim

        self.class_embed = nn.Sequential(
            nn.Linear(emb_dim, 1),
            nn.LeakyReLU(),
        )
        self.bbox_embed = MLP(emb_dim, emb_dim, 4, 3)
        self.class_embed_aux = nn.Sequential(
            nn.Linear(emb_dim, 1),
            nn.LeakyReLU(),
        )
        self.bbox_embed_aux = MLP(emb_dim, emb_dim, 4, 3)
        self.adapt_features = PrototypeAttention(
            transformer_dim=emb_dim,
            num_prototype_attn_steps=3,
            num_image_attn_steps=2,
        )
        self.sam_prompt_encoder = PromptEncoder(
            embed_dim=emb_dim,
            image_embedding_size=(
                image_size // reduction,
                image_size // reduction,
            ),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        self.shape_or_objectness = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.box_correction = BoxCorrection(
            reduction,
            image_size,
            emb_dim,
            sam_checkpoint_path,
        )

    def extract_prototypes(self, feats, sequence):
        src = feats["vision_features"]
        exemplar = sequence["exemplar"][0].unsqueeze(0)
        exemplar = exemplar.to(src.device)
        exemplar[:, :, 2] = exemplar[:, :, 0] + exemplar[:, :, 2]
        exemplar[:, :, 3] = exemplar[:, :, 1] + exemplar[:, :, 3]
        num_objects = exemplar.size(1)
        batch_size, _, width, _ = src.shape
        reduction = 1024 / width
        boxes_roi = torch.cat(
            [
                torch.arange(
                    batch_size,
                    device=exemplar.device,
                    requires_grad=False,
                ).repeat_interleave(num_objects).reshape(-1, 1),
                exemplar.flatten(0, 1),
            ],
            dim=1,
        )
        kernel_dim = 1
        exemplars = roi_align(
            src,
            boxes=boxes_roi.to(src.dtype),
            output_size=kernel_dim,
            spatial_scale=1.0 / reduction,
            aligned=True,
        ).permute(0, 2, 3, 1).reshape(
            batch_size,
            num_objects * kernel_dim**2,
            self.emb_dim,
        )

        level_1 = feats["backbone_fpn"][0]
        level_2 = feats["backbone_fpn"][1]
        exemplars_level_1 = roi_align(
            level_1,
            boxes=boxes_roi,
            output_size=kernel_dim,
            spatial_scale=1.0 / reduction * 2 * 2,
            aligned=True,
        ).permute(0, 2, 3, 1).reshape(
            batch_size,
            num_objects * kernel_dim**2,
            self.emb_dim,
        )
        exemplars_level_2 = roi_align(
            level_2,
            boxes=boxes_roi,
            output_size=kernel_dim,
            spatial_scale=1.0 / reduction * 2,
            aligned=True,
        ).permute(0, 2, 3, 1).reshape(
            batch_size,
            num_objects * kernel_dim**2,
            self.emb_dim,
        )

        box_hw = exemplar[:, :, 2:4] - exemplar[:, :, 0:2]
        shape = self.shape_or_objectness(box_hw).reshape(
            batch_size,
            -1,
            self.emb_dim,
        )
        prototype_embeddings = torch.cat(
            [exemplars, shape],
            dim=1,
        )
        prototype_embeddings_level_1 = torch.cat(
            [exemplars_level_1, shape],
            dim=1,
        )
        prototype_embeddings_level_2 = torch.cat(
            [exemplars_level_2, shape],
            dim=1,
        )
        high_quality_prototypes = [
            prototype_embeddings_level_1,
            prototype_embeddings_level_2,
        ]
        return [
            prototype_embeddings,
            high_quality_prototypes,
        ]

    def forward(
        self,
        feats,
        sequence,
        queries,
    ):
        src = feats["vision_features"]
        prototype_embeddings, high_quality_prototypes = queries

        with autocast(enabled=False):
            if src.type != torch.float32:
                src = src.float()
                prototype_embeddings = prototype_embeddings.float()
                high_quality_prototypes = [
                    prototype.float() for prototype in high_quality_prototypes
                ]
                feats["backbone_fpn"] = [feature.float() for feature in feats["backbone_fpn"]]
                feats["vision_pos_enc"] = [feature.float() for feature in feats["vision_pos_enc"]]

            adapted_features, _ = self.adapt_features(
                image_embeddings=src,
                image_pe=self.sam_prompt_encoder.get_dense_pe(),
                prototype_embeddings=prototype_embeddings,
                hq_features=feats["backbone_fpn"],
                hq_prototypes=high_quality_prototypes,
                hq_pos=feats["vision_pos_enc"],
            )

        batch_size, _, width, height = adapted_features.shape
        adapted_features = adapted_features.view(
            batch_size,
            self.emb_dim,
            -1,
        ).permute(0, 2, 1)
        centerness = self.class_embed(adapted_features).view(
            batch_size,
            width,
            height,
            1,
        ).permute(0, 3, 1, 2)
        output_coordinates = self.bbox_embed(adapted_features).sigmoid().view(
            batch_size,
            width,
            height,
            4,
        ).permute(0, 3, 1, 2)
        outputs, _ = boxes_with_scores(
            centerness,
            output_coordinates,
            sort=False,
            validate=self.validate,
        )

        outputs = self.box_correction(feats, outputs, sequence)
        mask_logits = outputs[1]
        outputs = outputs[0]
        batch_index = 0
        if outputs[batch_index]["box_v"].shape[1] > 0:
            filter_mask = (outputs[batch_index]["box_v"] > outputs[batch_index]["box_v"].max() / 5)
            keep = ops.nms(
                outputs[batch_index]["pred_boxes"][filter_mask],
                outputs[batch_index]["box_v"][filter_mask],
                0.6,
            )
            boxes = outputs[batch_index]["pred_boxes"][filter_mask][keep]
            mask_logits = mask_logits[batch_index][filter_mask[0]][keep]
            sam_scores = outputs[batch_index]["scores"][filter_mask][keep]
            geco2_scores = outputs[batch_index]["box_v"][filter_mask][keep]
            merged_scores = geco2_scores * 0.2 + sam_scores * 0.8
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            scores = (
                merged_scores,
                geco2_scores / geco2_scores.max(),
            )
            return boxes, scores, centerness, mask_logits
        return [], ([], []), [], mask_logits[batch_index][:0]


def build_geco2_detector(
    checkpoint_path,
    sam_checkpoint_path,
    device,
):
    model = GeCo2Detector(sam_checkpoint_path=sam_checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    missing_detector_keys = [
        key for key in incompatible_keys.missing_keys
        if not key.startswith("box_correction.sam_mask.")
    ]
    if missing_detector_keys:
        raise RuntimeError(f"GECO2 checkpoint is missing detector keys: {missing_detector_keys}")
    return model.to(device).eval()
