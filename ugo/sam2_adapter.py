# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from Segment Anything 2; see third_party/sam2/LICENSE.

import copy
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2.modeling.sam2_utils import get_1d_sine_pe
from sam2.utils.misc import get_connected_components


class UGOSAM2Base(SAM2Base):

    def __init__(
        self,
        *args,
        memory_temporal_stride_for_eval=5,
        **kwargs,
    ):
        super().__init__(
            *args,
            memory_temporal_stride_for_eval=memory_temporal_stride_for_eval,
            **kwargs,
        )

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        batch_size = backbone_features.size(0)
        device = backbone_features.device
        assert backbone_features.size(1) == self.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == batch_size
            assert sam_point_labels.size(0) == batch_size
        else:
            sam_point_coords = torch.zeros(batch_size, 1, 2, device=device)
            sam_point_labels = -torch.ones(
                batch_size,
                1,
                dtype=torch.int32,
                device=device,
            )

        if mask_inputs is not None:
            assert len(mask_inputs.shape) == 4
            assert mask_inputs.shape[:2] == (batch_size, 1)
            if mask_inputs.shape[-2:] != self.sam_prompt_encoder.mask_input_size:
                sam_mask_prompt = F.interpolate(
                    mask_inputs.float(),
                    size=self.sam_prompt_encoder.mask_input_size,
                    align_corners=False,
                    mode="bilinear",
                    antialias=True,
                )
            else:
                sam_mask_prompt = mask_inputs
        else:
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                NO_OBJ_SCORE,
            )

        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        track_ious = None
        sam_output_token = sam_output_tokens[:, 0]
        obj_ptrs = None
        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(batch_size, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            track_ious = ious[batch_inds, best_iou_inds].unsqueeze(1)
            obj_ptrs = self.obj_ptr_proj(sam_output_tokens)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks = low_res_multimasks
            high_res_masks = high_res_multimasks

        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            track_ious,
            (ious, low_res_multimasks, obj_ptrs),
        )

    def _use_mask_as_output(
        self,
        backbone_features,
        high_res_features,
        mask_inputs,
    ):
        out_scale = 20.0
        out_bias = -10.0
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(
                high_res_masks.size(-2) // 4,
                high_res_masks.size(-1) // 4,
            ),
            align_corners=False,
            mode="bilinear",
            antialias=True,
        )
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            obj_ptr = torch.zeros(
                mask_inputs.size(0),
                self.hidden_dim,
                device=mask_inputs.device,
            )
        else:
            _, _, _, _, _, obj_ptr, _, _, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )

        is_obj_appearing = torch.any(
            mask_inputs.flatten(1).float() > 0.0,
            dim=1,
        )
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            None,
            (None, None, None),
        )

    def forward_image(self, img_batch):
        backbone_out = self.image_encoder(img_batch)
        original_backbone_out = copy.deepcopy(backbone_out)
        if self.use_high_res_features_in_sam:
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1])
        return backbone_out, original_backbone_out

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
    ):
        batch_size = current_vision_feats[-1].size(1)
        hidden_dim = self.hidden_dim
        height, width = feat_sizes[-1]
        device = current_vision_feats[-1].device

        if self.num_maskmem == 0:
            return (current_vision_feats[-1].permute(1, 2, 0).view(batch_size, hidden_dim, height,
                                                                   width))

        num_obj_ptr_tokens = 0
        if not is_init_cond_frame:
            object_ids = output_dict["obj_ids_list"]
            object_memories = []
            object_memory_positions = []
            object_pointers = []
            object_pointer_positions = []

            for object_id in object_ids:
                max_delta_t = 1
                memories = []
                memory_positions = []
                pointers = []
                pointer_positions = []
                memory_entries = output_dict["per_obj_dict"][object_id]

                for memory_index, memory_entry in enumerate(memory_entries):
                    memories.append(memory_entry["maskmem_features"])
                    pointers.append(memory_entry["obj_ptr"])
                    delta_t = frame_idx - memory_entry["frame_idx"]
                    if memory_entry["is_init"]:
                        memory_tpos = self.maskmem_tpos_enc[-1]
                    else:
                        tpos_index = len(memory_entries) - memory_index - 1
                        memory_tpos = self.maskmem_tpos_enc[tpos_index]
                        if delta_t > max_delta_t:
                            max_delta_t = delta_t
                    memory_positions.append(output_dict["maskmem_pos_enc"] +
                                            memory_tpos.permute(2, 0, 1).unsqueeze(0))
                    pointer_positions.append(delta_t)

                pointer_positions = torch.tensor(pointer_positions).to(
                    device=device,
                    non_blocking=True,
                )
                tpos_dim = (hidden_dim if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim)
                pointer_positions = get_1d_sine_pe(
                    pointer_positions / max_delta_t,
                    dim=tpos_dim,
                )
                pointer_positions = self.obj_ptr_tpos_proj(pointer_positions)

                object_memories.append(torch.cat(memories, 0))
                object_memory_positions.append(torch.cat(memory_positions, 0))
                object_pointers.append(torch.cat(pointers, 0))
                object_pointer_positions.append(pointer_positions)

            conditioned_features = []
            for object_index, _ in enumerate(object_ids):
                memory = (object_memories[object_index].flatten(2).permute(2, 0, 1))
                memory_position = (object_memory_positions[object_index].flatten(2).permute(
                    2, 0, 1))
                if self.mem_dim >= hidden_dim:
                    raise RuntimeError("UGO private memory requires mem_dim < hidden_dim")

                obj_ptrs = object_pointers[object_index].reshape(
                    -1,
                    hidden_dim // self.mem_dim,
                    self.mem_dim,
                )
                obj_positions = object_pointer_positions[object_index]
                obj_positions = obj_positions.unsqueeze(1).repeat_interleave(
                    hidden_dim // self.mem_dim,
                    dim=1,
                )
                num_obj_ptr_tokens = obj_ptrs.shape[0] * obj_ptrs.shape[1]
                memory = torch.cat(
                    [
                        memory.permute(1, 0, 2).flatten(0, 1),
                        obj_ptrs.flatten(0, 1),
                    ],
                    dim=0,
                ).unsqueeze(1)
                memory_position = torch.cat(
                    [
                        memory_position.permute(1, 0, 2).flatten(0, 1),
                        obj_positions.flatten(0, 1),
                    ],
                    dim=0,
                ).unsqueeze(1)

                current_features = [current_vision_feats[0][:, 0:1, :]]
                current_positions = [current_vision_pos_embeds[0][:, 0:1, :]]
                conditioned_feature = self.memory_attention(
                    curr=current_features,
                    curr_pos=current_positions,
                    memory=memory,
                    memory_pos=memory_position,
                    num_obj_ptr_tokens=num_obj_ptr_tokens,
                )
                conditioned_features.append(
                    conditioned_feature.permute(1, 2, 0).view(
                        1,
                        hidden_dim,
                        height,
                        width,
                    ))

            return torch.cat(conditioned_features, dim=0)

        if self.directly_add_no_mem_embed:
            conditioned_features = current_vision_feats[-1] + self.no_mem_embed
            return conditioned_features.permute(1, 2, 0).view(
                batch_size,
                hidden_dim,
                height,
                width,
            )

        memory = self.no_mem_embed.expand(1, batch_size, self.mem_dim)
        memory_position = self.no_mem_pos_enc.expand(
            1,
            batch_size,
            self.mem_dim,
        )
        conditioned_features = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_position,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        return conditioned_features.permute(1, 2, 0).view(
            batch_size,
            hidden_dim,
            height,
            width,
        )

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
        prev_masks=None,
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )
        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
            track_ious,
            (ious, low_res_multimasks, obj_ptrs),
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            current_out["object_score_logits"] = object_score_logits
            current_out["track_ious"] = track_ious
            current_out["ious"] = ious
            current_out["low_res_multimasks"] = low_res_multimasks
            current_out["obj_ptrs"] = obj_ptrs

        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )
        return current_out


def fill_holes_in_mask_scores(mask, max_area):
    assert max_area > 0
    input_mask = mask
    try:
        labels, areas = get_connected_components(mask <= 0)
        is_hole = (labels > 0) & (areas <= max_area)
        mask = torch.where(is_hole, 0.1, mask)
        labels, areas = get_connected_components(mask >= 0)
        min_area = torch.max(
            areas.reshape(areas.size(0), -1),
            1,
        )[0] * 0.2
        is_small_blob = (labels > 0) & (areas < min_area.view(-1, 1, 1, 1))
        mask = torch.where(is_small_blob, 0.0, mask)
    except Exception as error:
        warnings.warn(
            f"{error}\n\nSkipping the post-processing step due to the error above. "
            "You can still use SAM 2 and it's OK to ignore the error above, although "
            "some post-processing functionality may be limited (which doesn't affect "
            "the results in most cases; see "
            "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
            category=UserWarning,
            stacklevel=2,
        )
        mask = input_mask
    return mask


def build_ugo_sam2(config_path, checkpoint_path, device):
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    model = instantiate(config.model, _recursive_=True)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )["model"]
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint)
    if missing_keys or unexpected_keys:
        raise RuntimeError(f"SAM2 checkpoint mismatch: missing={missing_keys}, "
                           f"unexpected={unexpected_keys}")
    model = model.to(device)
    model.eval()
    return model
