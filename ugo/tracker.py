import torch
from torch.nn import functional as F
from torchvision import transforms as T

from .hypothesis_merging import HypothesisMerger
from .mot_format import mask2bbox, mask_iou
from .sam2_adapter import build_ugo_sam2, fill_holes_in_mask_scores


class UGOTracker:

    def __init__(
        self,
        sam_checkpoint,
        sam_config,
        device="cuda",
        tau_lo=0.2,
        tau_hi=0.8,
        match_overlap_threshold=0.6,
    ):
        self.device = torch.device(device)
        self.sam = build_ugo_sam2(sam_config, sam_checkpoint, self.device)
        self.fill_hole_area = 8
        self._resize_to_input = None
        self.frame_index = 0
        self.memory_position_encoding = None
        self.theta = 0.5
        self.trajectory_memory = {}
        self.next_trajectory_id = 1
        self.trajectory_ids = []
        self.last_active_frame = {}
        self.active_trajectory_interval = 10
        self.match_overlap_threshold = match_overlap_threshold
        self.tau_lo = tau_lo
        self.tau_hi = tau_hi
        self.max_batch_size = 400
        self.confirmations_per_trajectory = {}
        self.trajectory_lengths = {}
        self.resize_for_iou = T.Resize((512, 512), antialias=True)

    def _extract_features(self, image):
        backbone_output, original_backbone_output = self.sam.forward_image(image)
        feature_pyramid = backbone_output["backbone_fpn"]
        position_encodings = backbone_output["vision_pos_enc"]
        for feature_index, feature in enumerate(feature_pyramid):
            feature_pyramid[feature_index] = feature.repeat(1, 1, 1, 1).contiguous()
        for feature_index, position_encoding in enumerate(position_encodings):
            position_encodings[feature_index] = position_encoding.repeat(1, 1, 1, 1).contiguous()
        backbone_output = {
            "backbone_fpn": feature_pyramid,
            "vision_pos_enc": position_encodings,
        }
        features = self.sam._prepare_backbone_features(backbone_output)
        _, feature_maps, position_embeddings, feature_sizes = features
        return feature_maps, position_embeddings, feature_sizes, original_backbone_output

    def _mask_to_prompt(self, mask):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        mask_height, mask_width = mask.shape
        original_mask = mask[None, None].float().to(mask.device)
        if mask_height != self.sam.image_size or mask_width != self.sam.image_size:
            resized_mask = F.interpolate(
                original_mask,
                size=(self.sam.image_size, self.sam.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )
            return (resized_mask > 0).float()
        return original_mask

    def compute_features(self, image):
        feature_maps, position_embeddings, feature_sizes, backbone_output = self._extract_features(
            image)
        self.feature_maps = feature_maps
        self.position_embeddings = position_embeddings
        self.feature_sizes = feature_sizes
        return backbone_output

    def process_frame(self, detector_mask_logits, sequence):
        self.frame_index += 1
        tracking_output = None
        if self._resize_to_input is None:
            self._resize_to_input = torch.nn.Upsample(
                size=(sequence["img_size"][-2], sequence["img_size"][-1]),
                mode="bilinear",
                align_corners=False,
            )
        feature_maps = self.feature_maps
        position_embeddings = self.position_embeddings
        feature_sizes = self.feature_sizes
        num_detections = detector_mask_logits.shape[0]
        resized_detector_masks = None
        active_trajectory_ids = [
            trajectory_id for trajectory_id in self.trajectory_ids
            if trajectory_id in self.last_active_frame and self.frame_index -
            self.last_active_frame[trajectory_id] <= self.active_trajectory_interval
        ]
        active_index_by_id = {
            trajectory_id: trajectory_index
            for trajectory_index, trajectory_id in enumerate(active_trajectory_ids)
        }

        if active_trajectory_ids:
            active_memory_by_id = {
                trajectory_id: self.trajectory_memory[trajectory_id]
                for trajectory_id in active_trajectory_ids
            }
            tracking_context = {
                "per_obj_dict": active_memory_by_id,
                "maskmem_pos_enc": self.memory_position_encoding,
                "obj_ids_list": active_trajectory_ids,
            }
            batch_count = (len(active_trajectory_ids) - 1) // self.max_batch_size + 1
            for batch_index in range(batch_count):
                tracking_output = self._track_active_batch(
                    tracking_output,
                    feature_sizes,
                    feature_maps,
                    batch_index,
                    tracking_context,
                    position_embeddings,
                )
            (
                binary_track_masks,
                mask_memory_features,
                positive_pixel_counts,
                propagated_mask_logits,
            ) = self._prepare_propagated_masks(tracking_output)
            confirmation_rates = {
                trajectory_id:
                self.confirmations_per_trajectory[trajectory_id] /
                self.trajectory_lengths[trajectory_id]
                for trajectory_id in self.confirmations_per_trajectory
                if trajectory_id in self.trajectory_lengths
                and self.trajectory_lengths[trajectory_id] != 0
            }
            ranked_active_trajectories = sorted(
                enumerate(active_trajectory_ids),
                key=lambda indexed_trajectory: confirmation_rates[indexed_trajectory[1]],
                reverse=True,
            )
            if num_detections > 0:
                resized_detector_masks = self.resize_for_iou(detector_mask_logits) > 0
                overlaps = mask_iou(
                    self.resize_for_iou(propagated_mask_logits)[:, 0] > 0,
                    resized_detector_masks,
                )
            else:
                overlaps = torch.zeros(
                    (len(propagated_mask_logits), 1)).to(propagated_mask_logits.device)

            for trajectory_index, trajectory_id in ranked_active_trajectories:
                if positive_pixel_counts[trajectory_index] <= 0:
                    continue
                best_detection_index = overlaps[trajectory_index].argmax()
                detector_overlap = (-1 if num_detections == 0 else
                                    overlaps[trajectory_index][best_detection_index].item())
                memory_entry = {
                    "maskmem_features": mask_memory_features[trajectory_index].unsqueeze(0),
                    "bbox": mask2bbox(binary_track_masks[trajectory_index]),
                    "pred_masks": propagated_mask_logits[trajectory_index].unsqueeze(0),
                    "obj_ptr": tracking_output["obj_ptr"][trajectory_index].unsqueeze(0),
                    "is_init": False,
                    "frame_idx": self.frame_index,
                    "samscore": tracking_output["ious"][trajectory_index],
                    "det_overlap": detector_overlap,
                }
                self.trajectory_memory[trajectory_id].append(memory_entry)
                if overlaps[trajectory_index][best_detection_index] > self.match_overlap_threshold:
                    self.confirmations_per_trajectory[trajectory_id] += 1
                self.trajectory_lengths[trajectory_id] += 1

        if tracking_output is not None:
            propagated_mask_logits = self._resize_to_input(tracking_output['pred_masks'])
            if num_detections > 0:
                mask_hypotheses = torch.cat(
                    [detector_mask_logits.unsqueeze(1), propagated_mask_logits])[:, 0]
            else:
                mask_hypotheses = propagated_mask_logits[:, 0]
        else:
            mask_hypotheses = detector_mask_logits

        # mask assignment/correction and merging
        new_track_masks = torch.empty(
            (0, *mask_hypotheses.shape[1:]),
            dtype=torch.bool,
            device=self.device,
        )
        if mask_hypotheses.shape[0] > 0:
            hypothesis_weights = torch.ones(mask_hypotheses.shape[0], device=self.device)
            hypothesis_weights[:num_detections] = self.theta / 2
            assignment_probabilities = torch.ones_like(hypothesis_weights)
            hypothesis_merger = HypothesisMerger(
                hypothesis_weights,
                init_masks=mask_hypotheses > 0,
                device=self.device,
                theta=self.theta,
            )
            (mask_assignments,
             assignment_probabilities) = hypothesis_merger.run(mask_hypotheses,
                                                               assignment_probabilities,
                                                               num_detections,
                                                               iterations=3)
            confirmed_detection_ids = torch.where(
                assignment_probabilities[:num_detections] > 0.6)[0]
            new_track_masks = (mask_assignments.unsqueeze(0) == confirmed_detection_ids[:, None,
                                                                                        None])
            for trajectory_id in self.trajectory_ids:
                if trajectory_id not in active_trajectory_ids:
                    for inactive_memory_entry in self.trajectory_memory[trajectory_id]:
                        inactive_memory_entry["maskmem_features"] = None
                        inactive_memory_entry["obj_ptr"] = None
                    continue

                latest_memory_entry = self.trajectory_memory[trajectory_id][-1]
                trajectory_index = active_index_by_id[trajectory_id]
                merged_track_mask = mask_assignments == trajectory_index + num_detections
                track_probability = assignment_probabilities[num_detections:][trajectory_index]
                # if the assignment probability is above tau_lo (not overwritten by other masks)
                # and there are detections, check for overlap and update tracks
                if track_probability >= self.tau_lo and num_detections > 0:
                    best_detection_overlap = mask_iou(
                        self.resize_for_iou(merged_track_mask.unsqueeze(0)),
                        resized_detector_masks,
                    ).max()
                    mask_area_changed = (track_probability < self.tau_hi
                                         or track_probability > 1 / self.tau_hi)
                    # if assignment probability == change of area mask is significant,
                    # update the track with the new mask, otherwise keep the last mask
                    if mask_area_changed and best_detection_overlap > self.match_overlap_threshold:
                        self._initialize_or_update_tracks(
                            merged_track_mask.unsqueeze(0),
                            feature_sizes,
                            feature_maps,
                            position_embeddings,
                            trajectory_id,
                        )
                    else:
                        latest_memory_entry["pred_masks"] = merged_track_mask.unsqueeze(
                            0).unsqueeze(0)
                        latest_memory_entry["bbox"] = mask2bbox(
                            self._resize_to_input(merged_track_mask.unsqueeze(0).unsqueeze(0))[0])
                elif (track_probability < self.tau_lo
                      and latest_memory_entry["frame_idx"] == self.frame_index):
                    self.trajectory_memory[trajectory_id].pop()

        # init new tracks for detections that are not "overwritten" by other active trackers
        self._initialize_or_update_tracks(
            new_track_masks,
            feature_sizes,
            feature_maps,
            position_embeddings,
        )
        frame_predictions = []
        for trajectory_id in self.trajectory_ids:
            for memory_entry in self.trajectory_memory[trajectory_id]:
                if (memory_entry["frame_idx"] == self.frame_index
                        and memory_entry["bbox"] is not None):
                    self.last_active_frame[trajectory_id] = self.frame_index
                    sam_score = (memory_entry["samscore"].max().item()
                                 if "samscore" in memory_entry else -1)
                    frame_predictions.append({
                        "bbox":
                        memory_entry["bbox"].cpu(),
                        "score":
                        sam_score,
                        "trajectory_id":
                        trajectory_id,
                        "exemplar_candidate": ("det_overlap" in memory_entry
                                               and memory_entry["det_overlap"] >= self.tau_hi),
                    })
                    break

        # Tracker memory update
        if active_trajectory_ids:
            for _, trajectory_id in ranked_active_trajectories:
                memory = self.trajectory_memory[trajectory_id]
                if len(memory) <= self.sam.num_maskmem:
                    continue
                for memory_index, memory_entry in enumerate(memory):
                    memory_entry["pred_masks"] = None
                    if not memory_entry["is_init"]:
                        del memory[memory_index]
                        break

        return frame_predictions, self.confirmations_per_trajectory, self.trajectory_lengths

    def _initialize_or_update_tracks(
        self,
        initial_masks,
        feature_sizes,
        feature_maps,
        position_embeddings,
        trajectory_id=None,
    ):
        for mask_index in range(initial_masks.shape[0]):
            initial_mask = initial_masks[mask_index]
            mask_prompt = self._mask_to_prompt(initial_mask)
            self.sam.use_mask_input_as_output_without_sam = True
            tracking_context = {"per_obj_dict": {}, "maskmem_pos_enc": None}
            tracking_output = self.sam.track_step(
                frame_idx=self.frame_index,
                is_init_cond_frame=True,
                current_vision_feats=feature_maps,
                current_vision_pos_embeds=position_embeddings,
                feat_sizes=feature_sizes,
                point_inputs=None,
                mask_inputs=mask_prompt,
                output_dict=tracking_context,
                num_frames=None,
                track_in_reverse=False,
                run_mem_encoder=True,
                prev_sam_mask_logits=None,
            )
            initial_mask_logits = tracking_output["pred_masks"]
            resized_mask_logits = self._resize_to_input(initial_mask_logits)
            binary_mask = (resized_mask_logits[:, 0, :, :] > 0).to(dtype=torch.uint8)
            mask_memory_features = tracking_output["maskmem_features"].to(
                dtype=torch.bfloat16,
                device=self.device,
                non_blocking=True,
            )
            if self.memory_position_encoding is None:
                self.memory_position_encoding = tracking_output["maskmem_pos_enc"][0][0:1].clone()
                self.memory_position_encoding = self.memory_position_encoding.to(torch.bfloat16).to(
                    self.device)
            memory_entry = {
                "maskmem_features": mask_memory_features,
                "pred_masks": initial_mask_logits,
                "obj_ptr": tracking_output["obj_ptr"],
                "is_init": True,
                "frame_idx": self.frame_index,
                "bbox": mask2bbox(binary_mask[0]),
            }
            target_trajectory_id = trajectory_id
            if target_trajectory_id is None:
                target_trajectory_id = self.next_trajectory_id
                self.next_trajectory_id += 1
                self.confirmations_per_trajectory[target_trajectory_id] = 1
                self.trajectory_lengths[target_trajectory_id] = 1
                self.trajectory_ids.append(target_trajectory_id)

            if target_trajectory_id not in self.trajectory_memory:
                self.trajectory_memory[target_trajectory_id] = [memory_entry]
                continue

            memory = self.trajectory_memory[target_trajectory_id]
            initialization_entries = [
                (memory_index, stored_entry["frame_idx"])
                for memory_index, stored_entry in enumerate(memory[1:], start=1)
                if stored_entry["is_init"]
            ]
            memory[-1] = memory_entry
            if len(initialization_entries) >= 3:
                oldest_initialization_index = min(
                    initialization_entries,
                    key=lambda indexed_frame: indexed_frame[1],
                )[0]
                del memory[oldest_initialization_index]

    def _prepare_propagated_masks(self, tracking_output):
        propagated_mask_logits = tracking_output["pred_masks"]
        if self.fill_hole_area > 0:
            propagated_mask_logits = fill_holes_in_mask_scores(
                propagated_mask_logits,
                self.fill_hole_area,
            )
        resized_mask_logits = self._resize_to_input(propagated_mask_logits)
        binary_masks = (resized_mask_logits[:, 0, :, :] > 0).to(torch.uint8)
        positive_pixel_counts = binary_masks.flatten(1).sum(dim=1)
        mask_memory_features = tracking_output["maskmem_features"].to(torch.bfloat16)
        return (
            binary_masks,
            mask_memory_features,
            positive_pixel_counts,
            propagated_mask_logits,
        )

    def _track_active_batch(
        self,
        tracking_output,
        feature_sizes,
        feature_maps,
        batch_index,
        tracking_context,
        position_embeddings,
    ):
        start_index = batch_index * self.max_batch_size
        end_index = min(
            len(tracking_context["obj_ids_list"]),
            start_index + self.max_batch_size,
        )
        trajectory_ids = tracking_context["obj_ids_list"][start_index:end_index]
        batch_memory = {
            trajectory_id: tracking_context["per_obj_dict"][trajectory_id]
            for trajectory_id in trajectory_ids
        }
        batch_context = {
            "per_obj_dict": batch_memory,
            "maskmem_pos_enc": tracking_context["maskmem_pos_enc"],
            "obj_ids_list": trajectory_ids,
        }
        batch_output = self.sam.track_step(
            frame_idx=self.frame_index,
            is_init_cond_frame=False,
            current_vision_feats=feature_maps,
            current_vision_pos_embeds=position_embeddings,
            feat_sizes=feature_sizes,
            point_inputs=None,
            mask_inputs=None,
            output_dict=batch_context,
            num_frames=None,
            track_in_reverse=False,
            run_mem_encoder=True,
            prev_sam_mask_logits=None,
        )
        if tracking_output is None:
            return batch_output
        for key in ("pred_masks", "obj_ptr", "maskmem_features", "ious"):
            tracking_output[key] = torch.cat([tracking_output[key], batch_output[key]], dim=0)
        return tracking_output
