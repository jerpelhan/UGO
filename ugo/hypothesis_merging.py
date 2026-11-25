import torch
from torchvision import transforms as T

from .mot_format import mask_iou


class HypothesisMerger:

    def __init__(
        self,
        weights,
        bg_logit=1e-5,
        device="cuda",
        init_masks=None,
        theta=0.5,
    ):
        self.hypothesis_count = len(weights)
        self.weights = weights.to(device)
        self.background_score = torch.tensor(bg_logit, device=device)
        self.resize_for_iou = T.Resize((512, 512))
        self.original_areas = init_masks.sum(dim=(1, 2), dtype=torch.float32).to(device) + 1e-5
        self.theta = theta

    def _assign(self, logits, probabilities, track_overlaps):
        hypotheses, height, width = logits.shape
        assert hypotheses == self.hypothesis_count
        if track_overlaps is not None:
            weights = self.weights.detach().clone()
            weights[-track_overlaps.shape[0]:] *= track_overlaps
        else:
            weights = self.weights
        weighted_logits = weights[:, None, None] * logits * probabilities[:, None, None]
        best_scores, assignments = weighted_logits.max(dim=0)
        assignments[best_scores < self.background_score] = hypotheses
        assigned_areas = torch.bincount(
            assignments.flatten(),
            minlength=hypotheses + 1,
        )[:hypotheses].to(self.original_areas.dtype)
        updated_probabilities = assigned_areas / self.original_areas
        return assignments, updated_probabilities

    def run(self, logits, probabilities, detection_count, iterations=3):
        detections = logits[:detection_count]
        resized_detections = (self.resize_for_iou(detections) > 0 if detection_count > 0 else None)
        assignments = None
        for _ in range(iterations):
            if assignments is None:
                tracks = logits[detection_count:]
            else:
                background_id = assignments.max()
                instance_ids = assignments.unique()
                instance_ids = instance_ids[(instance_ids >= detection_count)
                                            & (instance_ids != background_id)]
                tracks = assignments.unsqueeze(0) == instance_ids[:, None, None]
            if len(tracks) > 0 and detection_count > 0:
                track_overlaps = (mask_iou(
                    self.resize_for_iou(tracks) > 0,
                    resized_detections,
                ).max(dim=-1).values + self.theta)
            else:
                track_overlaps = None
            assignments, probabilities = self._assign(logits, probabilities, track_overlaps)
        assignments[assignments == assignments.max()] = -1
        return assignments, probabilities
