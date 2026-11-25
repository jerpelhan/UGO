import torch


class MOTDetection:

    def __init__(
        self,
        frame_id,
        trajectory_id,
        bbox,
        conf,
        x3d,
        y3d,
        z3d,
    ):
        self.frame_id = frame_id
        self.trajectory_id = trajectory_id
        self.bbox = bbox
        self.conf = conf
        self.x3d = x3d
        self.y3d = y3d
        self.z3d = z3d

    def to_mot_line(self):
        return "%d,%d,%d,%d,%d,%d,%.3f,%.1f,%.1f,%.1f\n" % (
            self.frame_id,
            self.trajectory_id,
            int(self.bbox[0]),
            int(self.bbox[1]),
            int(self.bbox[2]),
            int(self.bbox[3]),
            self.conf,
            self.x3d,
            self.y3d,
            self.z3d,
        )


def mask2bbox(mask: torch.Tensor):
    mask = mask.squeeze()
    assert mask.dim() == 2, "Input mask must be a single mask"
    if not mask.any():
        return None
    x = torch.where(mask.sum(0) > 0)[0]
    y = torch.where(mask.sum(1) > 0)[0]
    x0 = x.min() - 1
    y0 = y.min() - 1
    x1 = x.max() + 1
    y1 = y.max() + 1
    return torch.stack([x0, y0, x1 - x0, y1 - y0])


def mask_iou(masks1: torch.Tensor, masks2: torch.Tensor):
    masks1 = (masks1 != 0).float()
    masks2 = (masks2 != 0).float()
    masks1_flat = masks1.reshape(masks1.shape[0], -1)
    masks2_flat = masks2.reshape(masks2.shape[0], -1)
    intersection = torch.matmul(masks1_flat, masks2_flat.T)
    area1 = masks1_flat.sum(dim=1, keepdim=True)
    area2 = masks2_flat.sum(dim=1, keepdim=True).T
    union = area1 + area2 - intersection
    return intersection / union.clamp(min=1e-6)
