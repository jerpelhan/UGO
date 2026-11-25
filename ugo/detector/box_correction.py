# SAM2 box refinement adapted from GECO2 commit ef6954a59d22620b9a3052f8500530a799c0379b:
# https://github.com/jerpelhan/GECO2/blob/ef6954a59d22620b9a3052f8500530a799c0379b/models/box_corr.py

from torch import nn

from .layers import MaskProcessor


class BoxCorrection(nn.Module):

    def __init__(
        self,
        reduction,
        image_size,
        emb_dim,
        sam_checkpoint_path,
    ):
        super().__init__()
        self.sam_mask = MaskProcessor(
            emb_dim,
            image_size,
            reduction,
            sam_checkpoint_path,
        )

    def forward(self, feats, outputs, sequence):
        masks, ious, corrected_bboxes = self.sam_mask(
            feats,
            outputs,
            sequence,
        )
        for index in range(len(outputs)):
            outputs[index]["scores"] = ious[index]
            outputs[index]["pred_boxes"] = corrected_bboxes[index].unsqueeze(0)
        return outputs, masks
