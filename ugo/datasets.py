import glob
import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

GMOT40_SEQUENCE_NAMES = (
    "airplane-0",
    "airplane-1",
    "airplane-2",
    "airplane-3",
    "ball-0",
    "ball-1",
    "ball-2",
    "ball-3",
    "balloon-0",
    "balloon-1",
    "balloon-2",
    "balloon-3",
    "bird-0",
    "bird-1",
    "bird-2",
    "bird-3",
    "boat-0",
    "boat-1",
    "boat-2",
    "boat-3",
    "car-0",
    "car-1",
    "car-2",
    "car-3",
    "fish-0",
    "fish-1",
    "fish-2",
    "fish-3",
    "insect-0",
    "insect-1",
    "insect-2",
    "insect-3",
    "person-0",
    "person-1",
    "person-2",
    "person-3",
    "stock-0",
    "stock-1",
    "stock-2",
    "stock-3",
)


class _LazyImageSequence:

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __iter__(self):
        for image_path in self.image_paths:
            image = T.ToTensor()(Image.open(image_path).convert("RGB"))
            yield self.transform(image)[0]


class GMOTSequenceDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        sequence_name: str = "test",
    ):
        root_dir = Path(root_dir)
        image_root = root_dir / "GenericMOT_JPEG_Sequence"
        if image_root.is_dir():
            self.prototype_root = root_dir / "template_box_by_seq_for_global_track"
        else:
            image_root = root_dir
            self.prototype_root = root_dir.parent / "template_box_by_seq_for_global_track"
        if not self.prototype_root.is_dir():
            raise FileNotFoundError(
                f"Missing GMOT40 template boxes: {self.prototype_root}"
            )
        if sequence_name == "test":
            missing = [
                name for name in GMOT40_SEQUENCE_NAMES
                if not (image_root / name).is_dir()
            ]
            if missing:
                raise FileNotFoundError("Missing GMOT40 sequences: " + ", ".join(missing))
            self.sequence_paths = [image_root / name for name in GMOT40_SEQUENCE_NAMES]
        else:
            if sequence_name not in GMOT40_SEQUENCE_NAMES:
                raise ValueError(f"Unknown GMOT40 sequence: {sequence_name}")
            self.sequence_paths = [image_root / sequence_name]
        self._img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self._img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self.sequence_paths)

    def _resize(self, image: torch.Tensor, size: int = 1024):
        _, original_height, original_width = image.shape
        scale_y = size / original_height
        scale_x = size / original_width
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )[0]
        image -= self._img_mean
        image /= self._img_std
        return image, (scale_x, scale_y)

    def __getitem__(self, index):
        sequence_path = self.sequence_paths[index]
        sequence_name = os.path.basename(sequence_path)
        image_paths = sorted(glob.glob(os.path.join(sequence_path, "img1", "*.jpg")))
        if not image_paths:
            raise RuntimeError(f"No images found in {os.path.join(sequence_path, 'img1')}")

        first_image = T.ToTensor()(Image.open(image_paths[0]).convert("RGB"))
        original_image_size = first_image.shape[1:]
        original_height, original_width = original_image_size
        scaling_factor = (1024 / original_width, 1024 / original_height)
        images = _LazyImageSequence(image_paths, self._resize)

        exemplar_path = os.path.join(
            self.prototype_root,
            f"{sequence_name}.txt",
        )
        exemplars = self._load_annotations(exemplar_path, scaling_factor)

        return {
            "images": images,
            "sequence_name": sequence_name,
            "exemplar": exemplars,
            "scaling_factor": scaling_factor,
            "img_size": original_image_size,
        }

    @staticmethod
    def _load_annotations(gt_path, scaling_factor=(1, 1)):
        annotations = []
        boxes = []
        frame = 0
        with open(gt_path) as handle:
            for line in handle:
                fields = line.strip().split(",")
                if len(fields) < 7:
                    continue
                frame_id = int(fields[0])
                if frame_id > frame:
                    frame = frame_id
                    scale_x, scale_y = scaling_factor
                    annotations.append(
                        torch.tensor(boxes) * torch.tensor([scale_x, scale_y, scale_x, scale_y]))
                    boxes = []
                boxes.append(list(map(float, fields[2:6])))
        scale_x, scale_y = scaling_factor
        annotations.append(torch.tensor(boxes) * torch.tensor([scale_x, scale_y, scale_x, scale_y]))
        return annotations


class AnimalTrackSequenceDataset(Dataset):

    def __init__(
        self,
        root_dir: str,
        sequence_name: str = "test",
    ):
        if sequence_name == "test":
            with open(os.path.join(root_dir, "videos_test.txt")) as handle:
                sequence_names = [
                    os.path.splitext(os.path.basename(line.strip()))[0]
                    for line in handle if line.strip()
                ]
            self.sequence_paths = [os.path.join(root_dir, name) for name in sequence_names]
        else:
            self.sequence_paths = [os.path.join(root_dir, sequence_name)]
        self._img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self._img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self.sequence_paths)

    def _resize(self, image: torch.Tensor, size: int = 1024):
        _, original_height, original_width = image.shape
        scale_y = size / original_height
        scale_x = size / original_width
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )[0]
        image = T.Normalize(mean=self._img_mean, std=self._img_std)(image)
        return image, (scale_x, scale_y)

    def __getitem__(self, index):
        sequence_path = self.sequence_paths[index]
        sequence_name = os.path.basename(sequence_path)
        image_paths = sorted(glob.glob(os.path.join(sequence_path, "img1", "*.jpg")))
        if not image_paths:
            raise RuntimeError(f"No images found in {os.path.join(sequence_path, 'img1')}")

        images = []
        original_image_size = None
        for image_path in image_paths:
            image = T.ToTensor()(Image.open(image_path).convert("RGB"))
            if original_image_size is None:
                original_image_size = image.shape[1:]
            image, scaling_factor = self._resize(image)
            images.append(image)

        annotations = self._load_annotations(os.path.join(sequence_path, "gt", "gt.txt"),
                                             scaling_factor)
        exemplars = self._select_exemplars(annotations)

        return {
            "images": images,
            "sequence_name": sequence_name,
            "exemplar": exemplars,
            "scaling_factor": scaling_factor,
            "img_size": original_image_size,
        }

    @staticmethod
    def _select_exemplars(annotations):
        if not annotations or len(annotations[0]) == 0:
            return torch.zeros((1, 4), dtype=torch.float32)

        candidates = annotations[0].clone()
        indices = torch.randperm(candidates.shape[0])[:5]
        return candidates[indices].to(torch.float32).unsqueeze(0)

    @staticmethod
    def _load_annotations(gt_path, scaling_factor=(1, 1)):
        annotations = []
        boxes = []
        frame = 1
        scale_x, scale_y = scaling_factor
        scale = torch.tensor([scale_x, scale_y, scale_x, scale_y], dtype=torch.float32)
        with open(gt_path) as handle:
            for line in handle:
                fields = line.strip().split(",")
                if len(fields) < 6:
                    continue
                try:
                    frame_id = int(float(fields[0]))
                except ValueError:
                    continue
                if frame_id > frame:
                    frame = frame_id
                    if boxes:
                        annotations.append(torch.tensor(boxes, dtype=torch.float32) * scale)
                        boxes = []
                boxes.append(list(map(float, fields[2:6])))
        if boxes:
            annotations.append(torch.tensor(boxes, dtype=torch.float32) * scale)
        return annotations
