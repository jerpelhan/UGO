"""One-pass ScienceCount evaluation with the refactored UGO detector and tracker.

The local VideoCount layout contains the ScienceCount Penguins and Crystals subsets.
Each video has a separate exemplar image and an ``exemplars.json`` file. For
the paper's ScienceCount results, the prediction is the number of distinct
tracks; the confirmation threshold is 0.0. The historical script prints a
threshold sweep but aggregates its final error at 0.1 instead.
"""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


ROOT = Path(__file__).resolve().parent
SUBSETS = {
    "Penguins": ("penguins-count-gt.json", "penguin"),
    "Crystals": ("crystals-count-gt.json", "white crystal"),
}
IMAGE_SIZE = 1024
MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]


def resize_and_pad(image_path):
    with Image.open(image_path) as source:
        image = T.ToTensor()(source.convert("RGB"))
    _, height, width = image.shape
    scale = IMAGE_SIZE / max(height, width)
    image = F.interpolate(
        image.unsqueeze(0), scale_factor=scale, mode="bilinear", align_corners=False
    )[0]
    image = F.pad(
        image,
        (0, IMAGE_SIZE - image.shape[2], 0, IMAGE_SIZE - image.shape[1]),
        mode="constant",
        value=0,
    )
    image = T.Normalize(mean=MEAN, std=STD)(image)
    return image, (height, width), scale


def video_inputs(data_root, subset, sequence):
    subset_root = data_root / subset
    frame_paths = sorted((subset_root / "frames" / sequence).glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No JPG frames for {subset}/{sequence}")
    exemplar_root = subset_root / "exemplars" / sequence
    exemplar_image = next(
        (path for suffix in ("jpg", "png")
         if (path := exemplar_root / f"exemplar_image.{suffix}").is_file()),
        None,
    )
    if exemplar_image is None:
        raise FileNotFoundError(f"No exemplar image for {subset}/{sequence}")
    with (exemplar_root / "exemplars.json").open() as handle:
        prompts = json.load(handle)["exemplars"]
    boxes = [
        [prompt[0], prompt[1], prompt[3], prompt[4]]
        for prompt in prompts
        if len(prompt) >= 6 and prompt[2] == 2.0 and prompt[5] == 3.0
    ]
    if not boxes:
        raise ValueError(f"No valid exemplar boxes for {subset}/{sequence}")
    return frame_paths, exemplar_image, boxes


def ground_truth(data_root, subset):
    gt_file, label = SUBSETS[subset]
    with (data_root / subset / "anno" / gt_file).open() as handle:
        annotations = json.load(handle)
    return {sequence: int(classes[label]) for sequence, classes in annotations.items()}


@torch.inference_mode()
def infer_video(
    frame_paths,
    exemplar_image,
    boxes,
    detector,
    sam_checkpoint,
    sam_config,
    device,
    confirmation_threshold,
):
    from ugo.tracker import UGOTracker

    tracker = UGOTracker(
        sam_checkpoint=sam_checkpoint,
        sam_config=sam_config,
        device=device,
        tau_lo=0.2,
        tau_hi=0.8,
        match_overlap_threshold=0.6,
    )
    exemplar, _, scale = resize_and_pad(exemplar_image)
    exemplar_boxes = torch.tensor(boxes, dtype=torch.float32)
    exemplar_boxes *= torch.tensor([scale, scale, scale, scale])
    exemplar_boxes[:, 2] -= exemplar_boxes[:, 0]
    exemplar_boxes[:, 3] -= exemplar_boxes[:, 1]
    exemplar_sequence = {"exemplar": exemplar_boxes.unsqueeze(0)}

    with torch.cuda.amp.autocast():
        features = tracker.compute_features(exemplar.to(device).unsqueeze(0))
        prototypes = detector.extract_prototypes(features, exemplar_sequence)
        del features, exemplar

        for frame_index, frame_path in enumerate(frame_paths, start=1):
            image, image_size, _ = resize_and_pad(frame_path)
            sequence = {"img_size": image_size}
            image = image.to(device).unsqueeze(0)
            features = tracker.compute_features(image)
            _, _, _, mask_logits = detector(features, sequence, queries=prototypes)
            tracker.process_frame(mask_logits, sequence)
            del image, features, mask_logits
            if frame_index % 10 == 0 or frame_index == len(frame_paths):
                print(f"  {frame_index}/{len(frame_paths)} frames", flush=True)

    rates = {
        trajectory_id: confirmed / tracker.trajectory_lengths[trajectory_id]
        for trajectory_id, confirmed in tracker.confirmations_per_trajectory.items()
        if tracker.trajectory_lengths.get(trajectory_id, 0) > 0
    }
    count = sum(rate >= confirmation_threshold for rate in rates.values())
    del tracker
    gc.collect()
    torch.cuda.empty_cache()
    return count, len(rates)


def save_json_exclusive(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


def evaluate(args):
    data_root = args.data_root.resolve()
    subsets = tuple(SUBSETS) if args.subset == "all" else (args.subset,)
    manifest = {}
    for subset in subsets:
        gt = ground_truth(data_root, subset)
        selected = [args.sequence] if args.sequence else list(gt)
        for sequence in selected:
            if sequence not in gt:
                raise ValueError(f"{sequence} is not a {subset} sequence")
            frames, exemplar_image, boxes = video_inputs(data_root, subset, sequence)
            manifest[(subset, sequence)] = (frames, exemplar_image, boxes, gt[sequence])

    if args.validate_data:
        for (subset, sequence), (frames, exemplar_image, boxes, count) in manifest.items():
            print(f"{subset}/{sequence}: frames={len(frames)} exemplars={len(boxes)} "
                  f"ground_truth={count} exemplar_image={exemplar_image}")
        return

    if args.device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("ScienceCount inference requires a visible CUDA GPU")
    torch.cuda.set_device(args.device)
    from ugo.detector import build_geco2_detector

    detector = build_geco2_detector(
        args.geco2_checkpoint, args.sam2_checkpoint, args.device
    )
    for subset in subsets:
        rows = []
        for (row_subset, sequence), (frames, exemplar_image, boxes, gt) in manifest.items():
            if row_subset != subset:
                continue
            output = args.output_dir / subset / f"{sequence}.json"
            if output.exists():
                with output.open() as handle:
                    row = json.load(handle)
                expected = {
                    "subset": subset,
                    "sequence": sequence,
                    "ground_truth": gt,
                    "confirmation_threshold": args.confirmation_threshold,
                    "geco2_checkpoint": str(args.geco2_checkpoint),
                    "sam2_checkpoint": str(args.sam2_checkpoint),
                }
                if any(row.get(key) != value for key, value in expected.items()):
                    raise RuntimeError(f"Existing result has different settings: {output}")
                print(f"Skipping existing {subset}/{sequence}: {output}", flush=True)
            else:
                print(f"Processing {subset}/{sequence} ({len(frames)} frames, "
                      f"{len(boxes)} exemplars)", flush=True)
                count, num_tracks = infer_video(
                    frames, exemplar_image, boxes, detector,
                    args.sam2_checkpoint, args.sam_config,
                    args.device, args.confirmation_threshold,
                )
                row = {
                    "subset": subset,
                    "sequence": sequence,
                    "frames": len(frames),
                    "exemplars": len(boxes),
                    "ground_truth": gt,
                    "predicted_count": count,
                    "absolute_error": abs(count - gt),
                    "num_trajectories": num_tracks,
                    "confirmation_threshold": args.confirmation_threshold,
                    "geco2_checkpoint": str(args.geco2_checkpoint),
                    "sam2_checkpoint": str(args.sam2_checkpoint),
                }
                save_json_exclusive(output, row)
                print(f"  count={count} gt={gt} error={row['absolute_error']}", flush=True)
            rows.append(row)

        errors = [row["absolute_error"] for row in rows]
        summary = {
            "subset": subset,
            "scope": "full" if args.sequence is None else "single-video",
            "videos": len(rows),
            "mae": sum(errors) / len(errors),
            "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
            "results": rows,
        }
        print(json.dumps({key: value for key, value in summary.items() if key != "results"},
                         sort_keys=True), flush=True)
        if args.sequence is None:
            summary_path = args.output_dir / f"{subset}_summary.json"
            if summary_path.exists():
                with summary_path.open() as handle:
                    if json.load(handle) != summary:
                        raise RuntimeError(f"Existing summary differs: {summary_path}")
            else:
                save_json_exclusive(summary_path, summary)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/storage/datasets/VideoCount"))
    parser.add_argument("--subset", choices=(*SUBSETS, "all"), default="all")
    parser.add_argument("--sequence", help="Run one video for a smoke test")
    parser.add_argument("--confirmation-threshold", type=float, default=0.0)
    parser.add_argument("--geco2-checkpoint", type=Path,
                        default=ROOT / "weights/UGO.pth")
    parser.add_argument("--sam2-checkpoint", type=Path,
                        default=ROOT / "weights/sam2.1_hiera_large.pt")
    parser.add_argument("--sam-config", type=Path,
                        default=ROOT / "configs/sam2.1_hiera_l_ugo.yaml")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/ScienceCount/UGO")
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"))
    parser.add_argument("--validate-data", action="store_true",
                        help="Check data and exemplars without loading models or using a GPU")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
