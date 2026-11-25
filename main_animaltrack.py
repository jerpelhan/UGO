import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ugo.datasets import AnimalTrackSequenceDataset
from ugo.detector import build_geco2_detector
from ugo.mot_format import MOTDetection
from ugo.tracker import UGOTracker

REPOSITORY_ROOT = Path(__file__).resolve().parent


def run_sequence(
    sequence,
    detector,
    sam_checkpoint,
    sam_config,
    output_path,
    device,
):
    tracker = UGOTracker(
        sam_checkpoint=sam_checkpoint,
        sam_config=sam_config,
        device=device,
        tau_lo=0.2,
        tau_hi=0.8,
        match_overlap_threshold=0.6,
    )
    predictions_by_frame = []
    prototypes = None
    confirmations = {}
    track_lengths = {}
    for frame_index, image in tqdm(
            enumerate(sequence["images"]),
            total=len(sequence["images"]),
            desc=sequence["sequence_name"],
    ):
        image = image.to(device).unsqueeze(0)
        features = tracker.compute_features(image)
        if frame_index == 0:
            prototypes = detector.extract_prototypes(features, sequence)
        _, _, _, mask_logits = detector(
            features,
            sequence,
            queries=prototypes,
        )
        predictions, confirmations, track_lengths = tracker.process_frame(
            mask_logits,
            sequence,
        )
        predictions_by_frame.append(predictions)
        del features, image
    confirmation_rates = {
        trajectory_id: confirmations[trajectory_id] / track_lengths[trajectory_id]
        for trajectory_id in confirmations
        if trajectory_id in track_lengths and track_lengths[trajectory_id] != 0
    }
    lines = []
    for frame_index, predictions in enumerate(predictions_by_frame, start=1):
        for prediction in predictions:
            trajectory_id = prediction["trajectory_id"]
            if confirmation_rates.get(trajectory_id, 0) < 0.3:
                continue
            detection = MOTDetection(
                frame_index,
                trajectory_id,
                prediction["bbox"],
                prediction["score"],
                0,
                0,
                0,
            )
            lines.append(detection.to_mot_line())
    try:
        with output_path.open("x") as handle:
            handle.writelines(lines)
    except FileExistsError:
        print(f"Skipping {sequence['sequence_name']} because {output_path} exists")
    del tracker, predictions_by_frame
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def main(args):
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("UGO reproduction requires a CUDA device")
    torch.cuda.set_device(device)
    detector = build_geco2_detector(
        args.geco2_checkpoint,
        args.sam2_checkpoint,
        device,
    )
    sequence_name = args.sequence if args.sequence is not None else "test"
    dataset = AnimalTrackSequenceDataset(
        args.data_root,
        sequence_name=sequence_name,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sam_config = REPOSITORY_ROOT / "configs" / "sam2.1_hiera_l_ugo.yaml"
    with torch.cuda.amp.autocast():
        for index, sequence_path in enumerate(dataset.sequence_paths):
            sequence_name = Path(sequence_path).name
            output_path = output_dir / f"{sequence_name}.txt"
            if output_path.exists():
                annotations = dataset._load_annotations(
                    Path(sequence_path) / "gt" / "gt.txt"
                )
                dataset._select_exemplars(annotations)
                print(f"Skipping {sequence_name} because {output_path} exists")
                continue
            sequence = dataset[index]
            run_sequence(
                sequence,
                detector,
                args.sam2_checkpoint,
                sam_config,
                output_path,
                device,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="UGO tracking on AnimalTrack")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--geco2-checkpoint",
        default=str(REPOSITORY_ROOT / "weights" / "UGO.pth"),
    )
    parser.add_argument(
        "--sam2-checkpoint",
        default=str(REPOSITORY_ROOT / "weights" / "sam2.1_hiera_large.pt"),
    )
    parser.add_argument(
        "--output-dir",
        default="results/AnimalTrack-train/UGO/data",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
