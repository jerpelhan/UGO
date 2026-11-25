import argparse
import gc
from pathlib import Path

import torch
from tqdm import tqdm

from ugo.datasets import GMOTSequenceDataset
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
        prototypes = update_prototype_bank(
            features,
            confirmations,
            detector,
            device,
            track_lengths,
            predictions,
            prototypes,
            sequence,
        )
        predictions_by_frame.append(predictions)
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
                confirmation_rates[trajectory_id],
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
    dataset = GMOTSequenceDataset(
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


def update_prototype_bank(
    features,
    confirmations,
    detector,
    device,
    track_lengths,
    predictions,
    prototypes,
    sequence,
):
    confirmation_rates = {
        trajectory_id: confirmations[trajectory_id] / track_lengths[trajectory_id]
        for trajectory_id in confirmations
        if trajectory_id in track_lengths and track_lengths[trajectory_id] != 0
    }
    confirmation_rates = {
        trajectory_id: rate
        for trajectory_id, rate in confirmation_rates.items() if track_lengths[trajectory_id] >= 10
    }
    candidates = []
    for trajectory_id, _ in sorted(confirmation_rates.items(),
                                   key=lambda item: item[1],
                                   reverse=True):
        for prediction in predictions:
            if (prediction["trajectory_id"] == trajectory_id
                    and prediction.get("exemplar_candidate", False)):
                candidates.append(prediction["bbox"].unsqueeze(0))
                break
        if candidates:
            break
    if not candidates:
        return prototypes

    scale_x, scale_y = sequence["scaling_factor"]
    sequence["exemplar"] = torch.cat(candidates, dim=0).unsqueeze(0).to(device) * torch.tensor(
        [scale_x, scale_y, scale_x, scale_y], device=device).view(1, 1, 4)
    new_prototypes = detector.extract_prototypes(features, sequence)
    prototype_embeddings, hq_prototype_embeddings = prototypes
    new_embeddings, new_hq_embeddings = new_prototypes
    original_embeddings = prototype_embeddings[:, :2]
    original_hq_l1, original_hq_l2 = hq_prototype_embeddings
    new_hq_l1, new_hq_l2 = new_hq_embeddings
    adaptive_slots = 3 * 2
    if prototype_embeddings.shape[1] > adaptive_slots:
        hq_l1 = torch.cat(
            [
                original_hq_l1[:, :2],
                original_hq_l1[:, -adaptive_slots + 2:],
                new_hq_l1,
            ],
            dim=1,
        )
        hq_l2 = torch.cat(
            [
                original_hq_l2[:, :2],
                original_hq_l1[:, -adaptive_slots + 2:],
                new_hq_l2,
            ],
            dim=1,
        )
        hq_prototype_embeddings = (hq_l1, hq_l2)
        prototype_embeddings = torch.cat(
            [
                original_embeddings,
                prototype_embeddings[:, -adaptive_slots + 2:],
                new_embeddings,
            ],
            dim=1,
        )
    else:
        prototype_embeddings = torch.cat([prototype_embeddings, new_embeddings], dim=1)
        hq_prototype_embeddings = (
            torch.cat([original_hq_l1, new_hq_l1], dim=1),
            torch.cat([original_hq_l2, new_hq_l2], dim=1),
        )
    return prototype_embeddings, hq_prototype_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="UGO tracking on GMOT40")
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
        default="results/GMOT-train/UGO/data",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
