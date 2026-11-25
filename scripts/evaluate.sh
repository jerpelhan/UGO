#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <GMOT|AnimalTrack> <gt-root>" >&2
    exit 2
fi

benchmark=$1
gt_root=$2
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tracker_root="$repo_root/results"
run_name=UGO

case "$benchmark" in
    GMOT|AnimalTrack) ;;
    *) echo "Benchmark must be GMOT or AnimalTrack" >&2; exit 2 ;;
esac

split="${benchmark}-train"
trackeval="$repo_root/third_party/TrackEval/scripts/run_mot_challenge.py"
python -c 'import numpy as np, runpy, sys; np.bool = bool; np.float = float; np.int = int; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")' "$trackeval" \
    --GT_FOLDER "$gt_root" \
    --TRACKERS_FOLDER "$tracker_root" \
    --BENCHMARK "$benchmark" \
    --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL "$run_name" \
    --METRICS HOTA CLEAR Identity VACE \
    --THRESHOLD 0.5 \
    --DO_PREPROC False \
    --USE_PARALLEL False \
    --NUM_PARALLEL_CORES 1
