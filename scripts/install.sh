#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v conda >/dev/null 2>&1; then
    if [[ ! -x "${CONDA_EXE:-}" ]]; then
        echo "Conda is required" >&2
        exit 1
    fi
    conda() { "$CONDA_EXE" "$@"; }
fi

git submodule update --init --recursive

if conda run -n ugo true >/dev/null 2>&1; then
    conda install --yes -n ugo \
        --override-channels -c nvidia/label/cuda-12.6.3 -c defaults \
        python=3.10.14 cuda-nvcc_linux-64=12.6.85 pip wheel
else
    conda create --yes -n ugo \
        --override-channels -c nvidia/label/cuda-12.6.3 -c defaults \
        python=3.10.14 cuda-nvcc_linux-64=12.6.85 pip wheel
fi

conda run --no-capture-output -n ugo python -m pip install -r requirements.txt

ugo_prefix="$(conda run -n ugo python -c 'import sys; print(sys.prefix)' | grep '^/' | tail -n 1)"
ugo_site="$(conda run -n ugo python -c 'import site; print(site.getsitepackages()[0])' | grep '^/' | tail -n 1)"
cuda_packages="$ugo_site/nvidia"
cuda_includes="$(find "$cuda_packages" -type d -name include -print | paste -sd: -)"
cuda_libraries="$(find "$cuda_packages" -type d -name lib -print | paste -sd: -)"
cuda_arch="$(conda run -n ugo python -c 'import torch; a = torch.cuda.get_device_capability(); print(f"{a[0]}.{a[1]}")' | grep -E '^[0-9]+\.[0-9]+$' | tail -n 1)"

cuda_build() {
    CUDA_HOME="$ugo_prefix" \
    CPATH="$cuda_includes${CPATH:+:$CPATH}" \
    LIBRARY_PATH="$cuda_libraries${LIBRARY_PATH:+:$LIBRARY_PATH}" \
    LD_LIBRARY_PATH="$cuda_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    TORCH_CUDA_ARCH_LIST="$cuda_arch" \
        conda run --no-capture-output -n ugo "$@"
}

SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=0 \
    cuda_build python -m pip install --no-build-isolation --no-deps -e third_party/sam2
cuda_build python -m pip install --force-reinstall --no-build-isolation --no-deps ugo/detector/ops
conda run --no-capture-output -n ugo python -m pip install --no-build-isolation --no-deps -e third_party/TrackEval

mkdir -p weights
sam_checkpoint="weights/sam2.1_hiera_large.pt"
sam_url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
sam_sha="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
ugo_checkpoint="weights/UGO.pth"
ugo_sha="5e1dc4101a7944433d03e77447ba7108f5df5c5e0788643b1e163a9edc514a95"

if [[ ! -f "$sam_checkpoint" ]]; then
    curl --fail --location --retry 3 --output "$sam_checkpoint" "$sam_url"
fi

echo "$sam_sha  $sam_checkpoint" | sha256sum --check

if [[ ! -f "$ugo_checkpoint" ]]; then
    echo "Missing $ugo_checkpoint" >&2
    echo "Download it from the link in README.md before running the installer." >&2
    exit 1
fi

echo "$ugo_sha  $ugo_checkpoint" | sha256sum --check
echo "UGO installation complete"
