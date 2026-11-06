#!/usr/bin/env python3
"""Build-time validation for the Python 3.11 EgoVerse training image."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys


EXPECTED_DISTRIBUTIONS = {
    "accelerate": "1.12.0",
    "av": "16.0.1",
    "decord": "0.6.0",
    "deepspeed": "0.18.3",
    "diffusers": "0.36.0",
    "flash-attn": "2.8.3",
    "h5py": "3.15.1",
    "numcodecs": "0.16.5",
    "numpy": "2.2.6",
    "omegaconf": "2.3.0",
    "opencv-python": "4.12.0.88",
    "Pillow": "12.0.0",
    "scipy": "1.15.3",
    "torch": "2.7.1+cu128",
    "torchvision": "0.22.1+cu128",
    "transformers": "5.0.0rc0",
    "triton": "3.3.1",
    "zarr": "3.1.5",
}

IMPORTS = (
    "accelerate",
    "av",
    "cv2",
    "decord",
    "deepspeed",
    "diffusers",
    "flash_attn",
    "h5py",
    "numcodecs",
    "numpy",
    "omegaconf",
    "PIL",
    "scipy",
    "torch",
    "torchvision",
    "transformers",
    "zarr",
)


def main() -> int:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"expected Python 3.11, found {sys.version}")

    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        actual = importlib.metadata.version(distribution)
        if actual != expected:
            raise RuntimeError(
                f"{distribution}: expected {expected}, found {actual}"
            )
        print(f"VERSION_OK {distribution}={actual}", flush=True)

    for module_name in IMPORTS:
        importlib.import_module(module_name)
        print(f"IMPORT_OK {module_name}", flush=True)

    import torch

    if torch.version.cuda != "12.8":
        raise RuntimeError(
            f"expected torch CUDA 12.8, found {torch.version.cuda}"
        )
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("torch NCCL backend is unavailable")

    print(f"python={sys.version}", flush=True)
    print(f"platform={platform.platform()}", flush=True)
    print(f"torch_cuda={torch.version.cuda}", flush=True)
    print("MOTUS311_IMAGE_BUILD_VALIDATION_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
