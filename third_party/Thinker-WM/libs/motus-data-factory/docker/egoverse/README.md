# Motus Python 3.11 EgoVerse image

This build keeps the original `motus` Python 3.10 environment intact and adds
an isolated `motus311` environment. The immutable `v0.0.3` image remains the
production rollback point.

## Scope

- Base image: `ubhub.ubtrobot.com/ogi/chen-liao-world-model:v0.0.3`
- OS/CUDA/NCCL: unchanged
- Python: new 3.11 environment
- Training packages: reconstructed from the `v0.0.3` `motus` environment
- EgoVerse: `zarr==3.1.5` and its pinned core dependencies
- Official EgoVerse simulation, Ray, MuJoCo, and Aria SDK dependencies: omitted

The baseline records both GUI and headless OpenCV distributions. The headless
wheel is deliberately reinstalled last because training pods do not provide
`libGL.so.1`.

## Build

Run from this directory:

```bash
docker build \
  --progress=plain \
  -t ubhub.ubtrobot.com/ogi/chen-liao-world-model:v0.0.4-py311-egoverse-test \
  .
```

Override package indexes when an internal mirror is required:

```bash
docker build \
  --progress=plain \
  --build-arg CONDA_MAIN_CHANNEL=https://<conda-main-mirror> \
  --build-arg PIP_INDEX_URL=https://<pip-mirror>/simple \
  --build-arg PYTORCH_INDEX_URL=https://<pytorch-cu128-mirror> \
  -t ubhub.ubtrobot.com/ogi/chen-liao-world-model:v0.0.4-py311-egoverse-test \
  .
```

`flash-attn` is built for H200 (`TORCH_CUDA_ARCH_LIST=9.0`) by default.

## Runtime selection

The image puts `motus311` first on `PATH` for non-interactive commands. For an
interactive shell, select it explicitly:

```bash
source /home/fengodchen/miniconda3/etc/profile.d/conda.sh
conda activate motus311
python --version
```

Do not activate the legacy `motus` environment for EgoVerse.

## Cluster acceptance order

1. Run `docker/egoverse/validate_image.py`.
2. Run `tools/check_egoverse_environment.py` against the read-only dataset.
3. Run the bounded Aria real payload/queue smoke.
4. Run a one-GPU model smoke.
5. Run a single-node eight-GPU distributed smoke.
6. Run the three-node IB/NCCL smoke.

No production EgoVerse Catalog should be generated before steps 1-3 pass.
