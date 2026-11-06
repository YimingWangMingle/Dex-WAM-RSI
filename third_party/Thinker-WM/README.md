# Thinker-WM

This is the world-model implementation snapshot used by the A6000 validated
ThinkerWAM-RSI source. The directory spelling is intentional.

- `models/`: model architecture and `inference_api.py` (action generation/A2V).
- `runtime/`: inference policies, model observation transforms, history/cache,
  action postprocessing and model-side chunk handling.
- `scripts/eval_video_ckpt.py`: checkpoint/config loading and offline inference.
- `configs/`, `data_prepare/`, `utils/`, `train/`, `train_debug/`,
  `periodic_eval/`, `monitor/`: support code preserved for existing imports and
  training/inference preprocessing consistency. Moving it does not start training.
- `wan/`, `libs/`: physical copies of the existing dependency snapshots, not links
  to the original installation. No checkpoint or dataset is included.

Root-level relative links and compatibility modules preserve existing imports.
They all point inside this checkout. Simulator lifecycle/control adapters live
under `adapters/`; they are not model implementations.

`UPSTREAM.json` records source content hashes and migration origins. The source
snapshot did not include a complete license inventory (in particular Wan's
repository-level license file). Original source headers and supplied notices are
preserved. This directory is an internal source snapshot, not a declaration of
redistribution rights or a newly assigned license. Complete upstream provenance
and licensing before an external release.

Model inference uses the existing WAM Python environment. The portable RSI core
only requires `requirements-core.txt`; installing it does not install torch,
CUDA, CuRobo, RoboTwin or RoboDojo. Keep simulator and model services in separate
environments as needed. Configure checkpoint/config/service paths explicitly.
