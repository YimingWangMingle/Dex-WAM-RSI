"""Inspect an existing training run, or seal a completed checkpoint for RSI.

No training, model load, promotion, simulator action or automatic latest selection.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from environments.robodojo.training import MARKER, inspect_checkpoint, load_training, seal_checkpoint, verify_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inspect', 'seal', 'verify'))
    parser.add_argument('--config')
    parser.add_argument('--field-map')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output')
    parser.add_argument('--bundle')
    args = parser.parse_args()
    if args.command == 'verify':
        if not args.bundle:
            parser.error('verify needs --bundle')
        document = verify_bundle(args.bundle)
        result = dict(bundle_id=document['bundle_id'], status='verified_candidate',
                      acceptance=document['acceptance'])
    else:
        if not args.config:
            parser.error('inspect/seal needs --config')
        if args.command == 'seal':
            if not args.checkpoint or not args.output:
                parser.error('seal needs an explicit --checkpoint and new --output')
            document = seal_checkpoint(args.config, args.checkpoint, args.output, args.field_map)
            result = dict(bundle_id=document['bundle_id'], status='sealed_candidate', output=args.output)
        else:
            cfg, fm, profile, _ = load_training(args.config, args.field_map)
            candidates, rejected = [], []
            for marker in sorted(Path(cfg['system']['checkpoint_dir']).rglob(MARKER)):
                try:
                    candidates.append(inspect_checkpoint(marker.parent))
                except (ValueError, OSError, json.JSONDecodeError) as error:
                    rejected.append(dict(path=str(marker.parent), reason=str(error)))
            result = dict(profile=asdict(profile), complete_checkpoints=candidates,
                          rejected_checkpoints=rejected, checkpoint_selected=None,
                          status='await_explicit_checkpoint' if candidates else 'no_complete_checkpoint',
                          training_config_unchanged=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
