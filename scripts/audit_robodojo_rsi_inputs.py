"""Read-only comparison of recorded RoboDojo training inputs with RSI live adaptation.

Samples start/middle/end rows from one recorded episode per catalog task. This
is an input contract audit, not a model or simulator success evaluation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
from thinker_wm.data_prepare.robodojo import RoboDojoEpisodeReader
from thinker_wm.data_prepare.robodojo_core import _read_frames
from environments.robodojo.training import load_training
from thinker_wm.runtime.robodojo_adapter import ObservationBinding, RoboDojoObservationAdapter


def audit(config: str, catalog: str, field_map: str | None = None) -> dict:
    _, fm, profile, fm_path = load_training(config, field_map)
    reader = RoboDojoEpisodeReader(field_map=str(fm_path), robot=profile.robot, resize_mode=profile.resize_mode)
    selected = {}
    with open(catalog) as stream:
        for line in stream:
            record = json.loads(line)
            selected.setdefault(record['meta']['task_name'], record)
    if not selected:
        raise ValueError('empty episode catalog')
    binding = ObservationBinding({view: view for view in fm.layout.views},
                                 state_keys=tuple(f'state_{i}' for i in range(len(fm.state_paths))))
    adapter = RoboDojoObservationAdapter(fm, binding, model_hw=profile.model_hw,
                                        resize_mode=profile.resize_mode, robot_name=profile.robot)
    results = []
    for task, record in sorted(selected.items()):
        if record['meta']['field_map_digest'] != fm.digest or record['robot'] != profile.robot:
            raise ValueError('catalog identity differs from deployment contract')
        path = record['video']['storage']['path']
        row_count = min(record['video']['count'], record['state']['count'], record['action']['count'])
        if row_count < 2:
            raise ValueError('episode too short for next-state alignment audit')
        rows = sorted({0, (row_count - 2) // 2, row_count - 2})
        repository_record = {name: dict(locator=dict(uri=record[name]['storage']['path']))
                             for name in ('video', 'state', 'action')}
        trained_rgb = reader.read_video(repository_record, rows, output_format='rgb_float32_chw',
                                        video_size=profile.model_hw)
        trained_state = reader.read_state(repository_record, rows)
        trained_action = reader.read_action(repository_record, rows)
        next_state = reader.read_state(repository_record, [r + 1 for r in rows])
        rgb_error = state_error = 0.
        with h5py.File(path, 'r') as h5:
            for index, row in enumerate(rows):
                live = {view: _read_frames(h5[fm.rgb_views[view]], [row], fm.rgb_encoding)[0]
                        for view in fm.layout.views}
                live.update({f'state_{i}': h5[key][row] for i, key in enumerate(fm.state_paths)})
                obs, depth = adapter.observation(live, row)
                rgb_error = max(rgb_error, float((obs.rgb - trained_rgb[index]).abs().max()))
                state_error = max(state_error, float((obs.state - trained_state[index]).abs().max()))
                if depth is not None or obs.time_s != row / profile.control_hz:
                    raise AssertionError('RGB-only/time contract mismatch')
        next_error = float((trained_action - next_state).abs().max())
        if rgb_error != 0 or state_error != 0 or not np.isfinite(next_error) or next_error > 1e-6:
            raise AssertionError(f'{task}: train/runtime RGB={rgb_error}, state={state_error}, action_next={next_error}')
        results.append(dict(task=task, episode_id=record['episode_id'], path=path, rows=rows,
                            rgb_max_abs_error=rgb_error, state_max_abs_error=state_error,
                            action_next_state_max_abs_error=next_error))
    return dict(status='passed', kind='recorded_input_contract_only', tasks=results,
                control_hz=profile.control_hz, actions_per_block=profile.actions_per_block,
                model_inference_tested=False, closed_loop_tested=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--catalog', required=True)
    parser.add_argument('--field-map')
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(audit(args.config, args.catalog, args.field_map), indent=2))


if __name__ == '__main__':
    main()
