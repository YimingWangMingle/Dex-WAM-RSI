"""Freeze the audited RoboDojo training contract for RSI deployment.

This imports completed training artifacts; it never submits a training job or
promotes a checkpoint based on open-loop loss. Explicit paths replace 'latest'.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

ROBOT = 'robodojo_arx_x5'
MARKER = '.motus_checkpoint_complete.json'
SCHEMA = 'thinkerwam.robodojo-deployment/v1'
WEIGHTS = ('pytorch_model_0.bin', 'pytorch_model.bin', 'mp_rank_00_model_states.pt',
           'pytorch_model/mp_rank_00_model_states.pt')


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


# Shared runtime identity: model normalization is reversed before dispatch.
NATIVE_JOINT_NAMES = tuple([f'left_j{i}' for i in range(1, 7)] + ['left_gripper'] +
                           [f'right_j{i}' for i in range(1, 7)] + ['right_gripper'])
NATIVE_ACTION_CONTRACT = 'robodojo-joint14/v1:' + digest(dict(
    robot=ROBOT, hz=25, representation='abs_joint', names=NATIVE_JOINT_NAMES,
    arm_unit='rad', gripper='native_0_closed_1_open', normalization='raw_at_runtime'))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class RoboDojoProfile:
    robot: str
    control_hz: float
    action_dim: int
    training_actions: int
    training_frames: int
    frames_per_block: int
    actions_per_block: int
    model_hw: tuple
    resize_mode: str
    action_contract: str
    action_stats_version: str
    field_map_digest: str

    @classmethod
    def from_config(cls, config: Mapping, field_map: Any, statistics: Mapping) -> 'RoboDojoProfile':
        c, wm = config['common'], config['model']['wm']
        sources = config['dataset']['sources']
        if len(sources) != 1 or sources[0]['dataset'] != 'robodojo':
            raise ValueError('this deployment profile requires the audited RoboDojo source')
        source = sources[0]
        robot = source['reader']['robot']
        spec = config['dataset']['robot_names'][robot]
        if (robot != ROBOT or float(c['control_hz']) != 25 or field_map.control_hz != 25
                or field_map.action_semantics != 'abs_joint'
                or any(int(c[k]) != 14 for k in ('action_dim', 'state_dim', 'unified_action_dim'))
                or int(spec['action_dim']) != 14 or list(spec['unified_dim_map']) != list(range(14))
                or int(spec['robot_token_index']) != 17):
            raise ValueError('requires RoboDojo ARX-X5: 25 Hz, native joint14, identity map, token 17')
        expected = NATIVE_JOINT_NAMES
        if field_map.state_names != expected or field_map.action_names != expected:
            raise ValueError('field-map joint/gripper ordering differs from the trained contract')
        # Two audited layouts of the same 12-frame/24-step window: AR (3 blocks of
        # 4 frames/8 steps) and Joint24 (one jointly denoised block of 12/24).
        latents_per_block = int(wm['ar_latents_per_block'])
        if (wm['temporal_mode'] != 'block_causal' or wm.get('ar_packed_teacher_forcing') is not True
                or latents_per_block not in (1, 3) or int(c['video_action_freq_ratio']) != 2
                or wm['action_target_repr'] != 'absolute_native' or wm['action_norm_policy'] != 'zscore'):
            raise ValueError('unsupported RoboDojo temporal/action representation')
        if latents_per_block == 3 and (wm.get('rgb_prefix_error_enable') or wm.get('joint_action_prefix_error_enable')):
            raise ValueError('Joint24 has no generated future prefix; prefix-error replay must be disabled')
        if (wm.get('enable_depth') is not True or wm.get('depth_condition_mode') != 'rgb_only'
                or wm.get('history_enable') is not True or config.get('history', {}).get('enable') is not True):
            raise ValueError('profile requires the trained RGB-only depth prediction and measured history')
        if (int(c['num_video_frames']) != 12 or int(c['action_chunk_size']) != 24
                or source['pool'].get('action_start_offset') != 0):
            raise ValueError('requires 12 future frames / 24 commands and action[t] after state[t]')
        size = (int(c['video_height']), int(c['video_width']))
        if (size != (480, 640) or field_map.layout.size != size or source['reader']['resize_mode'] != 'pad'
                or field_map.layout.views != ('head', 'left_wrist', 'right_wrist')
                or field_map.layout.row_hw != ((320, 426), (160, 213))):
            raise ValueError('requires the audited 480x640 head/wrist T-layout')
        stats = statistics[robot]
        for key in ('mean', 'std'):
            if len(stats[key]) != 14 or not all(math.isfinite(x) for x in stats[key]):
                raise ValueError('missing/invalid RoboDojo action statistics')
        if not all(x > 0 for x in stats['std']):
            raise ValueError('normalization standard deviations must be positive')
        stats_id = 'sha256:' + digest(stats)
        return cls(robot, 25., 14, 24, 12, 4 * latents_per_block, 8 * latents_per_block, size, 'pad',
                   NATIVE_ACTION_CONTRACT, stats_id, field_map.digest)


def inspect_checkpoint(directory: str | Path) -> dict:
    root = Path(directory).resolve(strict=True)
    marker = root / MARKER
    if not marker.is_file():
        raise ValueError('checkpoint is incomplete: no completion marker')
    receipt = json.loads(marker.read_text())
    if receipt.get('status') != 'complete' or type(receipt.get('step')) is not int or receipt['step'] <= 0:
        raise ValueError('invalid checkpoint completion receipt')
    weights = next((root / name for name in WEIGHTS if (root / name).is_file()), None)
    if weights is None or weights.stat().st_size == 0:
        raise ValueError('checkpoint lacks supported complete model weights')
    return dict(path=str(root), step=receipt['step'], marker_sha256=sha256(marker),
                weights=str(weights), weights_bytes=weights.stat().st_size)


def load_training(config_path: str | Path, field_map_path: str | Path | None = None) -> tuple:
    from omegaconf import OmegaConf
    from thinker_wm.data_prepare.robodojo_core import FieldMap
    from thinker_wm.models.action_stats import ACTION_STATS
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    source = config['dataset']['sources'][0]
    path = Path(field_map_path or source['reader']['field_map']).resolve(strict=True)
    fm = FieldMap.load(path)
    profile = RoboDojoProfile.from_config(config, fm, ACTION_STATS)
    return config, fm, profile, path


def seal_checkpoint(config_path: str | Path, checkpoint: str | Path, output: str | Path,
                    field_map_path: str | Path | None = None) -> dict:
    """Create a deployment manifest referencing weights, preserving full config and stats.

    Existing trainer config.json omits wm/history/normalization details, so it is
    not sufficient to construct the inference model. Caller supplies the actual
    training config; provenance is recorded, not inferred from weight shapes.
    """
    from thinker_wm.models.action_stats import ACTION_STATS
    config, fm, profile, fm_path = load_training(config_path, field_map_path)
    ckpt = inspect_checkpoint(checkpoint)
    partial = Path(checkpoint) / 'config.json'
    if partial.is_file() and json.loads(partial.read_text()).get('common') != config['common']:
        raise ValueError('checkpoint common config differs from the supplied training config')
    ckpt['weights_sha256'] = sha256(Path(ckpt['weights']))
    if inspect_checkpoint(checkpoint) != {k: v for k, v in ckpt.items() if k != 'weights_sha256'}:
        raise ValueError('checkpoint changed during sealing')
    root = Path(__file__).resolve().parents[2]
    # Model and preprocessing code are part of the inference identity.
    names = [p.relative_to(root).as_posix() for folder in ('models', 'utils', 'data_prepare')
             for p in (root / 'third_party/Thinker-WM' / folder).glob('*.py')]
    names += ['thinker_wm.py', 'third_party/Thinker-WM/runtime/history_buffer.py', 'third_party/Thinker-WM/runtime/robodojo_adapter.py', 'third_party/Thinker-WM/scripts/eval_video_ckpt.py',
              'environments/robodojo/training.py', 'skills/wam/thinker.py', 'skills/contracts.py',
              'environments/robodojo/runtime/robodojo_execution.py']
    code = {name: sha256(root / name) for name in sorted(names)}
    document = dict(schema=SCHEMA, publication_status='candidate', config=config,
                    training_config_sha256=sha256(Path(config_path)),
                    training_config_path=str(Path(config_path).resolve()),
                    field_map=json.loads(json.dumps(__import__('yaml').safe_load(fm_path.read_text()))),
                    field_map_sha256=sha256(fm_path), profile=asdict(profile),
                    action_statistics=ACTION_STATS[profile.robot], code_sha256=code, checkpoint=ckpt,
                    acceptance=dict(strict_load=False, closed_loop=False, paired_evaluation=False))
    document['bundle_id'] = digest(document)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        stream.write(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return document


def verify_bundle(path: str | Path) -> dict:
    document = json.loads(Path(path).read_text())
    identity = document.pop('bundle_id')
    if document.get('schema') != SCHEMA or digest(document) != identity:
        raise ValueError('deployment bundle identity mismatch')
    document['bundle_id'] = identity
    ckpt = document['checkpoint']
    current = inspect_checkpoint(ckpt['path'])
    if current != {k: v for k, v in ckpt.items() if k != 'weights_sha256'}:
        raise ValueError('checkpoint receipt/weight location changed')
    if sha256(Path(ckpt['weights'])) != ckpt['weights_sha256']:
        raise ValueError('checkpoint content changed')
    root = Path(__file__).resolve().parents[2]
    for name, expected in document['code_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha256(root / relative) != expected:
            raise ValueError('inference code changed: ' + name)
    from thinker_wm.models.action_stats import ACTION_STATS
    if ACTION_STATS[document['profile']['robot']] != document['action_statistics']:
        raise ValueError('deployment normalization differs from training snapshot')
    return document
