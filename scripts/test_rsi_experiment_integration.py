"""Run the experiment/framework CPU regression suites without a live service."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    'wam_revision', 'wam_review_context', 'a2v_tool', 'agent_context', 'physicalrsi_adapter', 'rsi_data_roles', 'pi05_framework', 'llm_clients', 'framework_native', 'a2v_validation', 'control_skills', 'deepseek_agent', 'deepseek_stack_live',
    'experience', 'geometry_perception', 'guarded_direct', 'joint_continuity',
    'joint_ema_ablation', 'mode3_strict', 'robocoliseum_rsi', 'robodojo_a2v',
    'robodojo_ar2_backend', 'robodojo_async_replan', 'robodojo_bowl_geometry',
    'robodojo_bowl_perception', 'robodojo_contact_scene',
    'robodojo_joint24_runtime', 'robodojo_motion_plan', 'robodojo_pose_control',
    'robodojo_rsi_integration', 'task_skill_runner',
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport', action='store_true',
                        help='Also test local WebSocket transport; requires XPolicyLab')
    parser.add_argument('--physical-core', action='store_true',
                        help='Test actual PhysicalRSI campaign bridge; missing core is a failure')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    suites = SUITES + (('robodojo_a2v_transport',) if args.transport else ()) + (('framework_campaign',) if args.physical_core else ())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(ROOT), env.get('PYTHONPATH'))))
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suite in suites:
        name = 'test_' + suite + '.py'
        try:
            run = subprocess.run([sys.executable, str(ROOT / 'scripts' / name)],
                                 cwd=ROOT, env=env, capture_output=True, text=True,
                                 timeout=120)
            result = dict(test=name, code=run.returncode, output=run.stdout + run.stderr)
        except subprocess.TimeoutExpired:
            result = dict(test=name, code=124, output='Exceeded 120 second timeout')
        results.append(result)
        print(name, 'PASS' if result['code'] == 0 else 'FAIL', flush=True)
        args.output.write_text(json.dumps(dict(
            scope='CPU fixtures; no trained-model or simulator success claim',
            transport_requested=args.transport, physical_core_requested=args.physical_core, results=results), indent=2))
    return int(any(result['code'] for result in results))


if __name__ == '__main__':
    raise SystemExit(main())
