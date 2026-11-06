"""Archive stack_bowls-shaped CPU wiring checks; no native simulator or learned policy.

Runs production episode/campaign paths using the existing deterministic fixtures.
Outputs are protocol evidence, never a physical stacking success measurement.
"""
from pathlib import Path
import argparse
import importlib
import json
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

CASES = [
    ('direct_episode', 'test_framework_native', 'test_direct_uses_same_runner_executor_and_events'),
    ('direct_wam_direct', 'test_framework_native', 'test_direct_wam_direct_handoff_preserves_measured_history'),
    ('vla_episode', 'test_pi05_framework', 'test_pi05_baseline_shared_executor_exact_prefix'),
    ('direct_vla_direct', 'test_pi05_framework', 'test_direct_vla_direct_share_measured_state'),
    ('wam_vla_wam', 'test_pi05_framework', 'test_wam_vla_action_review_handoff'),
    ('agent_a2v_wam', 'test_framework_native', 'test_agent_calls_independent_a2v_then_reviews_wam'),
    ('direct_outer_rsi', 'test_framework_campaign', 'test_actual_core_inherits_then_retains_and_resume_does_not_move'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for label, module_name, method in CASES:
        case = importlib.import_module(module_name).Tests(method)
        row = dict(name=label, test=module_name + '.' + method, passed=False)
        try:
            case.setUp()
            getattr(case, method)()
            case.tearDown()
            row['passed'] = True
            if label == 'direct_outer_rsi':
                row['verified_outer'] = dict(rounds=['inherited', 'retained'], environment_resets=6, steps_per_episode=2, resume_without_reexecution=True)
        except Exception:
            row['error'] = traceback.format_exc()
        finally:
            source = getattr(case, 'root', None)
            if source is not None and source.exists():
                archived = out / label
                shutil.copytree(source, archived)
                row['artifacts'] = str(archived)
                row['episodes'] = []
                summaries = sorted(archived.rglob('episode-summary.json'))
                if label == 'direct_outer_rsi':
                    summaries = [p for p in summaries if 'artifacts' in p.relative_to(archived).parts]
                for summary in summaries:
                    d = json.loads(summary.read_text())
                    row['episodes'].append(dict(file=str(summary.relative_to(out)),
                        executed_steps=d.get('executed_steps'),
                        registered_skills=d.get('registered_skills'),
                        counters=d.get('counters')))
                row['execution_sequences'] = []
                event_files = sorted(archived.rglob('framework-events.jsonl'))
                if label == 'direct_outer_rsi':
                    event_files = [p for p in event_files if 'artifacts' in p.relative_to(archived).parts]
                for events in event_files:
                    rows = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
                    row['execution_sequences'].append(dict(file=str(events.relative_to(out)),
                        skills=[r.get('skill') for r in rows if r.get('event') == 'execution']))
            case.doCleanups()
        results.append(row)
        print(label, 'PASS' if row['passed'] else 'FAIL', flush=True)
    report = dict(task='stack_bowls', environment='native-shaped CPU fixture, not RoboDojo simulator',
        policies='deterministic test doubles, no model service calls',
        outer='production RSI campaign; deterministic candidate repair',
        artifact_note='Archived fixture artifacts may retain original temporary absolute paths; not replay bundles.',
        results=results)
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    return int(not all(r['passed'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
