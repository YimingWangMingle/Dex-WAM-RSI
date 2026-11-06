"""Run bounded RSI selection using a trusted native-environment driver.

The driver defines build_suite(initial, work_root, config, revision) and
repair(parent, verified_development_evaluations). See docs/RSI_USAGE.md §13.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json

from rsi.version_manifest import digest, file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from rsi.framework_campaign import FailureEdits, build_campaign
    # Fail before opening an environment if the actual upstream core is absent.
    from PhysicalRSI_core.self_harness.campaign import ImprovementCampaign
    config = json.loads(args.config.read_text())
    initial = json.loads(args.initial.read_text())
    revision = digest(dict(driver=file_digest(args.driver), config=config.get('driver', {})))
    ns = dict(__file__=str(args.driver.resolve()), __name__='rsi_campaign_driver')
    exec(compile(args.driver.read_bytes(),str(args.driver),'exec'),ns)
    suite = ns['build_suite'](initial, args.output/'native', config.get('driver',{}), revision)
    if suite.driver_revision != revision:
        raise ValueError('driver must bind its source/config revision to the suite')
    campaign = build_campaign(root=args.output,suite=suite,
        repair=FailureEdits(ns['repair'], revision=revision,max_candidates=config['max_candidates']),
        max_rounds=config['max_rounds'],max_candidates=config['max_candidates'],
        max_trials=config['max_trials'],seconds_per_episode=config['seconds_per_episode'],
        profile=config['profile'])
    result = campaign.run(initial)
    from PhysicalRSI_core.infra.storage import atomic_json
    atomic_json(args.output/'result.json',result)
    print(json.dumps(dict(status=result['status'],completed_rounds=result['completed_rounds'],
        selected_candidate=result['current']['harness']['id'],
        selected_revision=result['current']['freeze_sha256']),indent=2))


if __name__ == '__main__':
    main()
