"""Serve the strict RoboDojo WAM adapter over the native XPolicyLab protocol."""
import argparse
import asyncio
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    for name in ('config_path', 'field_map_path', 'checkpoint', 'log_dir'):
        parser.add_argument('--' + name.replace('_', '-'), required=True)
    parser.add_argument('--runtime-config', default=None)
    parser.add_argument('--port', type=int, default=19000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--enable-a2v-review', action='store_true',
                        help='Require candidate preview/selection RPCs before execution')
    parser.add_argument('--mode3', action='store_true', help='Strict visual-only 3-candidate A2V review')
    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    from thinker_wm.runtime.robodojo_policy import RoboDojoPolicy
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    policy_class = RoboDojoPolicy
    mode3 = args.pop('mode3')
    review_enabled = args.pop('enable_a2v_review')
    if mode3 or review_enabled:
        # Proposal service for the framework; legacy/async service stays explicit.
        from environments.robodojo.runtime.robodojo_a2v import A2VReviewMixin, Mode3ReviewMixin
        mixin = Mode3ReviewMixin if mode3 else A2VReviewMixin
        class ReviewedPolicy(mixin, RoboDojoPolicy):
            pass
        policy_class = ReviewedPolicy
    policy = policy_class(args)
    server = PolicyServer(policy, PolicyServerConfig(port=args['port']))
    asyncio.run(server.serve_forever())


if __name__ == '__main__':
    main()
