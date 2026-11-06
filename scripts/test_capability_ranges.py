"""Fixed verified starts, grouped branch sampling and deterministic replay."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from test_sampling_ranges_stride import _write_ranges, _StubRepository
from thinker_wm.data_prepare.capability_ranges import CapabilityRangesPool
from motus_data.sample_pool import EpisodeGroupedBatchSampler


class PairedPool(unittest.TestCase):
    def test_fixed_windows_and_rank_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"ranges.sqlite"
            ann = Path(temp)/"annotations.jsonl"
            _write_ranges(path)
            rows = []
            for group in range(8):
                for branch, episode in enumerate(("ep_long", "ep_mid")):
                    rows.append(dict(episode_id=episode, start_frame=group*4,
                        context_id=f"snapshot_{group}", instruction_id="goal",
                        branch_id=f"branch_{branch}", joint_valid=True, history_required=False,
                        outcome_source="simulator_recording"))
            ann.write_text("\n".join(json.dumps(r) for r in rows))
            pool = CapabilityRangesPool(path,_StubRepository(),annotations_path=ann,
                                        group_size=4,seed=20261020,action_start_offset=0)
            first = [pool.generate(i) for i in range(len(pool))]
            self.assertEqual(len(pool),32)
            self.assertEqual(first,[pool.generate(i) for i in reversed(range(len(pool)))][::-1])
            pool.set_epoch(3)
            self.assertEqual(first,[pool.generate(i) for i in range(len(pool))])
            dataset = SimpleNamespace(episode_groups=pool.episode_groups,set_epoch=pool.set_epoch)
            group_ids = []
            for rank in range(2):
                sampler = EpisodeGroupedBatchSampler(dataset,batch_size=4,num_replicas=2,rank=rank,
                                                     seed=20261020)
                seen = set()
                for batch in sampler:
                    selections = [pool.generate(i) for i in batch]
                    self.assertEqual(len({s['state_indices'][0] for s in selections}),1)
                    self.assertEqual({s['episode_id'] for s in selections},{"ep_long","ep_mid"})
                    self.assertTrue(all(s['state_indices'][0] == s['action_indices'][0] for s in selections))
                    seen.add(batch[0]//4)
                group_ids.append(seen)
            self.assertFalse(group_ids[0] & group_ids[1])
            self.assertEqual(len(group_ids[0] | group_ids[1]),8)
            rows[0]["start_frame"] = 100000
            ann.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError):
                CapabilityRangesPool(path,_StubRepository(),annotations_path=ann,group_size=4)


if __name__ == "__main__":
    unittest.main()
