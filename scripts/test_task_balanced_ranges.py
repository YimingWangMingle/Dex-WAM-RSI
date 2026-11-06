"""CPU checks for distribution, grouped loading, rank sharding and replay."""
import json
import sqlite3
import tempfile
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from test_sampling_ranges_stride import _write_ranges, _StubRepository
from thinker_wm.data_prepare.task_balanced_ranges import TaskBalancedRangesPool
from motus_data.sample_pool import EpisodeGroupedBatchSampler


class Repository(_StubRepository):
    def iter_episode_infos(self):
        # Unequal episode counts AND window counts must not bias task probability.
        for episode, task in (("short", "A"), ("long1", "B"), ("long2", "B")):
            yield SimpleNamespace(episode_id=episode, metadata={"task_name": task})


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ranges.sqlite"
        _write_ranges(path, layout="video_stride")
        with sqlite3.connect(path) as db:
            db.row_factory = sqlite3.Row
            template = dict(db.execute("SELECT * FROM episode_ranges LIMIT 1").fetchone())
            db.execute("DELETE FROM episode_ranges")
            for i, (eid, count) in enumerate((("short", 27), ("long1", 50000), ("long2", 100000))):
                row = dict(template, ordinal=i, episode_id=eid,
                           intervals=json.dumps([[0, count-1]]), episode_len=count+1000)
                db.execute("INSERT INTO episode_ranges (" + ",".join(row) + ") VALUES ("
                           + ",".join("?" for _ in row) + ")", tuple(row.values()))
        pool = TaskBalancedRangesPool(path, Repository(), seed=20260921,
                                     group_size=8, action_start_offset=0)
        counts = Counter(pool._locate(i)[0].episode_id for i in range(0, len(pool), 8))
        total = sum(counts.values())
        assert abs(counts["short"]/total - .5) < .025, counts
        assert abs(counts["long1"]/total - .25) < .025, counts
        assert abs(counts["long2"]/total - .25) < .025, counts
        initial = [pool.generate(i) for i in range(128)]
        assert initial == [pool.generate(i) for i in reversed(range(128))][::-1]
        for start in range(0, 128, 8):
            assert len({x["episode_id"] for x in initial[start:start+8]}) == 1
        for item in initial:
            assert item["state_indices"][0] == item["action_indices"][0]
        pool.set_epoch(1)
        assert initial != [pool.generate(i) for i in range(128)]
        pool.set_epoch(0)
        assert initial == [pool.generate(i) for i in range(128)]
        # Existing real sampler: all ranks same length, virtual groups disjoint.
        dataset = SimpleNamespace(episode_groups=pool.episode_groups, set_epoch=pool.set_epoch)
        groups = []
        for rank in range(16):
            sampler = EpisodeGroupedBatchSampler(dataset, batch_size=8,
                num_replicas=16, rank=rank, seed=20260921)
            batches = list(sampler)
            assert all(len(b) == 8 and len({i//8 for i in b}) == 1 for b in batches)
            groups.append({b[0]//8 for b in batches})
        assert len({len(g) for g in groups}) == 1
        assert sum(map(len, groups)) == len(set.union(*groups))
        print("PASS task A~50%, task B~50%; B episodes~25% each:", counts)
        print("PASS same-episode batches, offset=0, deterministic replay, new epoch, 16-rank sharding")
        wor = TaskBalancedRangesPool(path, Repository(), seed=20260921,
            group_size=8, action_start_offset=0, without_replacement=True)
        draws = defaultdict(list)
        for i in range(2048):
            ep, offset = wor._locate(i)
            assert ep.episode_id == pool._locate(i)[0].episode_id
            selection = wor.generate(i)
            draws[ep.episode_id].append(selection["state_indices"][0])
        assert len(set(draws["short"][:27])) == 27
        assert len(set(draws["short"][27:54])) == 27
        assert len(set(draws["long1"])) == len(draws["long1"])
        assert len(set(draws["long2"])) == len(draws["long2"])
        before = [wor.generate(i) for i in range(128)]
        assert before == [wor.generate(i) for i in reversed(range(128))][::-1]
        wor.set_epoch(1)
        assert before != [wor.generate(i) for i in range(128)]
        wor.set_epoch(0)
        assert before == [wor.generate(i) for i in range(128)]
        print("PASS episode-wide decks across groups, unchanged task draws, deterministic epoch replay")


def check_real(config_path):
    from omegaconf import OmegaConf
    from motus_data.repository import EpisodeRepository
    from thinker_wm.data_prepare.factory import _indexed_sources, _readers_from_spec
    config = OmegaConf.load(config_path)
    source = config.dataset.sources[0]
    repo = EpisodeRepository(source.catalog_path, index_path=source.index_path,
        verify_sha256=False, readers=_readers_from_spec(source, source_name=source.name))
    spec = _indexed_sources(config)[0]
    pool = spec["sample_pool"](repository=repo, seed=config.training.seed)
    assert len(pool._tasks) == 50
    counts = Counter()
    task_of = {ep.episode_id: task for task, episodes in
               zip(pool._tasks, pool._task_episodes) for ep in episodes}
    for group in range(len(pool)//pool.group_size):
        ep, _ = pool._locate(group*pool.group_size)
        counts[task_of[ep.episode_id]] += 1
    total = sum(counts.values())
    assert max(abs(n/total - .02) for n in counts.values()) < .002
    eval_spec = _indexed_sources(config, eval=True)[0]
    eval_pool = eval_spec["sample_pool"](repository=repo, seed=config.periodic_eval.seed)
    assert type(eval_pool).__name__ == "SamplingRangesPool"
    assert eval_pool.generate(0) == eval_pool.generate(0)
    print("PASS REAL:", len(pool._tasks), "tasks", len(pool._episodes), "episodes",
          total, "virtual groups; task fractions", min(counts.values())/total,
          max(counts.values())/total, "fixed evaluation pool preserved")
    # Exercise actual grouped decoding (CPU only), not just index arithmetic.
    from motus_data.sample_pool import SamplePoolDataset
    dataset = SamplePoolDataset(repo, pool, video_size=(480, 640))
    batch = dataset.get_grouped_samples(list(range(pool.group_size)))
    assert len(batch) == 8
    assert len({item.info.episode_id for item in batch}) == 1
    print("PASS REAL grouped decode: 8 samples", batch[0].info.episode_id)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_real(sys.argv[1])
    else:
        main()
