"""CPU/Gloo integration: real evaluator, uneven/empty shards, failures and RNG."""
import json
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party/Thinker-WM/train')]
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
import distributed_validation as dv


class Pipeline:
    def __init__(self, rank, world, total):
        self.total_samples = total
        self.global_ids = list(range(rank, total, world))

    def set_epoch(self, epoch):
        assert epoch == 0
        self.cursor = 0

    def close(self):
        pass

    def next(self):
        if self.cursor == len(self.global_ids):
            raise AssertionError('read beyond finite shard')
        i = self.global_ids[self.cursor]
        self.cursor += 1
        return dict(first_frame=torch.full((1, 3, 2, 2), float(i)),
                    video_frames=torch.zeros(1, 12, 3, 2, 2),
                    action_sequence=torch.zeros(1, 24, 14),
                    initial_state=torch.zeros(1, 14),
                    task_names=[TASKS[i % 3]], episode_ids=[f'episode_{i}'],
                    frame_indices=torch.tensor([[i, i + 2]]),
                    language_embedding=None, vlm_inputs=None), {}


TASKS = ('insert_tubes', 'pour_liquid_into_cup', 'stack_bowls')


class Model(torch.nn.Module):
    def __init__(self, fail=False, blocks=3):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.config = NS(enable_depth=False)
        self.fail = fail
        self.blocks = blocks

    def inference_step(self, **kw):
        assert kw['num_blocks'] == self.blocks, kw['num_blocks']
        assert kw['guidance_scale'] == 5.0
        if self.fail:
            raise ValueError('injected nonzero-rank failure')
        i = kw['first_frame'].float().mean()
        # Randomness must be tied to window id, not rank/world size.
        v = i + torch.rand(())
        return torch.full((1, 3, 12, 2, 2), v.item()), torch.full((1, 24, 14), v.item())


def config(latents=1, media_root=None):
    system = dict(val_seed=20261021)
    if media_root is not None:
        system.update(val_a2v=True, val_depth=True, checkpoint_dir=media_root)
    return OmegaConf.create(dict(
        common=dict(num_video_frames=12, action_chunk_size=24, video_action_freq_ratio=2),
        training=dict(batch_size=8), system=system,
        model=dict(wm=dict(ar_packed_teacher_forcing=True, ar_latents_per_block=latents),
                   inference=dict(num_inference_timesteps=10, guidance_scale=5))))


def fake_evaluate_model(model, batches, accelerator, cfg, **kw):
    """Media mode only: one saved sample with the tensors the publisher renders."""
    batch = batches[0]
    i = int(batch['first_frame'].mean())
    frames = torch.full((3, 3, 4, 4), 0.5)
    sample = dict(gt_first_frame=frames[0], gt_frames=frames, predicted_frames=frames.clone(),
                  gt_depth_frames=frames.clone(), predicted_depth=frames.clone(),
                  depth_valid_mask=torch.ones(3, 4, 4, dtype=torch.bool),
                  a2v_predicted_frames=frames.clone(), gt_actions=torch.zeros(24, 14),
                  predicted_actions=torch.ones(24, 14), vlm_input_messages=f'prompt {i}',
                  original_vlm_input_messages=None, t5_input_messages=None)
    assert kw['max_saved_samples'] == 1
    return {'video_mse': float(i)}, [sample]


def worker(rank, world, rendezvous, output, total, failure, latents, media):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=world,
                            init_method='file://' + rendezvous, timeout=timedelta(seconds=45))
    dv.create_video_grid = lambda *a, **k: None  # Rendering is unrelated to collectives.
    from thinker_wm.periodic_eval import render, validation_artifacts as va
    render.write_video = lambda path, frames, fps: Path(path).write_bytes(b'video')
    if media:
        dv.evaluate_model = fake_evaluate_model
        if failure and rank == 2:
            original_stage = va.stage_window_media
            def broken(sample, root, writer_rank):
                # First round succeeds. Fail after partially writing the second.
                if sample['window_id'] >= world:
                    partial = Path(root, 'partial-encode')
                    partial.mkdir()
                    (partial / 'partial.mp4').write_bytes(b'partial')
                    raise ValueError('injected media failure')
                return original_stage(sample, root, writer_rank)
            va.stage_window_media = broken
    model = Model(fail=failure and rank == 2 and not media, blocks=3 // latents)
    pipe = Pipeline(rank, world, total)
    acc = NS(process_index=rank, num_processes=world, unwrap_model=lambda m: m)
    torch.manual_seed(912 + rank)
    before = torch.get_rng_state().clone()
    try:
        metrics, saved = dv.evaluate_distributed(
            model, pipe, acc, config(latents, output if media else None))
        assert not failure
        assert pipe.cursor == len(pipe.global_ids)
        assert metrics['num_samples'] == total
        if media:
            if rank == 0:
                assert [e['window_id'] for e in saved] == list(range(total)), saved
                assert {e['eval_rank'] for e in saved} == set(range(min(world, total)))
                for e in saved:
                    assert sorted(p.name for p in Path(e['media_dir']).iterdir()) == sorted(
                        va.MEDIA_FILES.values())
                    assert 'gt_frames' not in e and e['inference_seconds'] >= 0
                va.discard_staged_media(saved)
            else:
                assert saved == []
        else:
            assert saved == []
        result = {k: v for k, v in metrics.items() if isinstance(v, (float, int))}
    except RuntimeError as exc:
        assert failure and 'rank 2' in str(exc) and 'injected' in str(exc), str(exc)
        result = {'error_shared': True}
    assert model.training and torch.equal(before, torch.get_rng_state())
    Path(output, f'{rank}.json').write_text(json.dumps(result))
    dist.destroy_process_group()


def run(world, total, failure=False, latents=1, media=False):
    with tempfile.TemporaryDirectory() as folder:
        mp.spawn(worker, args=(world, folder + '/rdzv', folder, total, failure, latents, media),
                 nprocs=world, join=True)
        results = [json.loads(Path(folder, f'{rank}.json').read_text()) for rank in range(world)]
        assert all(result == results[0] for result in results)
        if media:
            assert not list(Path(folder, 'validation_samples_tmp').iterdir())
        return results[0]


if __name__ == '__main__':
    serial = run(1, 5)
    parallel = run(4, 5)
    assert serial == parallel, (serial, parallel)
    run(4, 2)  # Empty shards still join every collective.
    run(4, 5, True)
    assert run(4, 5, latents=3) == run(1, 5, latents=3)  # Joint24: one 24-step block per window
    run(4, 7, media=True)                 # every window staged by the rank that scored it
    run(4, 2, media=True)                 # empty shards
    run(4, 7, True, media=True)           # nonzero-rank encode failure is shared
    try:
        dv.aggregate_window_metrics([{'id': 0, 'metrics': {'mse': 1}}] * 2, 2)
        raise AssertionError('duplicate windows accepted')
    except RuntimeError:
        pass
    print('PASS serial == distributed; uneven/empty shards; exact coverage; 3 AR blocks and 1 Joint24 block; '
          'all-window media staging; CFG5; RNG; shared failure')
