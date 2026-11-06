"""CPU-only launch protection; all state lives in temporary directories."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinker_wm.utils.robodojo_launch_guard import guard_launch


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = self.root / 'config.yaml'
        self.cfg.write_text('training:\n  seed: 20261023\nresume:\n  reset_scheduler: false\n')
        self.out, self.ckpts = self.root / 'output', self.root / 'checkpoints'

    def launch(self, node=0, resume=None):
        guard_launch(self.cfg, self.out, self.ckpts, node, resume)

    def checkpoint(self, step=2000, complete=True):
        path = self.ckpts / f'checkpoint_step_{step}'
        path.mkdir(parents=True)
        if complete:
            (path / '.motus_checkpoint_complete.json').write_text(json.dumps({'status': 'complete', 'step': step}))
        return path

    def test_second_node_joins_but_duplicate_node_fails(self):
        self.launch()
        (self.out / 'train_node0.log').write_text('running')
        self.launch(1)
        with self.assertRaisesRegex(ValueError, 'already has'): self.launch()

    def test_existing_checkpoint_requires_resume_even_for_new_node(self):
        self.launch()
        self.checkpoint()
        with self.assertRaisesRegex(ValueError, 'already has'): self.launch(1)

    def test_resume_preserves_snapshot_and_accepts_incomplete_future(self):
        self.launch()
        cp = self.checkpoint()
        self.checkpoint(4000, False)
        original = (self.out / 'config_at_launch.yaml').read_bytes()
        self.launch(resume=cp)
        self.assertEqual(original, (self.out / 'config_at_launch.yaml').read_bytes())

    def test_rewind_is_rejected(self):
        self.launch()
        cp = self.checkpoint()
        self.checkpoint(4000)
        with self.assertRaisesRegex(ValueError, 'newer'): self.launch(resume=cp)

    def test_missing_or_invalid_marker_is_rejected(self):
        self.launch()
        cp = self.checkpoint(complete=False)
        with self.assertRaises(OSError): self.launch(resume=cp)
        (cp / '.motus_checkpoint_complete.json').write_text('{"status":"complete","step":4000}')
        with self.assertRaisesRegex(ValueError, 'receipt'): self.launch(resume=cp)

    def test_config_drift_is_rejected(self):
        self.launch()
        cp = self.checkpoint()
        self.cfg.write_text(self.cfg.read_text().replace('20261023', '20261024'))
        with self.assertRaisesRegex(ValueError, 'configuration differs'): self.launch(resume=cp)

    def test_missing_snapshot_and_foreign_checkpoint_are_rejected(self):
        cp = self.checkpoint()
        with self.assertRaisesRegex(ValueError, 'missing'): self.launch(resume=cp)
        foreign = self.root / 'checkpoint_step_2000'
        foreign.mkdir()
        with self.assertRaisesRegex(ValueError, 'directly inside'): self.launch(resume=foreign)

    def test_old_log_requires_resume(self):
        self.out.mkdir()
        (self.out / 'train_node0.log').touch()
        with self.assertRaisesRegex(ValueError, 'already has'): self.launch()


if __name__ == '__main__': unittest.main(verbosity=2)
