"""CPU-only subprocess regression: a poisoned CUDA worker must skip teardown."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CudaFailureExit(unittest.TestCase):
    def check_exit(self, message, expected_code, cleanup_expected):
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / 'cleanup'
            code = f'''from pathlib import Path
from thinker_wm.utils.cuda_failure import exit_on_fatal_cuda_error
try:
    exit_on_fatal_cuda_error(RuntimeError({message!r}))
finally:
    Path({str(marker)!r}).write_text('entered')
'''
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            result = subprocess.run([sys.executable, '-c', code], env=env,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, expected_code, result.stderr)
            self.assertEqual(marker.exists(), cleanup_expected)
            if expected_code:
                self.assertIn('without NCCL teardown', result.stderr)

    def test_illegal_access_exits_before_finally(self):
        self.check_exit('CUDA error: an illegal memory access was encountered', 1, False)

    def test_device_assert_exits_before_finally(self):
        self.check_exit('CUDA error: device-side assert triggered', 1, False)

    def test_oom_keeps_cleanup(self):
        self.check_exit('CUDA out of memory. Tried to allocate 2 GiB', 0, True)

    def test_fatal_exit_reaps_owned_workers(self):
        with tempfile.TemporaryDirectory() as d:
            pid_file = Path(d) / 'child_pid'
            code = f'''import multiprocessing as mp, time
from pathlib import Path
from thinker_wm.utils.cuda_failure import exit_on_fatal_cuda_error
child = mp.Process(target=time.sleep, args=(30,))
child.start()
Path({str(pid_file)!r}).write_text(str(child.pid))
exit_on_fatal_cuda_error(RuntimeError('CUDA error: an illegal memory access was encountered'))
'''
            result = subprocess.run([sys.executable, '-c', code],
                env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid_file.read_text()), 0)

    def test_data_failure_keeps_cleanup(self):
        self.check_exit('Missing episode', 0, True)


if __name__ == '__main__':
    unittest.main()
