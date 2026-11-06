"""Core distribution does not install legacy module aliases."""
from pathlib import Path
import subprocess
import sys
import unittest

class Tests(unittest.TestCase):
    def test_core_imports_without_compat(self):
        code = "import rsi, execution.episode, environments.registry, skills.wam; import sys; assert 'compat' not in sys.modules; assert environments.registry.available() == ('robodojo',)"
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_entrypoints_are_outside_core(self):
        root = Path(__file__).resolve().parents[1]
        self.assertFalse((root / 'adapters.py').exists())
        self.assertFalse((root / 'runtime').exists())
