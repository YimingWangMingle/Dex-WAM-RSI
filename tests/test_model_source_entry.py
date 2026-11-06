"""Model imports stay namespaced even when legacy names are unavailable."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ModelSourceTests(unittest.TestCase):
    def run_isolated(self, code):
        result = subprocess.run([sys.executable, '-S', '-c', code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_loader_is_lazy_and_does_not_change_search_path(self):
        self.run_isolated("""
import sys
before = list(sys.path)
import thinker_wm
from thinker_wm.utils.output_paths import repo_root
assert before == sys.path
assert 'torch' not in sys.modules
assert 'compat' not in sys.modules
assert 'utils' not in sys.modules
assert repo_root() == thinker_wm.SOURCE_ROOT.parent.parent
assert thinker_wm.source_path('models/inference_api.py').is_file()
try:
    thinker_wm.source_path('../RoboProbe')
except ValueError:
    pass
else:
    raise AssertionError('resource escape accepted')
""")

    def test_root_has_no_model_links_or_second_vendor_directory(self):
        for name in ('models', 'utils', 'configs', 'libs', 'wan', 'train',
                     'train_debug', 'data_prepare', 'monitor', 'periodic_eval'):
            self.assertFalse((ROOT / name).exists(), name)
            self.assertFalse((ROOT / name).is_symlink(), name)
        self.assertFalse((ROOT / 'third_party/thinker_wm').exists())

    def test_optional_aliases_share_submodule_identity(self):
        if not (ROOT / 'compat').exists():
            self.skipTest('compat intentionally absent from this distribution')
        self.run_isolated("""
import importlib
from compat import enable
enable()
for legacy, canonical in [('utils.output_paths', 'thinker_wm.utils.output_paths'),
                          ('third_party.thinker_wm.utils.output_paths', 'thinker_wm.utils.output_paths')]:
    assert importlib.import_module(legacy) is importlib.import_module(canonical)
""")


if __name__ == '__main__':
    unittest.main()
