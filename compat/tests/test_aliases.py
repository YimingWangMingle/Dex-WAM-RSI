"""Moved implementations keep the old import identity without duplicate state."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from compat import enable
        enable()
        # Wan's upstream T5 constructor evaluates current_device at import time.
        # This identity-only CPU check must not initialize a GPU context.
        from unittest.mock import patch
        cuda_default = patch('torch.cuda.current_device', return_value=0)
        cuda_default.start()
        cls.addClassCleanup(cuda_default.stop)
        # Alias identity test only; the native simulator's ModelTemplate is optional.
        import types
        from unittest.mock import patch
        try:
            importlib.import_module('XPolicyLab.utils.model_template')
        except ModuleNotFoundError as exc:
            if exc.name not in ('XPolicyLab', 'XPolicyLab.utils', 'XPolicyLab.utils.model_template'):
                raise
            stub = types.ModuleType('XPolicyLab.utils.model_template')
            stub.ModelTemplate = type('ModelTemplate', (), {})
            context = patch.dict(sys.modules, {'XPolicyLab.utils.model_template': stub})
            context.start()
            cls.addClassCleanup(context.stop)

    def test_rsi_contains_only_outer_core(self):
        self.assertEqual({p.name for p in (ROOT/'rsi').glob('*.py')}, {
            '__init__.py','__main__.py','framework_campaign.py','core_adapter.py',
            'proposal_adapter.py','version_manifest.py','selection_policy.py'})
        self.assertFalse((ROOT/'rsi/perception').exists())
        self.assertFalse((ROOT/'rsi/environments').exists())

    def test_all_legacy_aliases_share_canonical_modules(self):
        mappings=json.loads((ROOT/'compat/module_aliases.json').read_text())
        for old,new in mappings.items():
            with self.subTest(old=old):
                legacy=importlib.import_module(old)
                canonical=importlib.import_module(new)
                self.assertIs(legacy,canonical)
                self.assertEqual(canonical.__spec__.name,new)
                self.assertTrue(Path(canonical.__file__).is_file() if canonical.__file__ else canonical.__spec__.submodule_search_locations)

    def test_old_commands_work_with_new_runner_types_and_patches(self):
        from unittest.mock import patch
        import rsi.pi05_backend as legacy
        import skills.pi05.backend as canonical
        from rsi.task_skill_runner import CallSkill
        from execution.runner import CallSkill as NewCallSkill
        self.assertIsInstance(CallSkill('wam_control','move',1),NewCallSkill)
        with patch.object(legacy,'PI05Client',object()):
            self.assertIs(legacy.PI05Client,canonical.PI05Client)

    def test_import_order_is_independent_and_core_stays_lightweight(self):
        for first,second in [('execution.runner','rsi.task_skill_runner'),('rsi.task_skill_runner','execution.runner'), ('adapters.robodojo.framework','environments.robodojo.framework'), ('environments.robodojo.framework','adapters.robodojo.framework'), ('skills.wam.physicalrsi','skills.wam.core')]:
            result=subprocess.run([sys.executable,'-c',
                f'from compat import enable; enable(); import importlib,sys; a=importlib.import_module({first!r}); b=importlib.import_module({second!r}); assert a is b; assert {first.startswith("skills.")!r} or "torch" not in sys.modules'],
                cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
