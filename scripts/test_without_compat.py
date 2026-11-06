"""Verify a physical source export excluding compat; CPU fixtures only."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix='rsi-core-export-') as tmp:
        exported = Path(tmp) / 'code'
        shutil.copytree(root, exported, symlinks=True,
            ignore=shutil.ignore_patterns('.git', 'compat', '__pycache__', 'validation', 'outputs'))
        assert not (exported / 'compat').exists()
        audit = Path(tmp) / 'audit'
        audit.mkdir()
        (audit / 'sitecustomize.py').write_text(
            "import sys, importlib.abc\n"
            "class DenyLegacy(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname.split('.')[0] in ('compat','runtime','adapters','models','utils','configs','wan','train','train_debug','data_prepare','periodic_eval'):\n"
            "            raise ImportError('Core export imported legacy module: '+fullname)\n"
            "sys.meta_path.insert(0,DenyLegacy())\n")
        # Keep only external dependency paths, never the original checkout.
        external = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep)
                    if p and not Path(p).resolve().is_relative_to(root)]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(audit), str(exported), *external]),
                   CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
        reports = Path(tmp) / "reports"
        reports.mkdir()
        commands = [
            ['scripts/test_modular_refactor.py', '--output', str(reports / 'robodojo-focus.json')],
            ['scripts/test_rsi_experiment_integration.py', '--transport', '--physical-core',
             '--output', str(reports / 'no-compat-integration.json')],
            ['scripts/test_sparse_vae_cache.py'],
        ]
        for command in commands:
            result = subprocess.run([sys.executable, *command], cwd=exported,
                                    env=env, capture_output=True, text=True, timeout=900)
            results.append(dict(command=command, returncode=result.returncode,
                                output=result.stdout + result.stderr))
            print(command[0], result.returncode, flush=True)
        for name in ['robodojo-focus.json', 'no-compat-integration.json']:
            source = reports / name
            if source.exists():
                shutil.copyfile(source, output.parent / ('without-compat-' + name))
    output.write_text(json.dumps(dict(scope='CPU export without compat; no live model/simulator calls',
        legacy_imports_blocked=['compat', 'runtime', 'adapters', 'models', 'utils', 'configs', 'wan', 'train', 'train_debug', 'data_prepare', 'periodic_eval'], results=results), indent=2))
    return int(any(r['returncode'] for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
