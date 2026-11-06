"""Lazy aliases preserving module/class identity for legacy rsi.* imports.

All mappings are explicit. The finder does not affect unrelated imports and
never eagerly loads NumPy, Torch, model services or simulator dependencies.
"""
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

ALIASES = json.loads(Path(__file__).with_name('module_aliases.json').read_text())


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        module = importlib.import_module(self.target)
        self.original = (module.__spec__, module.__loader__, module.__package__)
        return module

    def exec_module(self, module):
        module.__spec__, module.__loader__, module.__package__ = self.original


class _AliasFinder(importlib.abc.MetaPathFinder):
    def __init__(self, aliases):
        self.aliases = dict(aliases)

    def find_spec(self, fullname, path=None, target=None):
        canonical = self.aliases.get(fullname)
        if canonical is None:
            for legacy in sorted(self.aliases, key=len, reverse=True):
                if fullname.startswith(legacy + '.'):
                    canonical = self.aliases[legacy] + fullname[len(legacy):]
                    break
        if canonical is None:
            return None
        canonical_spec = importlib.util.find_spec(canonical)
        if canonical_spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(canonical),
            is_package=canonical_spec.submodule_search_locations is not None)


def install_legacy_aliases(aliases=None):
    mappings = ALIASES if aliases is None else aliases
    if not any(isinstance(finder, _AliasFinder) and finder.aliases == mappings
               for finder in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder(mappings))
