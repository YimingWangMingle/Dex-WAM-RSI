"""Opt-in launcher: python -m compat SCRIPT [args] or python -m compat -m MODULE [args]."""
import runpy
import sys
from . import enable

enable()
args = sys.argv[1:]
if not args:
    raise SystemExit(__doc__)
if args[0] == '-m':
    if len(args) < 2:
        raise SystemExit('module required after -m')
    sys.argv = args[1:]
    runpy.run_module(args[1], run_name='__main__', alter_sys=True)
else:
    sys.argv = args
    runpy.run_path(args[0], run_name='__main__')
