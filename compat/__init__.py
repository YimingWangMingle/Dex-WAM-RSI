"""Optional legacy import support. New code never imports this package."""
from pathlib import Path
import sys

def enable():
    from .imports import install_legacy_aliases
    legacy = str(Path(__file__).resolve().parent / 'legacy')
    if legacy not in sys.path:
        sys.path.insert(0, legacy)
    install_legacy_aliases()
