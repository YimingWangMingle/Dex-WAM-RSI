"""Legacy import bootstrap; new code uses environments.* directly."""
import sys
import environments
from compat.imports import install_legacy_aliases

install_legacy_aliases()
sys.modules[__name__] = environments
