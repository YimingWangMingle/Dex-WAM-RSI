"""Motus training monitor library.

Training side (cluster)::

    from thinker_wm.monitor import create_monitor
    monitor = create_monitor(config, run_name=..., checkpoint_dir=..., ...)
    monitor.log_step_record(step_record)      # per-step metrics + timings
    monitor.log_checkpoint(step, path, timing)
    monitor.close(state="completed")

Local side (this machine)::

    python -m monitor serve --port 8377 \
        --remote-target fengodchen@localhost \
        --remote-dir /code1/llm_team/chen.liao/ThinkerWM/motus_monitor_data \
        --ssh-args "-J chen.liao@10.10.94.35 -p 10023"

Everything is stdlib-only on the server side and a strict no-op on the
training side unless enabled (``monitor.enabled: true`` in the YAML config or
``MOTUS_MONITOR_ENABLED=1``).
"""

from .client import MonitorLogger, NullMonitor, create_monitor
from .paths import default_data_dir

__version__ = "0.1.0"

__all__ = [
    "MonitorLogger",
    "NullMonitor",
    "create_monitor",
    "default_data_dir",
    "__version__",
]
