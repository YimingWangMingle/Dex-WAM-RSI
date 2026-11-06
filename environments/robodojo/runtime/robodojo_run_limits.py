"""Optional development budgets; zero disables a budget, not execution guards."""
import math


def read_limits(environ):
    calls = int(environ.get('RSI_MAX_AGENT_CALLS', '0'))
    seconds = float(environ.get('RSI_MAX_EPISODE_WALL_SECONDS', '0'))
    if calls < 0 or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Agent budgets must be finite and nonnegative; 0 means unlimited')
    return calls, seconds


def exceeded(calls, elapsed, limits):
    max_calls, max_seconds = limits
    if max_calls and calls >= max_calls:
        return 'agent_call_budget'
    if max_seconds and elapsed >= max_seconds:
        return 'episode_wall_time_budget'
    return None
