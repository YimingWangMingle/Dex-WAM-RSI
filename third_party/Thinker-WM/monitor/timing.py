"""Lightweight section timing helpers for model internals.

The training loop uses :class:`TimingStack` for hierarchical, stack-based
timing.  ``TimingStack`` acts as a dict for backward compatibility (so
existing ``timed_section`` and bare ``timing[key]=…`` calls still work),
but also provides a ``span()`` context manager that captures a proper
containment tree.

    with timing.span("train_step"):
        with timing.span("zero_grad"):
            optimizer.zero_grad()
        with timing.span("forward"):
            loss = model.training_step(..., timing=timing)

The tree is exposed via :meth:`TimingStack.to_tree`, and the flat dict
(via :meth:`TimingStack.to_flat_dict`) keeps the existing key format
for the monitor front-end.

It lives in the monitor package (not ``models/``) because it is part of the
observability plumbing, but it is imported only by training-side model code
(the monitor web server never imports it, so the torch dependency is safe).

Design notes:

* The set of checkpoints is intentionally NOT a fixed schema -- different code
  versions emit different keys, and downstream consumers (monitor web UI)
  discover keys from the data itself.
* Per-block timing is gated by a detail level (``timing_level >= 2``) because
  each section costs two CUDA synchronizations when ``sync_cuda`` is on.
* ``timing=None`` makes every wrapper a pure no-op with negligible overhead.
* ``TimingStack(enabled=False)`` makes ``span()`` a no-op (for interval-based
  sampling).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

import torch


# ---------------------------------------------------------------------------
# TimingSpan -- a single node in the timing tree
# ---------------------------------------------------------------------------

class TimingSpan:
    """One timing interval in the stack-based tree."""

    __slots__ = ("name", "start", "end", "children", "metadata")

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.start: float = 0.0
        self.end: float = 0.0
        self.children: List[TimingSpan] = []
        self.metadata: Dict[str, Any] = {}

    @property
    def total(self) -> float:
        """Wall-clock duration of this span (seconds)."""
        return max(0.0, self.end - self.start)

    @property
    def children_sum(self) -> float:
        """Sum of all direct children's ``total``."""
        return sum(c.total for c in self.children)

    @property
    def self_time(self) -> float:
        """Time *not* accounted for by any child (unexplained gap)."""
        return max(0.0, self.total - self.children_sum)

    def to_dict(self) -> Dict[str, Any]:
        """Recursively serialise to a plain dict (JSON-safe)."""
        result: Dict[str, Any] = {
            "name": self.name,
            "total_seconds": self.total,
            "self_seconds": self.self_time,
        }
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        if self.metadata:
            result["meta"] = self.metadata
        return result


# ---------------------------------------------------------------------------
# TimingStack -- hierarchical timer with dict-like backward compat
# ---------------------------------------------------------------------------

class TimingStack:
    """Stack-based hierarchical timing with dict-like backward compatibility.

    Usage::

        timing = TimingStack(sync_cuda=True, enabled=True)
        with timing.span("train_step"):
            with timing.span("zero_grad"):
                ...
        # Flat dict for backward compat (keys: "train_step_seconds", ...)
        record.update(timing.to_flat_dict())
        # Tree for future flame-graph visualisation
        tree = timing.to_tree()

    When ``enabled=False`` every operation is a no-op (cheap enough to call
    at every step when using interval-based sampling).
    """

    def __init__(
        self,
        sync_cuda: bool = False,
        enabled: bool = True,
    ) -> None:
        self._sync_cuda = sync_cuda
        self._enabled = enabled
        self._stack: List[TimingSpan] = []
        self._roots: List[TimingSpan] = []
        # Flat dict for backward compat (manual keys + span-accumulated keys)
        self._flat: Dict[str, float] = {}

    # -- context manager ---------------------------------------------------

    @contextmanager
    def span(
        self,
        name: str,
        sync_cuda: Optional[bool] = None,
    ) -> Iterator[TimingSpan]:
        """Time a code block.

        Parameters
        ----------
        name:
            Logical name.  The flat-key ``{name}_seconds`` is automatically
            accumulated into the backward-compat dict.
        sync_cuda:
            Override the stack-level CUDA-sync setting for this span only.
        """
        if not self._enabled:
            yield _NOOP_SPAN
            return
        do_sync = sync_cuda if sync_cuda is not None else self._sync_cuda
        if do_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        node = TimingSpan(name)
        node.start = time.perf_counter()
        self._stack.append(node)
        try:
            yield node
        finally:
            if do_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            node.end = time.perf_counter()
            self._stack.pop()
            if self._stack:
                self._stack[-1].children.append(node)
            else:
                self._roots.append(node)
            flat_key = f"{name}_seconds"
            self._flat[flat_key] = self._flat.get(flat_key, 0.0) + node.total

    # -- dict-like interface (backward compat) -----------------------------

    def __getitem__(self, key: str) -> float:
        return self._flat[key]

    def __setitem__(self, key: str, value: float) -> None:
        # 与 update() 一致：非数值（如 nonfinite 检查报告里的 None 占位键）
        # 原样保存，不因 float(None) 崩训练（对齐引入 TimingStack 前 plain
        # dict 的行为）；消费侧（step_record JSONL/前端）按名容错解析。
        self._flat[key] = float(value) if isinstance(value, (int, float)) else value

    def __contains__(self, key: str) -> bool:
        return key in self._flat

    def get(self, key: str, default: float = 0.0) -> float:
        """Mirrors ``dict.get`` for ``timed_section`` compatibility."""
        return self._flat.get(key, default)

    def items(self):
        """Iterate ``(key, value)`` pairs (enables ``dict.update(timing)``)."""
        return self._flat.items()

    def update(self, other: Dict[str, Any]) -> None:
        """Merge another dict's keys into the flat store."""
        for k, v in other.items():
            self._flat[k] = float(v) if isinstance(v, (int, float)) else v

    def keys(self):
        return self._flat.keys()

    def values(self):
        return self._flat.values()

    # -- serialisation -----------------------------------------------------

    def to_flat_dict(self) -> Dict[str, float]:
        """Return a copy of the flat ``{key: seconds}`` dictionary.

        This is the backward-compat format expected by ``step_timing`` JSONL
        and the monitor front-end.
        """
        return dict(self._flat)

    def to_tree(self) -> List[Dict[str, Any]]:
        """Return the root spans as a list of nested dicts (flame-graph ready).

        Each node: ``{name, total_seconds, self_seconds, children?, meta?}``.
        """
        return [r.to_dict() for r in self._roots]

    # -- housekeeping ------------------------------------------------------

    def clear(self) -> None:
        """Reset all state for reuse (e.g. between steps)."""
        self._stack.clear()
        self._roots.clear()
        self._flat.clear()

    @property
    def enabled(self) -> bool:
        return self._enabled


# Singleton no-op span returned when TimingStack.enabled == False
_NOOP_SPAN = TimingSpan("")


# ---------------------------------------------------------------------------
# Legacy helpers (kept for branch code that hasn't migrated to TimingStack)
# ---------------------------------------------------------------------------

@contextmanager
def timed_section(
    timing: Optional[Dict[str, float]],
    name: str,
    sync_cuda: bool = False,
) -> Iterator[None]:
    """Accumulate elapsed seconds into ``timing[name]``.

    Works with both a plain ``dict`` and a :class:`TimingStack` (which
    implements the dict interface).

    Repeated sections with the same name are summed (e.g. per-block loops use
    one key per block index, but accumulation keeps multi-call totals correct).
    """
    if timing is None:
        yield
        return
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        timing[name] = float(timing.get(name, 0.0)) + (time.perf_counter() - start)


def block_timing(
    timing: Optional[Dict[str, float]],
    timing_level: int,
    minimum: int = 2,
) -> Optional[Dict[str, float]]:
    """Return the timing dict only when per-block checkpoints are enabled.

    Use as::

        bt = block_timing(timing, timing_level)
        for i, block in enumerate(self.blocks):
            with timed_section(bt, f"wm_xxx_block_{i:02d}_seconds", sync_cuda):
                ...
    """
    if timing is not None and int(timing_level) >= minimum:
        return timing
    return None
