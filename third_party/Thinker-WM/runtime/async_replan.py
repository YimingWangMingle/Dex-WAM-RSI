"""One bounded inference worker; no generated observations enter history."""
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy


def snapshot_history(history):
    """Share immutable measured arrays, detach mutable indexes and encoding caches."""
    result = copy(history)
    for name in ('_rgb', '_state', '_times', '_event_rgb', '_event_state', '_event_times'):
        setattr(result, name, getattr(history, name).copy())
    result.detector = deepcopy(history.detector)
    result._cache_key = result._cache_val = None
    result._sparse_vae_cache = result._encoding_identity = None
    return result


class AsyncReplan:
    def __init__(self):
        self.executor = None
        self.future = None
        self.source_step = None

    @property
    def pending(self):
        return self.future is not None

    def start(self, step, generate, *args):
        if self.pending:
            raise RuntimeError('only one asynchronous prediction may be pending')
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='wam-replan')
        self.source_step = step
        self.future = self.executor.submit(generate, *args)

    def take(self):
        if not self.pending:
            raise RuntimeError('no asynchronous prediction pending')
        future, step = self.future, self.source_step
        try:
            return step, future.result()
        finally:
            self.future = self.source_step = None

    def close(self):
        # CUDA work cannot be cancelled safely; finish before episode reset.
        try:
            if self.pending:
                self.take()
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True)
                self.executor = None
