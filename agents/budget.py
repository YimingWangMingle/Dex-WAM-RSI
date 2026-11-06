"""Per-episode decision budget shared by controllers and review callbacks."""
from contextvars import ContextVar
from threading import Lock


class ModelBudgetExhausted(BaseException):
    """Escape generic skill-error recovery without submitting another request."""


class ModelBudget:
    def __init__(self, limit: int = 170):
        if type(limit) is not int or limit < 1:
            raise ValueError('positive model call limit required')
        self.limit = limit
        self.used = 0
        self.by_role = {}
        self._lock = Lock()
        self._credit = ContextVar('model_budget_credit', default=False)

    def charge(self, role: str) -> None:
        with self._lock:
            if self.used >= self.limit:
                raise ModelBudgetExhausted('model_budget_exhausted')
            self.used += 1
            self.by_role[role] = self.by_role.get(role, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            return dict(limit=self.limit, used=self.used, by_role=dict(self.by_role))

    def client(self, client, role: str):
        budget = self
        class MeteredClient:
            def complete(self, *args, **kwargs):
                if budget._credit.get():
                    budget._credit.set(False)
                else:
                    budget.charge(role)
                return client.complete(*args, **kwargs)
        return MeteredClient()

    def callback(self, callback, role: str):
        if callback is None:
            return None
        def invoke(*args, **kwargs):
            # Reserve before publishing a file-review request. A metered client
            # inside the callback reuses this first reservation; retries cost more.
            self.charge(role)
            token = self._credit.set(True)
            try:
                return callback(*args, **kwargs)
            finally:
                self._credit.reset(token)
        return invoke
