"""Frozen per-version skill/tool exposure. Host requirements remain separate."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class SelectionPolicy:
    skills: tuple[str, ...]
    tools: tuple[str, ...] = ()
    automatic_tools: tuple[str, ...] = ()
    decision_mode: str = 'fixed'  # fixed reviewer chain or agent-driven pending review

    def __post_init__(self):
        for key in ('skills','tools','automatic_tools'):
            values = getattr(self,key)
            if not isinstance(values,(tuple,list)) or any(not isinstance(v,str) or not v for v in values):
                raise ValueError('policy names must be explicit lists of nonempty strings')
            object.__setattr__(self,key,tuple(values))
            if len(set(values)) != len(values): raise ValueError('duplicate policy capability')
        if not self.skills or self.decision_mode not in ('fixed','agent'):
            raise ValueError('policy requires skills and fixed/agent decision mode')
        if not set(self.automatic_tools) <= set(self.tools):
            raise ValueError('automatic tools must be allowed')

    @classmethod
    def load(cls,path):
        value=json.loads(Path(path).read_text())
        if not isinstance(value,dict) or set(value)-{'skills','tools','automatic_tools','decision_mode'}:
            raise ValueError('unknown selection policy fields; host limits are not candidate options')
        return cls(**value)

    def identity(self):
        config=asdict(self)
        return dict(config=config,sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())
