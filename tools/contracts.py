"""Read-only task tool boundary and bounded, action-bound prediction receipts.

Callbacks are trusted adapters. They receive immutable observations/candidates,
never the executor. Tool calls do not constitute review approval.
"""
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import time
from typing import Callable

from skills.contracts import ActionCandidate, MeasuredObservation


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ToolLimits:
    max_calls: int = 32
    max_candidates: int = 3
    max_action_rows: int = 24
    max_seconds: float = 900
    max_result_bytes: int = 2_000_000

    def __post_init__(self):
        for key in ('max_calls','max_candidates','max_action_rows','max_result_bytes'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:raise ValueError('positive tool limits required')
        if isinstance(self.max_seconds,bool) or not math.isfinite(self.max_seconds) or self.max_seconds<=0:
            raise ValueError('positive tool duration required')


@dataclass(frozen=True)
class ToolRequest:
    observation: MeasuredObservation
    proposal_id: str
    action_contract: str
    candidates: tuple[ActionCandidate, ...]

    def binding(self):
        return dict(observation=asdict(self.observation),proposal_id=self.proposal_id,
                    action_contract=self.action_contract,
                    candidates=[dict(candidate_id=c.candidate_id,actions=c.actions) for c in self.candidates])


@dataclass(frozen=True)
class TaskTool:
    name: str
    revision: str  # Must bind predictor/checkpoint/config identity in production.
    run: Callable
    kind: str = 'action_prediction'

    def __post_init__(self):
        if not self.name or not self.revision or not callable(self.run) or self.kind!='action_prediction':
            raise ValueError('this tool boundary currently supports declared action predictors')

    def interface(self):
        return dict(name=self.name,revision=self.revision,kind=self.kind,moves_robot=False,
            arguments=dict(proposal_id='string',observation_id='string',candidate_ids='unique integer list'))


class ToolBudgetExceeded(RuntimeError): pass


class ToolSession:
    def __init__(self, tools, *, limits=ToolLimits()):
        self.tools=dict(tools or {})
        if any(not isinstance(t,TaskTool) or name!=t.name for name,t in self.tools.items()):
            raise ValueError('task tools require matching registered names')
        self.limits=limits
        self.calls=0
        self.by_tool={}
        self.receipts=[]

    def snapshot(self):
        return dict(calls=self.calls,by_tool=dict(self.by_tool),limits=asdict(self.limits))

    def invoke(self,name,request):
        tool=self.tools[name]
        candidates=request.candidates
        if (not candidates or len(candidates)>self.limits.max_candidates or
                len({c.candidate_id for c in candidates})!=len(candidates) or
                any(type(c.candidate_id) is not int or not c.actions or len(c.actions)>self.limits.max_action_rows
                    or any(not all(math.isfinite(v) for v in row) for row in c.actions) for c in candidates)):
            raise ValueError('prediction candidates exceed tool contract/limits')
        if self.calls>=self.limits.max_calls: raise ToolBudgetExceeded('tool_call_budget_exhausted')
        binding=copy.deepcopy(request.binding())
        self.calls+=1; self.by_tool[name]=self.by_tool.get(name,0)+1
        start=time.monotonic()
        data=tool.run(request)
        elapsed=time.monotonic()-start
        if elapsed>self.limits.max_seconds: raise TimeoutError('tool duration exceeded; result not usable')
        encoded=json.dumps(data,allow_nan=False).encode()
        if len(encoded)>self.limits.max_result_bytes:raise ValueError('tool result exceeds limit')
        if not isinstance(data,dict) or set(data)!={'predictions'} or not isinstance(data['predictions'],list):
            raise ValueError('tool must return a predictions list')
        ids=[]
        for item in data['predictions']:
            if not isinstance(item,dict) or set(item)!={'candidate_id','forecast'} or not isinstance(item['forecast'],dict) or not item['forecast']:
                raise ValueError('prediction needs candidate_id and nonempty forecast')
            if type(item['candidate_id']) is not int:raise ValueError('integer prediction ID required')
            ids.append(item['candidate_id'])
        if sorted(ids)!=sorted(c.candidate_id for c in candidates):raise ValueError('prediction IDs differ from request')
        receipt=dict(tool=name,revision=tool.revision,binding=binding,binding_sha256=digest(binding),
                     elapsed_s=elapsed,data=copy.deepcopy(data))
        receipt['receipt_id']=digest(receipt)
        self.receipts.append(copy.deepcopy(receipt))
        return receipt

    def matching(self,proposal,latest,rows,candidate_id,tool="a2v"):
        for receipt in reversed(self.receipts):
            b=receipt['binding']
            if receipt['tool'] != tool: continue
            if (b['proposal_id']!=proposal.proposal_id or b['observation']!=asdict(latest) or
                    b['action_contract']!=proposal.identity.action_contract):continue
            for c in b['candidates']:
                if c['candidate_id']==candidate_id and tuple(tuple(r) for r in c['actions'])==proposal.actions and tuple(rows)==proposal.actions[:len(rows)]:
                    return receipt['receipt_id']
        return None

    def clear(self): self.receipts.clear()
