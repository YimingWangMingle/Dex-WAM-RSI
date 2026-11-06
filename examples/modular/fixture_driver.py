"""CPU fixture for the public interfaces; not a robotics performance benchmark."""
import numpy as np
from environments import ActionContract, Observation, Outcome
from skills.contracts import BackendProposal, ControlSkill, SkillIdentity
from execution.runner import CallSkill, Review
from execution.review import ReviewDecision

CONTRACT = ActionContract('fixture-position/v1',('x',),('m',),10,'absolute','fixture',(-1,),(1,))

class Environment:
    contract = CONTRACT
    def identity(self): return dict(platform='fixture',revision='fixture-v1')
    def reset(self,case): self.step=0;self.x=0.;self.instruction=case['instruction']
    def observe(self):
        return Observation(self.instruction,(self.x,),{'head':np.zeros((8,8,3),np.uint8)},self.step,self.step/10)
    def execute(self,row): self.x=row[0];self.step+=1
    def is_terminal(self): return self.step>=2
    def outcome(self): return Outcome(self.x>=.5,self.is_terminal(),self.x)
    def close(self): pass

class Backend:
    def __init__(self,target): self.target=target
    def reset(self,episode): pass
    def propose(self,request): return BackendProposal(((self.target,),))
    def observe(self,feedback): pass

class Agent:
    def next(self,view):
        if view.pending:
            return Review(ReviewDecision(view.pending['proposal_id'],'accept','fixture approval','move',1))
        return CallSkill('direct_control','move',1)

def options(target):
    return dict(skill_factories=[lambda load,budget:ControlSkill(
        SkillIdentity('direct_control','fixture-v1',CONTRACT.name,1),Backend(target))],
        agent_factory=lambda load,budget:Agent())

def create_environment(config): return Environment()
def build_episode(env,config): return options(config.get('target',.6))
