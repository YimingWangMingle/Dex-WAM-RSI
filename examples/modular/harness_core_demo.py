"""Exercise proposal review, partial execution and handoff using test doubles."""
from dataclasses import asdict
import json
from agents.context import AgentContext,TaskState
from skills.contracts import BackendProposal,ControlSkill,ModelControlSkill,MeasuredObservation,SkillIdentity,SkillRegistry
from execution.review import ReviewDecision
from execution.runner import CallSkill,ExecutionResult,Review,Stop,TaskSkillRunner

CONTRACT='synthetic-joint-reference29/50hz/v1'

class ScriptedBackend:
    def __init__(self,rows):
        self.rows=rows
        self.feedback=[]

    def reset(self,episode_id):
        self.episode_id=episode_id

    def propose(self,request):
        return BackendProposal(self.rows[:request.max_actions])

    def observe(self,feedback):
        self.feedback.append(feedback)

class ScriptedExecutor:
    """An in-memory protocol double, not a simulator or sensor source."""
    def __init__(self):
        self.step=0

    def current_observation(self):
        return MeasuredObservation('synthetic-episode',f'observation-{self.step}',self.step,
            self.step/50,f'synthetic-fixture-{self.step}')

    def execute(self,rows,*,action_contract):
        if action_contract!=CONTRACT:
            raise ValueError('wrong synthetic action contract')
        self.step+=len(rows)
        return ExecutionResult(rows,self.current_observation(),False,'completed',
            measured={'source':'synthetic-fixture'})

    def in_flight(self):
        return False

    def cancel(self):
        pass

class ScriptedAgent:
    def __init__(self):
        self.turn=0

    def next(self,view):
        self.turn+=1
        if self.turn==1:
            return CallSkill('wholebody_motion','exercise reviewed motion prefix',3)
        if self.turn==3:
            return CallSkill('contact_recovery','exercise code-only recovery handoff',1)
        if self.turn in (2,4):
            return Review(ReviewDecision(view.pending['proposal_id'],'accept',
                'Synthetic fixture: exercise execution protocol only',
                'contract demonstration',2 if self.turn==2 else 1))
        return Stop('synthetic contract demonstration complete')

def main():
    registry=SkillRegistry()
    registry.register(ModelControlSkill(SkillIdentity('wholebody_motion','synthetic/v1',CONTRACT,29,
        'synthetic:no-real-weights'),ScriptedBackend(tuple((v,)*29 for v in (.01,.02,.03)))))
    registry.register(ControlSkill(SkillIdentity('contact_recovery','synthetic-code/v1',CONTRACT,29),
        ScriptedBackend(((0.,)*29,))))
    events=[]
    runner=TaskSkillRunner(registry,ScriptedExecutor(),ScriptedAgent(),
        AgentContext(TaskState('Exercise framework contracts without a robot')),sink=events.append)
    result=runner.run('synthetic-episode')
    result['final_observation']=asdict(result['final_observation'])
    print(json.dumps(dict(mode='synthetic-contract-demo',result=result,event_count=len(events)),indent=2))

if __name__=='__main__':
    main()
