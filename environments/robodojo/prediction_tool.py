"""Dedicated RoboDojo A2V service adapter; never selects or executes actions."""
from tools.contracts import TaskTool, digest


def create_a2v_tool(client, session):
    from environments.robodojo.framework import WAMCandidates
    from execution.trajectory import action_hash
    import numpy as np
    world = WAMCandidates(client, session, mode3=False, predict=False)
    identity=client.call('framework_prediction_identity')
    world.reset(session.episode_id)

    def predict(request):
        if request.action_contract != world.identity['action_contract']:
            raise ValueError('prediction action contract mismatch')
        world.sync_observation(request.observation)
        plans = [np.asarray(c.actions,dtype=np.float32) for c in request.candidates]
        result = client.call('framework_predict',dict(decision_step=request.observation.control_step,
            plans=[p.tolist() for p in plans],candidate_ids=[c.candidate_id for c in request.candidates]))
        if result['decision_step'] != request.observation.control_step:
            raise ValueError('stale prediction response')
        returned = result['candidates']
        if len(returned) != len(plans): raise ValueError('missing prediction candidates')
        predictions=[]
        for c,plan,item in zip(request.candidates,plans,returned):
            if item['candidate_id']!=c.candidate_id or item['action_hash']!=action_hash(plan):
                raise ValueError('prediction action identity mismatch')
            predictions.append(dict(candidate_id=c.candidate_id,forecast=item['forecast']))
        return dict(predictions=predictions)
    return TaskTool('a2v',digest(identity),predict)
