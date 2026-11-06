"""Shared WAM final-action forecast implementation used by fixed and tool paths."""
import copy
from dataclasses import replace


def forecast_final_plan(policy, plan, output_dir):
    import numpy as np
    from thinker_wm.models.inference_api import action_hash
    from environments.robodojo.runtime.robodojo_a2v import save_forecast, segment_frames
    inference=copy.copy(policy.inf)
    inference.settings=replace(policy.inf.settings,decode_outputs=True)
    forecast=inference.predict_outcome(policy.observation,policy.history,policy.instruction,plan.copy())
    if (forecast.decision_step!=policy.observation.step or forecast.action_hash!=action_hash(plan)
            or not np.array_equal(forecast.actions,plan)):
        raise ValueError('forecast is not bound to the final candidate')
    media=save_forecast(forecast,output_dir,policy.control_hz)
    media['segments']=segment_frames(media['frame_steps'],policy.observation.step,policy.block_size)
    return media
