"""Reuse a measured 3D point in a guarded direct stage without owning motion.

A robot-specific planner and an independent target/applicability verifier must
be supplied. A numeric geometry result alone never unlocks the planner.
"""
import numpy as np
from skills.direct.guarded import DirectStage
from execution.feedback import OperationReport
from tools.perception.geometry import GeometryUnavailable


def measured_point_stage(name, geometry, reference_for_request, *, verify_target,
                         plan_world_target, verify_after, required_after_checks,
                         offset_world_m=(0.,0.,0.)):
    offset=np.asarray(offset_world_m,dtype=float)
    if offset.shape!=(3,) or not np.isfinite(offset).all():
        raise ValueError('finite 3D world offset in metres required')

    def target(request):
        result=geometry.resolve(reference_for_request(request),request.observation)
        if result['tool'] not in ('triangulate_point','pixel_world') or 'point_world_m' not in result:
            raise GeometryUnavailable('direct target requires a point measurement, not a ray/region projection')
        return result

    def precondition(request):
        try:
            result=target(request)
        except (GeometryUnavailable,KeyError) as error:
            return OperationReport(name,failure_kind='perception',failure_detail=str(error),
                                   recovery_options=('remeasure_target',))
        # Must check both identity and geometric applicability (correspondence,
        # plane membership, precision required by the operation, etc.).
        return verify_target(request,result)

    def plan(request):
        result=target(request)
        point=np.asarray(result['point_world_m'])+offset
        return plan_world_target(request,tuple(point),result['world_frame'])

    return DirectStage(name,plan,verify_after,tuple(required_after_checks),precondition,
                       ('target_identity','geometry_applicable'))
