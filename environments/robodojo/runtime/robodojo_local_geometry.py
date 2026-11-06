"""RGB correspondences + actual camera calibration; never read task object poses.

CameraManager's parent Xform is NOT necessarily the rendered optical frame.
Use the rendered Camera's ROS view matrix and native pinhole intrinsics instead.
"""
import numpy as np
from tools.perception.geometry import CameraFrame, triangulate, project
from environments.robodojo.runtime.robodojo_motion_plan import numpy


def qualified_rgb_intrinsics(name, sdk_k, image_hw, *, tiled_capture):
    """Use renderer-qualified K, not an SDK algebra round-trip as qualification.

    RoboDojo's native tiled 640x480 head render uses square pixels despite the
    SDK's independently authored vertical aperture. Six isolated rendered
    markers qualified fy=fx (0.35..1.18px vs 9..33px SDK errors, 2026-10-22).
    Restrict the correction to that exact camera/intrinsics/capture signature.
    Never silently apply this rule to arbitrary cameras or changed resolutions.
    """
    k=np.array(sdk_k,dtype=float,copy=True)
    if name=='cam_head' and not np.isclose(k[0,0],k[1,1],rtol=1e-4):
        signature=np.array([[288.13253784,0,320],[0,336.46432495,240],[0,0,1]])
        if not tiled_capture or tuple(image_hw)!=(480,640) or not np.allclose(k,signature,atol=.01,rtol=0):
            raise ValueError('head_renderer_intrinsics_not_qualified_for_this_configuration')
        k[1,1]=k[0,0]
        return k,'robodojo-tiled-head-640x480-renderer-qualified-20261022'
    return k,'sdk-pinhole-square-pixel-v1'


class LocalGeometry:
    def __init__(self, env):
        self.env = env
        self.points = {}
        self.serial = 0

    def camera(self, name, obs, step):
        try:
            return self._camera(name, obs, step)
        except Exception as error:
            raise ValueError('camera_geometry_unavailable: '+type(error).__name__+': '+str(error)) from error

    def _camera(self, name, obs, step):
        manager = self.env.obs_manager.camera_manager
        index = manager.camera_names[0].index(name)
        camera = manager.cameras[0][index]
        image = np.asarray(obs['vision'][name]['color'])[..., :3].astype(np.uint8)
        if tuple(camera.get_resolution()) != (image.shape[1], image.shape[0]):
            raise ValueError('camera_resolution_mismatch; no implicit resizing')
        k = numpy(camera.get_intrinsics_matrix())
        if np.allclose(k, np.eye(3)):
            raise ValueError('identity_intrinsics_not_calibration')
        capture=getattr(self.env,'capture_manager',None)
        k,revision=qualified_rgb_intrinsics(name,k,image.shape[:2],
            tiled_capture=type(capture).__name__=='TiledCaptureManager')
        transform = np.linalg.inv(numpy(camera.get_view_matrix_ros()))
        return CameraFrame(image, k, transform, step/float(self.env.obs_manager.collect_freq),
                           revision+':'+name, True, image.shape[:2])

    def measure(self, obs, step, camera, uv, camera2, uv2, label):
        if camera == camera2 or not isinstance(label, str) or not label.strip():
            raise ValueError('two distinct cameras and a physical feature label required')
        first, second = self.camera(camera, obs, step), self.camera(camera2, obs, step)
        result = triangulate(first, uv, second, uv2)
        xyz = np.asarray(result['point_world_m'])
        errors = []
        for frame, pixel in ((first, uv), (second, uv2)):
            predicted, front = project(frame, xyz[None])
            errors.append(float(np.linalg.norm(predicted[0]-pixel)))
            if not front[0] or errors[-1] > 3:
                raise ValueError('reprojection_error_exceeds_3px')
        self.serial += 1
        identity = f'point-{step}-{self.serial}'
        result.update(measurement_id=identity, control_step=step, label=label,
                      cameras=[camera,camera2], pixels=[uv,uv2], reprojection_errors_px=errors,
                      source='RGB_correspondences_and_camera_calibration',
                      correspondence_verified=False, grasp_success=False,
                      warning='Ray agreement is not proof of correct feature identity or millimetre accuracy.')
        self.points[identity] = result
        return result

    def current(self, identity, step):
        result = self.points.get(identity)
        if result is None or result['control_step'] != step:
            raise ValueError('missing_or_stale_point; remeasure in current images')
        return result

    def check_following(self, before_id, after_id, before_tcp, after_tcp, step):
        before = self.points.get(before_id)
        after = self.current(after_id, step)
        if before is None or before['label'] != after['label'] or before['control_step'] >= step:
            raise ValueError('same labelled physical feature required before and after lift')
        tcp_delta = np.asarray(after_tcp)-np.asarray(before_tcp)
        object_delta = np.asarray(after['point_world_m'])-np.asarray(before['point_world_m'])
        residual = float(np.linalg.norm(object_delta-tcp_delta))
        return dict(object_displacement_m=object_delta.tolist(), tcp_displacement_m=tcp_delta.tolist(),
                    following_residual_m=residual,
                    follows_lift=bool(tcp_delta[2] >= .01 and object_delta[2] >= .005 and residual <= .01),
                    native_task_success=False,
                    caveat='Conditional on correct same-feature correspondence; not proof of stable grasp.')
