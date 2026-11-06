"""Analytic multiview fixtures and runner integration; no robot or real RGB dataset."""
import json
import sys
import tempfile
import subprocess
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skills.contracts import MeasuredObservation, ControlSkill, SkillIdentity
from tools.perception.geometry import CameraFrame, GeometryUnavailable, color_regions, pixel_ray, triangulate
from tools.perception.tools import CalibratedObservation, GeometryTools, load_bundle, save_bundle
from tools.perception.camera import CalibratedRGBAdapter, GeometryCameraBinding
from execution.runner import Measure, Stop, CallSkill, UpdateTaskState
from skills.direct.guarded import GuardedDirectBackend
from execution.feedback import ObservationEvidence, OperationCheck, OperationReport
from tools.perception.direct_target import measured_point_stage
from experience.packages import load_package, ExperienceError
import test_task_skill_runner as fixtures


def observation(step=0, frame=0):
    return fixtures.obs(step, frame)


def camera(x=0., textured=True, time_s=0.):
    h, w = 96, 128
    k = np.array([[90.,0.,64.],[0.,90.,48.],[0.,0.,1.]])
    t = np.eye(4); t[0,3]=x
    y,u=np.mgrid[:h,:w]
    xx=(u-64)/90 + x; yy=(y-48)/90  # render a horizontal plane at world z=1
    gray=128+40*np.sin(25*xx+7*yy)+30*np.cos(13*yy-4*xx)+20*np.sin(51*xx+34*yy)
    rgb=np.repeat(np.clip(gray,0,255).astype(np.uint8)[...,None],3,axis=2)
    if not textured:
        rgb[:]=127
    return CameraFrame(rgb,k,t,time_s,'fixture-calibration/v1',True,(h,w))


def bundle(obs=None, textured=True):
    o=obs or observation()
    return CalibratedObservation(o, {'head':camera(textured=textured,time_s=o.time_s),
        'wrist':camera(.2,textured=textured,time_s=o.time_s)},'fixture_world',{})


PLANE=dict(camera='head',camera2='wrist',roi=[30,15,105,80],roi2=[0,0,128,96],
           z_min_m=.65,z_max_m=1.35,step_m=.025)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.bundle=bundle()
        self.tools=GeometryTools(lambda o:self.bundle)

    def measure(self,name,**args):
        return self.tools.measure(name,self.bundle.observation,args)

    def test_ray_is_not_a_3d_point(self):
        result=self.measure('rgb_ray',camera='head',uv=[64,48])
        np.testing.assert_allclose(result.data['direction_world'],[0,0,1])
        self.assertFalse(result.data['depth_known'])
        self.assertNotIn('point_world_m',result.data)

    def test_triangulation_recovers_metric_point(self):
        result=self.measure('triangulate_point',camera='head',uv=[64,48],camera2='wrist',uv2=[46,48])
        np.testing.assert_allclose(result.data['point_world_m'],[0,0,1],atol=1e-10)
        self.assertLess(result.data['ray_gap_m'],1e-10)
        self.assertFalse(result.data['correspondence_verified'])
        self.assertFalse(result.report.permits_next_stage(('measurement_computed',)))

    def test_degenerate_or_inconsistent_geometry_returns_unknown(self):
        for uv2 in ([64,48],[80,48],[46,60]):
            with self.subTest(uv2=uv2):
                r=self.measure('triangulate_point',camera='head',uv=[64,48],camera2='wrist',uv2=uv2)
                self.assertEqual(r.data['status'],'unknown')
                self.assertNotIn('point_world_m',r.data)
        self.assertEqual(self.measure('rgb_ray',camera='head',uv=[128,0]).data['status'],'unknown')

    def test_camera_validation_and_transform_direction(self):
        c=camera()
        for change in (dict(rectified_native=False),dict(K=np.eye(2)),dict(T_world_camera=np.diag([2,1,1,1])),
                       dict(K=np.diag([-1,1,1])),dict(calibration_image_hw=(48,64))):
            with self.assertRaises(ValueError):
                replace(c,**change)
        t=np.array([[0,0,1,.3],[0,1,0,.2],[-1,0,0,.1],[0,0,0,1.]])
        origin,direction=pixel_ray(replace(c,T_world_camera=t),[64,48])
        np.testing.assert_allclose(origin,[.3,.2,.1])
        np.testing.assert_allclose(direction,[1,0,0])

    def test_plane_sweep_recovers_height_from_rgb(self):
        result=self.measure('measure_plane',**PLANE)
        self.assertEqual(result.data['status'],'ok',result.data)
        self.assertAlmostEqual(result.data['plane_z_m'],1.,delta=.004)
        self.assertGreater(result.data['ncc'],.95)
        self.assertGreater(result.data['peak_prominence'],.08)
        point=self.measure('pixel_world',camera='head',uv=[64,48],plane_ref=result.data['measurement_id'])
        np.testing.assert_allclose(point.data['point_world_m'],[0,0,1],atol=.004)
        self.assertFalse(point.data['plane_membership_verified'])

    def test_textureless_surface_never_gets_default_height(self):
        self.bundle=bundle(textured=False)
        r=self.measure('measure_plane',**PLANE)
        self.assertEqual(r.data['status'],'unknown')
        self.assertIn('texture',r.data['reason'])
        self.assertNotIn('plane_z_m',r.data)

    def test_boundary_and_invalid_search_are_rejected(self):
        for change in (dict(z_min_m=1.),dict(step_m=0),dict(step_m=.00001),dict(roi=[-1,0,10,10])):
            r=self.measure('measure_plane',**dict(PLANE,**change))
            self.assertEqual(r.data['status'],'unknown',r.data)

    def test_reference_cannot_cross_observations(self):
        plane=self.measure('measure_plane',**PLANE).data['measurement_id']
        self.bundle=bundle(observation(0,1))
        r=self.measure('pixel_world',camera='head',uv=[64,48],plane_ref=plane)
        self.assertEqual(r.data['status'],'unknown')
        self.assertIn('stale',r.data['reason'])

    def test_changed_payload_under_same_identity_is_rejected(self):
        self.measure('rgb_ray',camera='head',uv=[64,48])
        self.bundle=replace(self.bundle,cameras={'head':camera(.01),'wrist':camera(.2)})
        r=self.measure('rgb_ray',camera='head',uv=[64,48])
        self.assertEqual(r.data['status'],'unknown')
        self.assertIn('bundle_changed',r.data['reason'])

    def test_timestamp_skew_and_wrong_bundle_are_rejected(self):
        self.bundle=replace(self.bundle,cameras={'head':camera(time_s=.2),'wrist':camera(.2)})
        self.assertIn('timestamp',self.measure('rgb_ray',camera='head',uv=[64,48]).data['reason'])
        tool=GeometryTools(lambda o:bundle(observation(1)))
        r=tool.measure('rgb_ray',observation(),dict(camera='head',uv=[64,48]))
        self.assertIn('does_not_match',r.data['reason'])

    def test_color_regions_are_image_local_and_not_semantic_ids(self):
        image=np.zeros((96,128,3),dtype=np.uint8)
        image[10:20,20:40]=[255,0,0];image[60:80,90:110]=[255,0,0]
        self.bundle=replace(self.bundle,cameras={'head':replace(camera(),rgb=image)})
        r=self.measure('segment_color',camera='head',lower_rgb=[200,0,0],upper_rgb=[255,20,20])
        self.assertEqual([x['area_px'] for x in r.data['regions']],[200,400])
        self.assertEqual(r.data['regions'][0]['bbox'],[20,10,40,20])
        self.assertFalse(r.data['semantic_identity_verified'])

    def test_loader_requires_explicit_geometry_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); c=camera()
            np.save(root/'head.npy',c.rgb)
            metadata=dict(schema='thinkerwam.calibrated-rgb/v1',observation=vars(observation()),
                world_frame='world',cameras={'head':dict(rgb='head.npy',K=c.K.tolist(),
                    T_world_camera=c.T_world_camera.tolist(),time_s=0,calibration_revision='v1',rectified_native=True,calibration_image_hw=[96,128])})
            (root/'geometry.json').write_text(json.dumps(metadata))
            self.assertEqual(load_bundle(root).cameras['head'].rgb.shape,(96,128,3))
            metadata['cameras']['head']['rgb']='../outside.npy'
            (root/'geometry.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                load_bundle(root)

    def test_raw_adapter_preserves_pixels_and_converts_extrinsics(self):
        c=camera(.2)
        inverse=np.linalg.inv(c.T_world_camera);inverse[:3,3]*=1000
        binding=GeometryCameraBinding('rgb','K','pose','time','factory-v1',
                                      'world_to_camera',.001,True,(96,128),'bgr')
        adapted=CalibratedRGBAdapter({'head':binding},world_frame='world').adapt(
            dict(rgb=c.rgb[...,::-1],K=c.K,pose=inverse,time=0,ignored_object_truth='never read'),observation())
        np.testing.assert_array_equal(adapted.cameras['head'].rgb,c.rgb)
        np.testing.assert_allclose(adapted.cameras['head'].T_world_camera,c.T_world_camera)
        with self.assertRaises(ValueError):
            replace(binding,rectified_native=False)

    def test_snapshot_round_trip_and_cli_measured_plane_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);save_bundle(root/'obs',self.bundle)
            reloaded=load_bundle(root/'obs')
            np.testing.assert_array_equal(reloaded.cameras['head'].rgb,self.bundle.cameras['head'].rgb)
            with self.assertRaises(FileExistsError):
                save_bundle(root/'obs',self.bundle)
            requests=[dict(tool='measure_plane',arguments=PLANE,save_as='table'),
                      dict(tool='pixel_world',arguments=dict(camera='head',uv=[64,48],plane_ref='$table'))]
            (root/'requests.json').write_text(json.dumps(requests))
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('measure_geometry.py')),
                '--bundle',str(root/'obs'),'--requests',str(root/'requests.json')],capture_output=True,text=True,check=True)
            rows=[json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(rows),2)
            self.assertAlmostEqual(rows[1]['data']['point_world_m'][2],1,delta=.004)

    def test_explicit_masks_and_region_projection_share_current_plane(self):
        mask=np.zeros((96,128),dtype=bool); mask[15:80,30:105]=True
        self.bundle=replace(self.bundle,masks={'head/surface':mask,'wrist/surface':np.ones_like(mask)})
        args={k:v for k,v in PLANE.items() if k not in ('roi','roi2')}
        plane=self.measure('measure_plane',**args,mask='surface',mask2='surface')
        self.assertEqual(plane.data['status'],'ok')
        regions=self.measure('segment_color',camera='head',lower_rgb=[0,0,0],upper_rgb=[255,255,255])
        point=self.measure('region_center_on_plane',region_ref=regions.data['measurement_id'],
                           region_index=0,plane_ref=plane.data['measurement_id'])
        self.assertEqual(point.data['status'],'ok')
        self.assertAlmostEqual(point.data['point_world_m'][2],1,delta=.004)
        self.assertFalse(point.data['grasp_point_verified'])
        self.assertTrue(point.data['projection_only'])

    def test_candidate_experience_is_sealed_and_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            package_path=Path(temp)/'candidate'
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('build_geometry_experience.py')),
                '--output',str(package_path),'--package-id','geometry-fixture-v1'],
                capture_output=True,text=True,check=True)
            package=load_package(package_path)
            self.assertEqual(json.loads(result.stdout)['manifest_sha256'],package.manifest_sha256)
            self.assertEqual(package.metadata['publication_status'],'candidate')
            self.assertIn('tools/perception/notes.md',package.files)
            self.assertIn('scripts/test_geometry_perception.py',package.files)
            (package.payload/'tools/perception/notes.md').write_text('changed after sealing')
            with self.assertRaises(ExperienceError):
                load_package(package_path)



class RunnerGeometryTests(unittest.TestCase):
    build=fixtures.RunnerTests.build
    events=fixtures.RunnerTests.events

    def test_agent_parameters_result_and_interface_with_zero_motion(self):
        r=self.build([Measure('triangulate_point',dict(camera='head',uv=[64,48],camera2='wrist',uv2=[46,48])),Stop('done')])
        r.measurement_tools=GeometryTools(lambda o:bundle(o)).registry()
        result=r.run(fixtures.EP)
        self.assertEqual(result['counters']['executed_steps'],0)
        entry=self.events('measurement')[0]
        np.testing.assert_allclose(entry['data']['point_world_m'],[0,0,1],atol=1e-10)
        self.assertIn('uv2',self.agent.views[0].measurement_interfaces['triangulate_point']['parameters'])
        self.assertTrue(any('point_world_m' in m['text'] for m in self.agent.views[-1].messages if m['kind']=='measurement'))

    def test_bad_geometry_returns_control_to_agent(self):
        r=self.build([Measure('triangulate_point',dict(camera='head',uv=[64,48],camera2='wrist',uv2=[64,48])),Stop('new view needed')])
        r.measurement_tools=GeometryTools(lambda o:bundle(o)).registry()
        result=r.run(fixtures.EP)
        self.assertEqual(result['reason'],'agent_stop:new view needed')
        self.assertEqual(self.events('measurement')[0]['data']['status'],'unknown')
        self.assertEqual(self.executor.step,0)

    def test_measured_target_reaches_direct_planner_only_after_verification(self):
        def save_ref(view):
            message=next(m for m in reversed(view.messages) if m['kind']=='measurement')
            reference=json.loads(message['text'])['data']['measurement_id']
            return UpdateTaskState(fix_goal={'fixture_target':reference})
        requests=[Measure('triangulate_point',dict(camera='head',uv=[64,48],camera2='wrist',uv2=[46,48])),
                  save_ref,CallSkill('direct_control','approach',2),fixtures.accept(),Stop('done')]
        r=self.build(requests)
        geometry=GeometryTools(lambda o:bundle(o)); r.measurement_tools=geometry.registry()
        planned=[]
        def verify(request,result):
            evidence=(ObservationEvidence.from_observation(request.observation),)
            return OperationReport('fixture-check',checks=tuple(OperationCheck(n,'passed','fixture/v1',evidence)
                for n in ('target_identity','geometry_applicable')))
        def planner(request,point,frame):
            planned.append((point,frame))
            return ((.1,.2),)
        stage=measured_point_stage('approach',geometry,lambda request:self.state.fixed_goals['fixture_target'],
            verify_target=verify,plan_world_target=planner,
            verify_after=lambda before,after:(OperationCheck('position_reached','passed','fixture/v1',
                (ObservationEvidence.from_observation(after),)),),required_after_checks=('position_reached',))
        backend=GuardedDirectBackend({'approach':(stage,)})
        r.registry._skills['direct_control']=ControlSkill(SkillIdentity('direct_control','point/v1',fixtures.CONTRACT,2),backend)
        r.operation_verifiers['direct_control']=backend.verify_execution
        r.run(fixtures.EP)
        self.assertEqual(len(planned),1)
        np.testing.assert_allclose(planned[0][0],[0,0,1],atol=1e-10)
        self.assertEqual(self.executor.step,1)
        self.assertEqual(backend.index,1)

    def test_target_without_correspondence_verification_is_blocked(self):
        r=self.build([CallSkill('direct_control','approach',2),Stop('verify target')])
        geometry=GeometryTools(lambda o:bundle(o))
        point=geometry.measure('triangulate_point',observation(),dict(camera='head',uv=[64,48],camera2='wrist',uv2=[46,48]))
        planned=[]
        stage=measured_point_stage('approach',geometry,lambda request:point.data['measurement_id'],
            verify_target=lambda req,result:point.report,
            plan_world_target=lambda *args:planned.append(args),verify_after=lambda *args:(),
            required_after_checks=('position_reached',))
        backend=GuardedDirectBackend({'approach':(stage,)})
        r.registry._skills['direct_control']=ControlSkill(SkillIdentity('direct_control','point/v1',fixtures.CONTRACT,2),backend)
        r.run(fixtures.EP)
        self.assertEqual(planned,[])
        self.assertEqual(self.executor.step,0)
        self.assertEqual(len(self.events('proposal_blocked')),1)


if __name__=='__main__':
    unittest.main(verbosity=2)
