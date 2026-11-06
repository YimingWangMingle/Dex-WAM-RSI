# Calibrated RGB 感知工具接口（calibrated-rgb/v1）

输入是当前不可变观测的原始、已去畸变 RGB、相机内参、相机到公共世界系变换和采集时间。长度统一为米，像素顺序为 `[u,v]`。不接收仿真物体真值、任务资产、WAM 预测图像或原生任务评分。

## 六个已实现命令

- `rgb_ray(camera, uv)`：输出世界系射线原点和单位方向。一个像素没有深度，不输出虚构三维点。
- `triangulate_point(camera, uv, camera2, uv2)`：同一快照的两相机像素三角测量。检查射线夹角、正向交点和最近距离；输出三维点、射线间距和夹角。默认最小夹角 3°、最大射线间距 5 mm，作为起始数值门槛，尚未通过真实机器人标定。`correspondence_verified=false`，不会自动宣称两个像素是同一个物理点。
- `measure_plane(camera, camera2, roi/ mask, roi2/ mask2, z_min_m, z_max_m, step_m)`：在显式选择的两个表面区域中搜索水平面高度，用平面诱导投影后的 RGB 灰度归一化相关性选峰，并局部细化。返回高度、NCC、峰值显著度、有效样本数、搜索分辨率。无默认桌高。可用于桌面或平坦物体顶部，但调用者必须正确选择同一表面；不会自动识别桌子。
- `pixel_world(camera, uv, plane_ref)`：射线与当前观测中实测平面求交。`plane_ref` 只能来自成功的 measure_plane，不接受临时猜测的 z。输出 `plane_membership_verified=false`；必须检查像素确实属于该平面。
- `segment_color(camera, lower_rgb, upper_rgb, min_area=40)`：原始 RGB 范围筛选及四连通区域，返回 bbox、质心、像素面积和边缘截断标记。无神经检测器，不能确认语义身份、分开相接同色物体或补全遮挡。
- `region_center_on_plane(region_ref, region_index, plane_ref)`：将某颜色区域的图像质心投到已测平面。仅用于粗定位；区域可能包含物体侧面，投影不等于实际中心或抓取点。这个结果不能作为 measured_point_stage 的直接目标。

工具返回 `MeasurementResult(report, data)`；Agent 通过 `Measure(tool, arguments)` 调用。
报告进入后端共享工具调用链、Agent 上下文和 JSONL 事件。参数说明通过 `AgentView.measurement_interfaces` 提供。旧的不带参数的测量回调继续可用。

## 原始观测契约

模型输入常经过多相机拼接、裁剪、缩放。几何工具使用单相机原始图像，不能直接套用模型输入像素。

每个 `CameraFrame` 必须提供：

- uint8 H×W×3 RGB；已去畸变，且标定对应本图像；
- `K`：3×3 pinhole 内参，包含像素尺度焦距和主点；
- `T_world_camera`：4×4，相机光学系到公共世界系；光学轴 +Z 向前；旋转正交且右手；平移为米；
- `calibration_image_hw`：标定对应的图像高宽，必须与实际图像相同；
- `calibration_revision` 和秒单位的 `time_s`；
- `rectified_native=true`：该字段是采集适配器承诺，不由工具自动去畸变。

`CalibratedObservation` 还带 MeasuredObservation 身份、world_frame 和可选 image-derived mask（key 为 `camera/region`，bool HW）。禁止从任务资产生成 mask。工具不能证明调用方提交的内容来源真实，采集和访问权限由观测适配器保证。

`CalibratedRGBAdapter` 显式绑定原始 RGB、K、外参和时间字段；可转换 BGR、毫米外参以及 world-to-camera 约定，不猜测现场键名。两个相机与观测的时间差，以及彼此时间差，默认最多 20 ms；动态场景需按实际运动收紧。腕相机位姿须来自与图像同步的机器人状态/FK。

## 结果、缓存与失败

成功结果含 method_revision、measurement_id、观测 ID、标定版本、world_frame、单位和数值诊断。未知结果含稳定原因文本与恢复建议，不填默认几何。

所有初始工具返回 `action_ready=false`：几何一致性通过不能证明语义目标、表面关联、抓持或任务成功。报告把 measurement_computed 与 target_surface_association 分开，后者默认 unknown。

measurement_id 只在同一观测中有效。新图片（即使机器人步号不变）、跨 episode 或校准/图像内容变化都会使引用失效。同一观测 ID 下更换 bundle 内容被拒绝。历史日志仍可用于诊断，但不能当作当前坐标缓存直接重用。

弱纹理、视野重叠不足、峰值在搜索边界、峰值弱或多义、射线近平行、交点在相机后方、射线不一致、过期引用、时戳或标定错误均返回 unknown。移动相机重新观察必须通过动作技能和统一执行器。

NCC、射线间距和搜索步长都不是经过标定的概率或三维误差界。平面测高目前是两视角、显式区域、全局 NCC 的基础实现，不等同于 HexaShell 的多相机共识与局部 NCC 工具。

## 与 direct 动作代码连接

`measured_point_stage` 读取当前 snapshot 的三角测量或 pixel_world 引用，把世界系点和显式米单位偏移交给注入的机器人规划器。它要求独立前置检查明确通过 `target_identity` 与 `geometry_applicable`，之后仍使用 GuardedDirectBackend 原有提案、审核、执行和后置验证。

几何适用性应检查：双视角对应关系或平面成员关系、操作所需精度、标定可信度，以及目标相对当前夹爪的可用性。规划器负责公共世界系到机器人基座系、TCP/EEF、IK、运动约束和动作契约；本模块不实现这些机器人特定功能。

不要自动把一个成功的测量 report 改写成“目标已验证”。相关 CPU 测试使用显式 fixture 验证器，只证明控制流程，不证明真实视觉判断。

## 记录、离线复现与共享经验

`save_bundle` 写入新的 geometry.json 与 RGB/mask 文件；`load_bundle` 读取该显式格式，不覆盖旧观测。入口：

```bash
python scripts/measure_geometry.py --bundle <recorded-bundle> --requests <requests.json>
```

requests.json 是 `{tool, arguments, save_as?}` 的数组，后续参数可用 `$alias` 引用前一成功结果。例如先 measure_plane 保存为 table，再 pixel_world 的 plane_ref 填 `$table`。该入口只处理观测，不发机器人动作。

执行经验见同目录 notes.md；计算实现见 geometry.py / tools.py；控制连接见 direct_target.py；测试见 scripts/test_geometry_perception.py。用 scripts/build_geometry_experience.py 将这些指定文件封装成现有 ExperiencePackage 供按任务显式映射使用；经验文件只供读取，不能绕过运行时工具版本和审核直接执行代码。
