# AIST++：动作约定、质检与已知问题

运行入口、CSV字段和输出结构见[流水线说明](README.md)，RGB裁剪见[人物视频组件](components/person_video/README.md)。
本文保留AIST特有的处理约定和重要实验结论，不再维护已删除的旧调试脚本、下载审计或历史目录入口。

## 输入与重定向

本地原始视频为c01机位，动作在`data/AIST++/archives/motions_complete.zip`中。
1,408条原始序列排除官方ignore list的45条后，CSV包含1,363条候选；候选不等于质检通过。
视频访问仍需申请AIST授权并遵守条款，派生数据不改变原始许可。

当前解码器为`stages/motion.py`，实际GMR后端为
`backends/aist_smpl_gmr.py`，通过独立模块子进程调用以隔离旧SMPL依赖。

- 输入为`smpl_poses[T,72]`、`smpl_trans[T,3]`、`smpl_scaling`，姿态是24个关节的局部轴角。
- 使用原生SMPL前向运动学，不需要先拟合为SMPL-X；全局朝向沿parent tree累乘。
- 标准SMPL中用`transl=smpl_trans/smpl_scaling`恢复模板米制空间，不能固定除以100或重复缩放身高。
- Y-up转Z-up用`Rx(+90°)`，即`(x,y,z)→(x,-z,y)`；全局旋转左乘同一变换，不能对每个局部旋转重复施加。
- GMR输入为身体节点的全局位置与`wxyz`四元数，输出为G1的36维状态。
  首帧只重复求解预热，不增加帧、不删首帧；整段重定向完成后再切episode。
- 首帧骨盆水平位置移到原点，逆变换相关参数保存在`reference/report.json`。
- 当前地面处理是根据整段OMG脚底proxy最低点做一次常量root-z平移，目标余量5 mm。
  这不会压平相对跳跃高度，但不保证首帧真实接触，也不是手撑地接触优化。

当前明确使用neutral SMPL近似，而非AIST的male模型；`allow_nonofficial_model: true`
是显式原型选项，不能把本地结果宣称为官方预处理或正式男体标签。
GMR求解几何、OMG FK和tracker碰撞几何不能仅因关节名一致就视为完全等价；
历史小角度测试中，共同身体节点最大位置差约1 cm，仍需核对几何与关节朝向。

## 动作与时间轴契约

```text
qpos_36[T,36] = root_xyz[3] + root_quat_wxyz[4] + G1关节角[29]
```

单位为米/弧度，Z向上，关节顺序采用`G1_JOINT_NAMES`。
GMR某些默认保存入口会将根四元数写为xyzw，不能直接拼入本契约。

源动作是精确60Hz，参考动作取源帧`0,2,4,...`形成30Hz时间轴。
原片可能是`60000/1001` FPS，视频必须按真实PTS最近邻取帧，而非固定隔一帧取一帧。
CSV偏移统一定义为`sequence_time=asset_time+offset`；取视频时用公共时间减视频偏移。
当前超出视频覆盖范围会报错，不会自动截短动作、补帧或时间扭曲。

HoloMotion以50Hz控制，每次控制10个物理子步、`action_clip=10`。
保存的第i个执行状态在`(i+1)/50`秒；先按显式时间戳插值回30Hz，旋转用SLERP，
再排除t=0初始化帧作为标签。`execution.mp4`按原生50Hz回放，不能当成30Hz人物RGB。
参考与执行轨迹分别保存；训练动作取执行轨迹，不用GMR参考直接冒充执行结果。

当前导出episode最长240帧、步长120帧，短尾窗口对齐末帧；同源机位及片段进入同一split。
`observation.state`为36维状态，`action`为下一帧状态；末帧重复自身并标记`action_valid=false`。
125维模型特征仍由现有`G1MotionFeatureCodec`生成，不另写一套转换。
这些是本地明确采用的契约，不表示官方所有AIST episode都使用相同切片政策。

## 当前质检与失败分类

`results.csv`中`complete`表示动作与RGB配对完成，`rejected`表示动作质检拒绝，
`failed`表示运行或配对失败。已有有效执行轨迹的序列均保留执行视频，包括拒绝项。

跌倒检查在50Hz执行轨迹上进行。以下任一条件连续达到10帧（0.2秒），拒绝整条序列：

| 判据 | 阈值 |
|---|---|
| 根节点离地高度 | <0.20 m |
| 根节点相对竖直方向倾角 | >85° |
| 同时低矮且倾斜 | 高度<0.35 m且倾角>60° |

当前`strict_joint_limits: false`：关节越界只记录；跟踪误差默认只记录，未配置拒绝阈值。
可用`quality.max_joint_rmse_rad`、`quality.max_global_body_mpjpe_mm`启用本地误差门槛。
跌倒阈值来自`stages/motion_quality.py`中`MotionScreeningConfig`的默认值，当前并未将这些字段接到YAML配置。
非有限值、无效四元数、时间轴不一致等仍会报错；RGB首尾缺失、超过0.5秒的跟踪空缺、
标签超出视频PTS覆盖范围也会失败，不会偷偷换主体或无限外推。

### 肘腕组合筛查（2026-10-19新增）

实现为`stages/arm_twist.py`，仅保留扩大档规则（版本`g1_same_arm_twist_expanded_v2`）。
在重定向完成后、控制器执行之前检查参考动作，同一手臂连续至少0.3秒满足以下条件：

- 肩3轴、肘、腕3轴中，至少2个关节距离任一URDF限位≤5°；
- 同时肘角≤−40°，或abs(腕pitch)、abs(腕yaw)均≥45°。

两个分支分别计时，不拼接短异常；30Hz至少9帧。越界也计为限位饱和。
不再包含“肘角≤0°且双腕轴≥45°”的独立严格档。角度均按G1关节轴定义，
负肘角、肩角大或两个关节贴限位本身都不单独拒绝；规则不保证零误报或零漏报。

默认`quality.arm_twist: {enabled: true}`。命中立即标记整条序列rejected，
跳过tracker、执行回放、RGB裁剪及导出，保留参考动作和`reference/quality.json`中的事件证据。
已缓存的参考也每次重筛；已有执行缓存不能绕过前置筛查。没有实际执行的拒绝项不生成
“执行视频”，旧执行缓存和旧视频也不主动删除。通过后继续检查执行阶段的跌倒等质量。
原来的`action`、`tier`、`scope`配置已移除；可用`enabled: false`关闭前置筛查，
或覆盖`near_limit_deg`、`elbow_reverse_deg`、`wrist_bend_deg`、`duration_seconds`。

此前114条候选及其对比视频来自旧版两档并集，属于历史结果，不自动按新规则改写。
旧结果中的strict/review标签不代表新规则仍保留两档。修改筛查代码不会直接删改现有训练数据。

参考/执行并排可视化（只读缓存，不重跑仿真）：

```bash
PYTHONPATH=.:src MUJOCO_GL=egl .venv/bin/python -m data_pipeline.visualization.arm_twist_review \
  --sequences-root data/processed/aistpp_full_prototype/sequences/aistpp \
  --sequence gBR_sBM_cAll_d04_mBR0_ch06 \
  --output data/processed/arm_twist_review
```

可重复传`--sequence`。每条输出MP4、命中时刻JPG和指标JSON；左列参考，右列最终
30Hz执行标签，上排全身，下排另一视角上半身近景。15fps抽帧播放，保持原速；跳过
没有真实执行结果的初始化首帧。仅做正向几何回放，不裁剪参考关节角，不推进物理状态。
参考命中区间标红，关节角按G1轴定义；执行侧没有红字不等于动作已经正常。

批量渲染全部参考候选，将上面的`--sequence`替换为
`--report <筛查结果目录>/reference_candidates.json`，可加`--workers 4 --skip-existing`。
断点续跑会核对已有视频的尺寸、帧率、帧数以及配套JPG/JSON；不覆盖不完整的正式视频。
有其他训练占用GPU时，可通过`MUJOCO_EGL_DEVICE_ID`指定空闲渲染设备。

不能将“未跌倒”直接当成高质量视频动作监督：当前未自动验收语义同步、脚滑、
全部跟踪误差或手脚支撑关系；也没有把一条失败动作的正常区间自动切出来保留。

## 已知执行问题与下一步

目前首帧姿态配零速度初始化，尚未保持原动作的完整动态状态。对手撑地、旋转、
腾空开头的动作，参考姿态相似并不保证MuJoCo中的接触和动力学可执行。

2026-10-15对`gBR_sBM_cAll_d04_mBR0_ch09`的检查中，原视频首帧有手部支撑，
重定向后的初始碰撞几何最低点却离地约16 cm，根倾角约57.5°，随后迅速倒地。
另一个样本初始离地约22 cm。还观察到部分初始姿态存在自碰撞重叠。
少量诊断仿真中，补速度/改初始高度能缓解部分早期失稳，但不能稳定修复全部样本；
关闭自碰撞也未普遍解决。不能据此认定所有倒地均由某一个因素造成。
这些诊断未改正式数据、控制标准或质检阈值。

后续优先核查模型/坐标与几何一致性、手脚支撑接触、与参考相容的初始速度，
再判断tracker对该动作的可执行性。不要为提高通过率直接调松跌倒规则，
也不要用逐帧贴地、慢放或事后时间扭曲掩盖失败。

## 保留的小样本实验结论

以下是早期实验，不是当前全量任务统计。旧调试脚本及部分产物目录已清理，
不再提供失效的复跑命令，也不能把小样本参数直接应用到全库。

### 同条件官方标签重放与高度消融

样本为`gBR_sBM_cAll_d04_mBR0_ch01`，12秒；同一tracker、50Hz、10子步，
各自首帧配零速度初始化，按post-step时间对齐，无DTW或事后空间配准。
v3只比v2降低整段root-z 6 cm；这是单样本消融，不是批量修正值。

| 指标 | 官方标签重放 | 本地v2 | 本地v3 |
|---|---:|---:|---:|
| 跌倒帧 | 0 | 0 | 0 |
| 严格限位标记帧（1e-4 rad容差） | 29 | 11 | 4 |
| 最大越界角度 | 0.523° | 0.994° | 1.095° |
| 关节跟踪RMSE | 0.1493 rad | 0.1464 rad | 0.1472 rad |
| 全局身体MPJPE | 220.5 mm | 244.7 mm | 187.1 mm |
| 根相对身体MPJPE | 51.8 mm | 58.9 mm | 52.3 mm |
| 平均根高度误差（执行减参考） | +0.62 cm | −6.77 cm | −1.07 cm |

官方重放也触发严格限位标记并超过曾试过的150 mm误差门槛，因此旧报告的严格失败
不能证明本地重定向明显差于官方；150 mm并非当前默认门槛，也不是官方标准。
平滑只将该样本全局误差从245.9降到244.7 mm，不能依靠加大平滑解决主要问题。
两侧均观察到约+0.21 m平均根X误差；tracker未直接闭环世界XY位置是一种可能解释，非普遍结论。

## 参考资料

- [AIST loader与视角顺序](https://github.com/google/aistplusplus_api/blob/main/aist_plusplus/loader.py)
- [AIST SMPL拟合实现](https://github.com/google/aistplusplus_api/blob/main/processing/run_estimate_smpl.py)
- [相关SMPL实现](https://github.com/liruilong940607/smplx/blob/master/smplx/body_models.py)
- [OMG论文附录](https://arxiv.org/html/2606.10340v1#A2)、[官方数据卡](https://huggingface.co/datasets/THU-MARS/OMG-Data/blob/main/README.md)

参考文献不替代本地实现约定；地面修正、短尾窗口、误差门槛等本地策略不能标成官方处理代码。
