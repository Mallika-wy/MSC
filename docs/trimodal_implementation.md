# MSC 三模态重写交付与实测记录

基线：`96dfba3`，规格基线为 `df5bda7`。二者之间的提交只改变流水线及相关依赖、文档；本次保留这些变化。架构重写阶段未修改流水线逻辑、配置、六表 Schema、已有 Release 或原始媒体。后续模块内化仅同步更新流水线两处运动学导入路径，处理逻辑及数据不变。

## M1：Release 与窗口契约

文件：新增 `src/msc/data/release_dataset.py`、`temporal.py`；修改 `data/collator.py`、`factory.py`、`config.py`，供既有训练框架使用。相关测试集中于 `tests/test_trimodal.py`。

- `k` 为第一帧 future；从 `row_start+k-5` 读取恰好 35 帧。通过临时 Record 视图复用现有 `read_motion`，只读取相交 Row Groups，支持跨组窗口，不读取完整 Motion。FK 与 MSC Codec 使用同一历史末帧 Anchor。
- 发布时从 provenance 读取正式 Label 并构建 training/ 索引；父子 Motion、Actor、Sequence、Split 和歌曲身份在发布/validate 阶段检查。多库 index 命令执行一次跨库审计。
- 索引确定窗口及对应 Annotation/Link/Feature；任务池不接收缺失或无效条件的窗口。默认要求完整 future 覆盖，音视频上下文为 0.25 秒。`__getitem__` 不抽随机数。
- 特征、motion 窗口缓存有逐 Worker 字节上限。Worker 使用 spawn，避免继承 Arrow/线程状态。文本由本地冻结 T5 在线编码。
- training/summary.json 记录六表和 manifest 的 SHA256、索引文件 SHA256、Label、对齐策略和上下文长度；这些文件在发布/validate 阶段核验。训练身份由已发布索引指纹及所选 split/任务/子集配置生成，启动时不重算文件 hash、不读取完整 Inventory、不重新扫描窗口。修改 Release 后须重新运行 index/validate。

真实 AIST++ Release 读取结果：1052 条 Sequence，正式标签为 `tracker_executed`，可用 Audio 窗口 **317,536**，条件覆盖不足排除 **995** 个窗口。读取样本为 history `[5,125]`、future `[30,125]`、MERT `[La,768]`，保留真实 PTS。当前 1052 条 Sequence 全部为 Train；未擅自生成 Val/Test。

数值验证包括 12 帧 Row Group fixture 上跨组读取、35 帧边界、Codec parity、qpos36 解码、时间 scale/offset、origin 不重复补偿、缺失条件、features-only 包、缓存上限和 split 泄漏。

## M2：单一 Memory 与共享模型

文件：重写 `models/motion_generator.py`、`models/msc_model.py`；修改 `models/flow_matching.py`、`models/conditioning/{text,audio}.py`；新增 `models/conditioning/{fusion,video}.py`。迁移 `data/synthetic.py` 和 `tests/test_conditioning.py`、`test_flow.py`、`test_rtc.py`，保留回归断言的目的。

- ConditionMemory 是唯一的 Attention 条件接口，包含 tokens、mask、modality IDs、relative times、temporal support。采样和 RTC 都从 Memory 取得 batch/device，不保留另一份 context 字典。
- 每个样本保留 5 个 History Tokens；CFG null 从已编码 Memory 移除外部模态。每次采样只编码一次条件，逐 Euler step 复用。
- 有效 token/timestamp 的 NaN/Inf 报错，无效部分在投影和时间编码前清零。Video 每个空间 token 继承时间区间；K/D 必须来自已验证元数据。
- 默认 6 blocks、512 hidden、8 heads、2048 FFN；每层 self-attention → cross-attention → FFN，逐层 flow time 调制。默认逐头 softplus 距离偏置，支持 `none` 消融。
- 保持原 FM 方向、标量/逐帧时间、CFG 与 RTC；保留兼容的旧输入参数名称，但内部只使用统一 Memory。

参数量：Text+Audio 默认共享模型 **36,980,135**；本次 Audio-only 实验 **36,322,215**。均不包含离线 Encoder 权重；真实 Video Adapter 参数量等待 checkpoint 输出布局确认。

## M3：辅助监督、训练、评估

文件：新增 `models/losses.py`、`data/stats.py`、`training/experiments.py`；修改 `training/{trainer,checkpoint}.py`、`train.py`、`sample.py`、`evaluate.py`。新增两份 `configs/trimodal_*.yaml`，扩展 `tests/test_train_step.py`、`tests/test_integration.py` 及 trimodal 测试。

- `x0_hat=x_tau-tau*v` 不 detach；FP32 反标准化后取 29D joint angles，用最后两帧历史计算速度/加速度边界。相邻有效性掩码屏蔽 padding/缺口。逐关节 Train 标准差下限为 `1e-4`，归一化残差用 Smooth L1。
- 默认辅助权重 0.05/0.01，默认前 1000 个 optimizer steps 关闭；短实验前 100 步关闭。`tau=0` 梯度为零，非零 tau 可独立反传。
- task sampler 的选择仅依赖 seed/绝对 microstep；flow noise/tau 使用独立、逐 microstep 的 RNG。默认开启 PyTorch deterministic algorithms；checkpoint 保存模型/数据/特征/统计/损失/采样身份、optimizer/scheduler/AMP/RNG，拒绝旧架构和不匹配恢复。
- 验证使用独立种子且恢复训练 RNG；训练记录各 task 的损失、有效条件比例和梯度。评估记录动作幅度、动态残差、条件 controls、FK、Jerk、边界、连续 rollout 与 SONIC。

### 新训练入口

此前模型训练的 checkpoint、采样、评估和执行结果已按要求清理。当前只支持 checkpoint format version 3 与 `g1_canonical_125d`，不迁移旧 checkpoint。模型需要重新训练后再采样、评估。

```bash
PYTHONPATH=src .venv/bin/python -m msc.training.experiments \
  --config configs/trimodal_audio_overfit.yaml \
  --output .runs/trimodal-audio-short \
  --sonic-config configs/data_pipeline/aistpp_motion.yaml
```

该命令在 GPU 上执行 200 步真实 Audio overfit。现有 Release 没有独立 Val/Test，因此三组 1000 步损失对照仍为 Pending；不会将 Train 结果充当独立验证结果。

## M4 与未完成项

- 正式 Text Release 使用本地冻结 T5 在线编码，不生成文本特征缓存。
- **Pending：真实 V-JEPA features、K/D/时段/Actor 验证与 Video 训练。** 已完成合成 Video forward/backward/mask/梯度测试，没有假设真实 V-JEPA 布局或宣称真实视频训练完成。
- **Pending：独立 Val/Test 与三组损失对照、可靠条件匹配和稳定 SONIC 跟踪。** 不重新划分现有 Release，不把静止/低幅度当成改善。
- 当前只保留原生训练路径；旧 checkpoint 与训练产物已清理，旧 checkpoint 加载会明确报错。

## 验证命令

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests -q -m 'not slow and not integration'
PYTHONPATH=src .venv/bin/python -m pytest tests/test_overfit.py -q
PYTHONPATH=src MSC_TEST_T5_PATH=/path/to/local-t5 \
  .venv/bin/python -m pytest tests/test_integration.py \
  -k 'frozen_t5_and_manifest_cache or t5_unfreeze_gradients' -q
PYTHONPATH=src .venv/bin/python -m compileall -q src/msc
git diff --check
```

核心测试覆盖真实布局的合成 Release、三种 Adapter 梯度、masked NaN、CFG Memory 复用、时间偏置、动态边界/短序列/零方差/tau 极值、BF16、任务采样、Worker 与梯度累积的精确恢复、错误指纹/旧 checkpoint 拒绝、验证 RNG 隔离和连续 rollout。最终核心测试：**55 passed，3 deselected**（19.19 秒）；600 步 synthetic overfit：**1 passed**；本地 T5 的冻结/缓存与旧 Encoder 接口回归：**2 passed**（9.57 秒）。compileall 与 git diff --check 通过。

额外仓库维护：将 `.gitignore` 的 `data/` 收紧为 `/data/`，使新增的 `src/msc/data/` 文件可以正常被版本管理；test extra 增加 Arrow 依赖。后续模块内化仅同步更新流水线的两处运动学导入，不改变处理逻辑。


## MSC 原生模块组织

运动学、旋转转换和动作编解码现位于 `src/msc/motion/{kinematics,rotations,codec}.py`，注意力组件位于 `src/msc/models/attention.py`。旧的专用数据适配器、窗口辅助代码及旧后端配置已从当前运行时代码移除；历史版本仍可通过 Git 基线查看。表示名称统一为 `g1_canonical_125d`。

当前 checkpoint format version 3 只接受 `g1_canonical_125d`，不保留旧名称映射或 checkpoint 迁移逻辑。旧训练结果已清理；来源版权与许可证声明保留于 THIRD_PARTY.md 和 licenses。

流水线 `motion/retarget.py` 与 `motion/tracking.py` 各只改变一处运动学 import，以调用新的原生模块路径。没有更改数据生成、筛选、训练特征或已有 Release。
