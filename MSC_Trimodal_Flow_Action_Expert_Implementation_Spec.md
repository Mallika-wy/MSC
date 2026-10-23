# MSC v1 — 冻结三模态编码器 + 共享 Flow Action Expert：模型改造实施规格

> **用途：直接交给 Codex 作为实施说明。**
> **基线：** `Mallika-wy/MSC`，`main` at `df5bda74d0c86ead2daa2de4fd27095e7d9450a8`（2026-10-21 核对）。实施前必须重新检查实际 HEAD，不能在不知道新差异的情况下覆盖后续提交。
> **论文参考：** 用户提供的 *Humanoid Motion Foundation Model for Multimodal Streaming Closed-loop Interaction*，主要参照第 3 节与附录 B.2–B.4。论文使用冻结 Qwen2.5-Omni-3B + Ψ₀-style 6-block Action Expert + 70D Action；**本文是针对当前 MSC 的独立工程方案**，不声称等价复现论文原模型或论文报告的实验结果。

## 0. 首要目标与边界（不得擅自更改）

本轮只实现 **Text-to-Motion、Audio-to-Motion、Video-to-Motion** 三类条件生成，并让它们使用同一个可训练动作模型。推理产出 G1 运动轨迹，调用现有 SONIC 做下游跟踪。此轮不做 steering、局部编辑、异步流式控制、反馈条件 `c_cor`、新的 Motion VAE、离散化 Action Tokenizer、自制 DAG 或大规模框架重写。

必须满足：

1. **三条独立冻结的 Encoder：** Text=T5、Audio=MERT（现有 MERT-v1-95M 特征）、Video=已经选定并待验证具体 checkpoint 的 V-JEPA 系列。所有 Encoder 均可离线预计算，训练时默认只读取特征，且不对其反向传播；仅其后面的 Modality Adapters、Condition Bridge、Flow Action Expert 可训练。
2. **保留 OMG 的 125D Canonical G1 表示**，`history=5` 帧、`future=30` 帧、30 FPS；Flow Head 的输出始终是 `[B,30,125]` 速度场。**不引入 70D 主输出、70D Head 或 Foot Contact Loss**。
3. 增加从 125D 关节角派生的 **Joint Velocity** 与 **Joint Acceleration** 一致性辅助损失，试图减少用户在先前 OMG Flow 实验中观察到的抖动；**不保证仅凭辅助损失就能消除抖动**，必须测试真实 rollout 与 SONIC。
4. **共享唯一 Action Expert**，同一模型处理三种单模态；能处理缺失模态。联合 Text+Audio/Video 属于架构可支持但须有真实配对数据才可验证的扩展，不得把独立任务的随机条件当成真实联合监督。
5. **不修改正在运行的 `data_pipeline/`、其配置、MSC v1 六表 Schema、已有 Release 文件及原始数据**。Model Dataset 只读 MSC Release。缺失 V-JEPA 特征时用小型合成 Fixture 测接口，不宣称视频分支已完成真实训练。
6. 尽量复用 `src/msc/` 当前 Flow、OMG Codec、G1 Kinematics、Trainer/Checkpoint；不引入另一个并行 Model Framework。保留现有 Synthetic 基线和测试，必要时为旧模型单独存档/标记版本，**不能把旧 Checkpoint 静默当作新结构加载**。

## 1. 目标架构（唯一训练路径）

```text
[Raw Text] ── Frozen T5 ──────────────────────────────┐
[Raw Audio] ─ Frozen MERT ── timestamps ──────────────┼─> 只读特征缓存 / MSC Release
[Raw Video] ─ Frozen V-JEPA ── time intervals + K ────┘
                                      |
                        independent trainable adapters
                    text P_t  /  audio P_a  /  video P_v
                                      |
                    modality id + temporal positions + masks
                                      |
                   Structured Condition Memory [B,Lc,512]
                                      ^
                       history 5×125 -> history adapter
                                      |
          noisy future x_tau [B,30,125] + flow time tau
                                      |
                  SHARED FLOW ACTION EXPERT  (6 blocks)
            future self-attn -> condition cross-attn -> FFN
                         flow timestep modulation
                                      |
                   velocity prediction [B,30,125]
                                      |
       FM loss + x0-hat joint velocity loss + acceleration loss
                                      |
                   sampling -> canonical OMG125D
                                      |
                existing OMG Codec -> G1 qpos36 -> SONIC
```

Encoder 输出不能简单求均值或直接将 T5、MERT、V-JEPA 维度拼接。**三个独立投影分别学习跨模态到动作特征空间的映射**；融合发生于共享 Action Expert 对 Condition Memory 的 Cross-Attention，而不是一个过大的可训练多模态 Backbone。

### 1.1 建议首版模型规模

| 参数 | v1 默认值 | 说明 |
|---|---|---|
| Motion representation | OMG G1 125D | `root_pos(3), root_rot6d(6), joint_angles(29), link_xyz(87)` |
| History / Future / FPS | 5 / 30 / 30 | 沿用当前已实现的窗口规范 |
| Expert blocks | 6 | 与论文的 6 层数相近，但网络宽度不同 |
| Hidden / Heads / FFN | 512 / 8 / 2048 | 以小 GPU 可训练为目标，后续实测参数与显存 |
| Motion self-attention | 每层 1 次 | 30 个未来 Action Tokens，双向注意力 |
| Condition cross-attention | 每层 1 次 | 共享读取 Text/Audio/Video/History Memory |
| Flow condition | 正弦/MLP timestep，注入各层 | 可复用现有 TimestepEmbedding；额外调制可用简单 AdaLN/FiLM |
| Encoder trainability | 全部 frozen | Encoder 本体不参与 optimizer，不加载三种大型 Encoder 到每个 DataLoader worker |
| Trainable | 3 adapters + history adapter + expert | 训练同一个动作分布模型 |
| Flow steps | 默认 10，支持配置 | 保留现有 Euler from noise `tau=1` to data `tau=0` |
| RTC / streaming | disabled | 保留现有代码如可无冲突，但本轮不扩展 |

首版不实现复杂的 Perceiver、Q-Former、MoE、三套 Decoder、分身体部位 Expert、非必要的参数共享机制。

## 2. 数据与张量契约（先实现并测试）

### 2.1 单个 Batch

```python
batch = {
    "motion_history": FloatTensor[B, 5, 125],
    "motion_future":  FloatTensor[B, 30, 125],
    "motion_mask":    BoolTensor[B, 30],
    "text_tokens":    FloatTensor[B, Lt, Dt],    # frozen T5; Dt 从真实模型元数据读取
    "text_mask":      BoolTensor[B, Lt],
    "audio_tokens":   FloatTensor[B, La, 768],   # frozen MERT，未强制重采样到 30 Hz
    "audio_times":    FloatTensor[B, La],        # 相对于 motion future window 起点的秒数
    "audio_mask":     BoolTensor[B, La],
    "video_tokens":   FloatTensor[B, Lv, K, Dv], # frozen V-JEPA; Dv/K 由真实输出确定
    "video_intervals": FloatTensor[B, Lv, 2],  # 相对于 future window 起点 [start,end) 秒
    "video_mask":     BoolTensor[B, Lv, K],
    "task":           ...,                      # text/audio/video，用于采样/审计，不作为必须的模型输入
}
```

允许每个 Batch 只存在一部分 `text/audio/video` 字段；Collator 将有效的变长序列 Pad，并严格给出 Mask。**没有该模态**与**有效但数值为零的模态**必须区分。时间单位统一为秒；Tensor dtype 可在进入网络时转换，但应保存原始特征指纹和 token timestamp 语义。`Dt`、`Dv` 不能依据网上示例盲猜。

### 2.2 MSC Release Dataset（不可依赖旧官方 OMG Adapter）

新增 `src/msc/data/release_dataset.py`，只读 `data_pipeline` 已发布的 MSC v1：`SequenceRecord`、`MotionRecord`、`AssetRecord`、`ModalityLink`、`TextAnnotation`、`FeatureRecord`。复用 `data_pipeline/release/export.py::read_motion`、`data_pipeline/release/features.py::read_feature`、`data_pipeline/schema/time_map.py::TimeMap` 读取工具，**不要复制一份协议实现**。

- 新数据后端 `msc_release`，支持一个或多个 Release Roots、`train/val/test`、明确的标签选择与模态任务选择；**默认对每个源 Motion 只选择一个正式 Label**（`train_default`，若无法解析需报错或要求显式选择，不要暗中同时使用 reference/executed）。
- 从单个 MotionRecord 的 30 Hz qpos36 分片中读取窗口，不跨 Motion、Actor、Episode 或 split；训练先只取**连续完整 35 帧**。严格定义 `k` 为第一帧 future 的 motion-local index：`history=qpos[k-5:k]`，`future=qpos[k:k+30]`；合法起点 `5 <= k <= T-30`。短序列首版跳过并统计，不从相邻 episode 借帧。
- 同一窗口必须使用**同一历史末帧**建立 OMG Canonical Anchor；用现有 `msc._omg.G1MotionFeatureCodec`、`G1Kinematics` / FK 得到 history `[5,125]` 和 future `[30,125]`，并写与旧 OMG 数值实现的回归测试。**不能把 36D 直接线性映射到 125D，也不能对 history 和 future 分开建 Anchor。**
- 发布 `MotionRecord.time_origin_sec` 的影响已被 ReleaseBuilder 用于 `ModalityLink.time_offset_sec` 的变换；DataLoader **不重复添加 1/30 秒**。视频裁剪后的时间坐标与源视频 PTS 的映射必须保留/核对。
- 对选中 Text Annotation 按本次窗口的 motion-local 时段判断相关性；本轮可选窗口覆盖度最高的有效标注或明确配置抽样，不能随机绑定无关文本；多条冲突标注不强制拼成一条 Caption。T5 特征允许在训练时按内容哈希离线去重缓存，避免 75 万个逐 episode 小文件。
- 对选中 Audio/Video Link 先检查 `alignment_status`、`motion_valid_start_sec` / `motion_valid_end_sec`，再通过 `asset_time = scale * motion_time + offset` 将 query 范围变换到资产时间；按缓存特征真实 PTS/Token 时间选取窗口及少量上下文，变换回 motion-local window time；生成有效掩码。估计对齐允许与否由 Dataset 配置决定，不可凭两个文件长度强制同步。
- 推荐 Audio 条件窗口覆盖未来约 1 秒 + 可配置边界上下文（例如 0.25 秒），MERT 大约 75Hz；仅取相关 Tokens，不能每个窗口喂整首歌给 Cross-Attention，也不能用 `index * 2.5` 猜时间。时间选择与 `point/interval` 的语义必须匹配 FeatureRecord。
- 视频 token 应允许 `[Lv,K,Dv]`，保留时间区间与 mask。推荐**先验证**每个时间位置池化为约 `K=4` 个空间 Tokens 的实现，不要把 V-JEPA 原始高密度 Patch 网格直接喂 Action Expert，不要未经实际模型测试就承诺固定 Video FPS 或 Dv。
- 训练不依赖原始 WAV/MP4；`assets.uri=None` 的 features-only 包仍可训练。按 Motion Shard / Feature ID 加有内存上限的 Worker-local 缓存，避免每个 30 帧窗口完整解压同一首歌；不得一次驻留所有 OMG Motion 或全部特征。
- 当前视频 Pipeline 尚无 V-JEPA 特征：`video` 训练任务若被启用但缺乏 FeatureRecord，**显式报错/禁用该任务并报告**，不得以零张量假装有真实 Video 条件。测试可以使用 synthetic Video tokens。

### 2.3 ConditionBridge 输出契约

```python
ConditionMemory(
    tokens=FloatTensor[B, Lc, 512],
    mask=BoolTensor[B, Lc],
    relative_times=FloatTensor[B, Lc],     # text 可忽略时间，history 使用负时间
    temporal_support=FloatTensor[B, Lc, 2] | None,
    modality_ids=LongTensor[B, Lc],       # history/text/audio/video
)
```

这只是**接口草图**，可用一个轻量 dataclass 或普通字典实现，不要开发 Registry / Plugin 系统。最好保证每个样本始终有 5 个有效 History Tokens，从而避免所有 Cross-Attention Keys 都被 mask 掉造成 NaN。History **默认不 dropout**；若未来要测试 History Dropout，需保留明确的 null token 策略。

## 3. 冻结 Encoder 与可训练 Condition Bridge

### 3.1 Frozen Encoder

- **Text**：复用 `src/msc/models/conditioning/text.py::T5TextEncoder`；实际维度从 `encoder.output_dim` 获取，默认 frozen；离线缓存应保存 attention mask、checkpoint hash、tokenizer / preprocess 参数。若使用预计算 Token，训练不加载 T5 权重。
- **Audio**：直接消费 `data_pipeline/audio/mert.py` 发布的 MERT 最后一层 Token `[Ta,768]`、token centre PTS；不在 `src/msc` 里重新跑 MERT、不退回 OMG 35D、不改既有 Audio Feature 数据。checkpoint identity 包含 encoder/preprocess fingerprints。
- **Video**：未来独立执行冻结 V-JEPA 视频提取。该 checkpoint 的实际 Token 布局、时间感受野、patch/grid shape 必须先用最小视频检查确认；缓存 Feature 时保留时间区间、人物裁剪信息及帧有效 mask。**当前 Data Pipeline 只发布 Crop/Video Asset，真实 Video Encode 是必要的另一个前置步骤，但不在这次模型代码 PR 内擅自改 Pipeline。**

### 3.2 Trainable Adapters

三个独立参数的投影均采用 `LayerNorm(input_dim) -> Linear(input_dim, 512) -> SiLU -> Linear(512,512)`；视频展平 K 个空间 Tokens 后使用相同思想。Text/MERT/V-JEPA 都加 **独立 modality type embedding**；Video 还需保留 K 个位置/区域的空间编码（非原视频画面坐标时不可把 pooled grid 误当关节坐标）。History 用 `LayerNorm(125) -> MLP(125,512)`，并加入顺序/相对时间信息。

首版**不训练 Encoder，不引入额外共享特征 Encoder 或多模态生成预训练任务**。`ConditionBridge` 只负责 tokens、masks、modality IDs、temporal positions，不从标签读取未来运动信息。

## 4. 共享 Flow Action Expert

每个输入 Batch 构建一次 Condition Memory，并在全部 Flow Sampling Steps **重复使用同一份 Memory**，禁止每一步重新加载 T5、MERT 或 V-JEPA。

Motion Query：

```text
noisy_motion[B,30,125]
-> Linear(125,512) + future positional encoding
-> 6 x { pre-norm self-attn of 30 future steps
       + pre-norm cross-attn to ConditionMemory
       + timestep-conditioned FFN / AdaLN }
-> final LayerNorm -> Linear(512,125)
-> flow velocity[B,30,125]
```

- Self-Attention 在 30 帧 Future Motion 内进行，**并非自回归**，使用现有 `motion_mask`；已有 RoPE Self-Attention 可复用。
- Cross-Attention 统一读取所有可用模态和 History；不再需要单独的 Audio FiLM 主路径。可保留原 Audio FiLM 实现作为旧实验 baseline，但不要让两套融合逻辑同时成为默认训练代码。
- 每层流时间 `tau` 作为 modulation/embedding 注入；数值方向保留当前 `FlowMatching`：`tau=1` 为高斯噪声，`tau=0` 为真实数据，模型预测 `eps-x0`，Euler Sampling 方向从 1 到 0。避免为了改 Transformer 同时翻转 Flow 符号。
- v1 **默认启用简单可调的时间偏置**：对 future query time `t_i=i/30` 和音视频 Token 时间 `t_j`，在每个 attention head 的 logits 加 `-softplus(alpha_h)*distance(t_i, token_support_j)`；point token 使用绝对时间差，interval token 在区间内距离取 0，区间外取到边界距离。文本时间偏置为 0；History 使用显式历史相对时间或单独类型位置编码。无效 Key 统一加 mask（逻辑上 `-inf`）。参数必须可训练，但初始化偏置不应强到使全部 Audio/Video Context 只能关注最近一帧。
- 为明确消融，配置支持 `temporal_bias: none | relative_distance`，默认 `relative_distance`；若实现 Flash/SDPA，必须保证 mask、偏置语义正确。不要为了加速牺牲可验证的正确性。
- 只用 Cross-Attention 不自动意味着 Text/Audio/Video 特征语义已对齐：对各模态的**独立真实任务监督**才让共享 Expert 学会使用条件。
- **旧代码接口迁移注意：** 当前 `FlowMatching.sample()` 使用 `conditions["context"]` 获取 Batch 大小与 Device。新的 Memory 可以继续提供 `context/context_mask` 兼容字段，也可以最小幅度修改 Flow 从 `ConditionMemory.tokens` 获取 shape；必须同步检查 `.sample()`、RTC 分支、CFG null、单元测试，避免仅改 `MotionGenerator.forward()` 而 Sampling 崩溃。
- **视频任务边界：** 本轮的 Video-to-Motion 是**离线示范视频条件生成**，允许读取与目标动作同步的示范区间；若未来用于真实在线机器人视觉观测，则绝不可以读取尚未发生的未来画面。当前先统一接口并在元数据中区分任务，不将离线结果宣称为实时闭环能力。

### 4.1 Null Conditions / CFG

- `drop_modalities=True` **只移除 Text/Audio/Video，不移除历史 Motion**。CFG 的 null 分支相当于“给定历史状态的 Motion Continuation”，不是无历史的生成。
- 为避免全空 Attention 导致 NaN，始终留有效 History Tokens；缺失模态不可伪装成有效零特征。
- 条件 dropout 建议每个可用外部模态 `p=0.1`，History 默认 `0`。不要对 DataLoader 的已经缺失模态再次产生所谓“真实零条件”。
- 相同 `initial_noise`、相同历史下测试有无条件的输出/梯度敏感性，但输出不同本身不等于语义匹配正确。

## 5. 动作表示和训练损失：125D + Dynamics（无 Contact）

使用目前 `MotionRepresentation.slices`：

```text
root_position: [0:3]
root_rot6d:    [3:9]
joint_angles:  [9:38]     # 29D
link_xyz:      [38:125]   # 87D
```

固定唯一生成目标：`x0 = normalize(ground_truth_125d)`，`x_tau=(1-tau)*x0 + tau*eps`，`target=eps-x0`，`pred_velocity=v_theta(x_tau,tau,C)`。

**主损失：**沿用已实现的 masked Flow MSE，不能改动标量时间与 `[B,30]` 时间两个输入模式；第一组实验完全保留原 Flow MSE 权重（Group-Balanced 125D FM Loss 只作为以后明确开启的消融，不加入首版默认配置）。

**Clean motion estimate：** `x0_hat = x_tau - tau*pred_velocity`；对其调用 `representation.denormalize()`，从 `[9:38]` 取 Joint Angles。与 Ground Truth 使用**同一 Canonical Anchor**。

Joint Velocity / Acceleration 必须包括 Future 第 0 帧与**最后两帧真实 History Joint** 的边界，不能把 Future 第 0 帧错误地当作静止状态。FPS=30：

```python
# pred_j, gt_j: [B,30,29] (physical joint angles, radians)
# h_j: [B,5,29] (same canonical contract)
pred_prev = cat([h_j[:, -1:], pred_j[:, :-1]], dim=1)
gt_prev   = cat([h_j[:, -1:], gt_j[:, :-1]], dim=1)
vel_pred  = 30.0 * (pred_j - pred_prev)
vel_gt    = 30.0 * (gt_j - gt_prev)

# Same boundary convention for acceleration, with h_j[-2] for frame 0:
pred_prev2 = cat([h_j[:, -2:-1], h_j[:, -1:], pred_j[:, :-2]], dim=1)
gt_prev2   = cat([h_j[:, -2:-1], h_j[:, -1:], gt_j[:, :-2]], dim=1)
acc_pred   = 30.0**2 * (pred_j - 2*pred_prev + pred_prev2)
acc_gt     = 30.0**2 * (gt_j - 2*gt_prev + gt_prev2)
```

**实现注意：上面 `prev2` 示意中使用拼接片段的写法要保证恰好 30 帧，并在 `T<2` / Mask 的边界单独测试。** 训练时还需要对动态误差按照**仅 Train Split 统计**得到的逐关节或共同尺度进行归一化，否则 `30²` 产生极大的梯度。可采用归一化速度/加速度残差的 Smooth L1/Huber，或显式记录尺度后以 L1 监督；禁止直接把未归一化的 `rad/s²` 误差乘 `0.01` 加入 FM Loss。对 padded future 的邻接计算必须仅在相邻有效的帧上进行（首帧用有效 History），避免 Padding 影响速度/加速度。

> **首版建议初始搜索参数**：`lambda_vel = 0.05`，`lambda_acc = 0.01`，仅在动态目标按训练数据尺度归一化后尝试；不是经验上已经验证最佳。建议先在一个短 warm-up 阶段让 FM 稳定，再开启辅助项，记录各项梯度与损失。

最终目标：

```text
L_total = L_FM + lambda_vel * L_joint_velocity + lambda_acc * L_joint_acceleration
```

Root Dynamic Loss、Foot Contact Loss、FK Consistency Loss 默认全部关闭，**首版不实现**。可以记录根部运动误差、FK、足部滑动作为 Evaluation 指标，但不要把它们悄悄加入训练目标。不得添加末帧硬平滑或惩罚任意运动速度的 `||q_t-q_{t-1}||`，因为这会伤害快速跳舞动作。

### 5.1 与当前 FlowMatching API 的集成

当前 `msc_model.py::compute_loss()` 直接返回 `self.flow.loss(...)` 标量。为避免复制 Flow 插值与目标的实现，建议对 `FlowMatching` **小范围扩展**为可选返回中间结果（例如 `loss_and_prediction()` 或 `return_details=True`），以提供 `x_tau, tau, pred_velocity, mask`。保持现有 `.loss()` 对旧测试返回标量、`MSCModel.compute_loss(batch)` 默认返回标量的行为；另增 `compute_loss(..., return_metrics=True)` 或 `compute_loss_components` 供 Trainer 记录 `loss_flow/loss_vel/loss_acc/loss_total`。保持原有 AMP、梯度裁剪、checkpoint resume；额外计算导数项时可在 FP32 做差分以防精度问题。

## 6. 多模态训练策略

第一版按**任务**而不是按全体样本数量随机采样。同一个 Expert 与参数用于三个任务。数据来自各 Release 内真实的 Motion+Text、Motion+Audio、Motion+Video 配对；禁止给独立 Motion 随机配不相关的条件作为“联合训练”。

- 仅 Text 与 Audio 真实可用时：`task_weights={text:0.5,audio:0.5}`，Video 只有 synthetic 测试，不参与正式优化。配比可以配置，与数据占比无关。
- Video Features 可用且验证后：初始 `task_weights={text:1/3,audio:1/3,video:1/3}`，对应论文附录 B.3 的混合任务思路。需要监测三个任务各自的验证损失，不只报告总 Loss。
- 各模态都用同一 125D 目标和同一 Loss。`motion_only` 作为 explicit baseline 或 CFG null training，不能成为默认大量样本来源从而掩盖条件学习。
- Trainer 目前 `StepBatchSampler` 用 `seed+step` 产生确定索引；请扩展为**确定性按 task 抽样 + task 内样本抽样**，仍可准确 resume，不能要求加载全部窗口/episode ID 到 GPU 或创建巨大 `randperm`。保存采样比例、任务数据列表身份和采样指纹。
- 各 task 允许有不同 Coverage Policy：Text 标注必须适配窗口，Audio/Video 至少符合配置的时间覆盖/有效 token 比例；不得静默用未对齐资料训练。
- 每个 task 记录 `loss/flow/vel/acc`、条件有效比例和梯度统计；有条件与打乱条件（same history、same initial noise）输出差异的消融。

## 7. 推荐配置（目标格式，必须同步修改 config parser）

```yaml
model:
  representation: omg_g1_125d
  fps: 30
  history_frames: 5
  future_frames: 30
  hidden_dim: 512
  num_layers: 6
  num_heads: 8
  ffn_dim: 2048
  dropout: 0.1
  rtc: false
  condition:
    fusion: token_cross_attention
    temporal_bias: relative_distance
    text: {enabled: true, backend: t5, frozen: true, dim: auto, dropout: 0.1}
    audio: {enabled: true, backend: mert_v1_95m, frozen: true, dim: 768, dropout: 0.1}
    video: {enabled: false, backend: vjepa, frozen: true, dim: auto, spatial_tokens: 4, dropout: 0.1}
    history_dropout: 0.0
  losses:
    flow: 1.0
    joint_velocity: 0.05
    joint_acceleration: 0.01
    dynamic_normalization: train_stats
    dynamic_warmup_steps: 1000
    foot_contact: 0.0
    root_dynamics: 0.0
    fk_consistency: 0.0

data:
  backend: msc_release
  roots: ["/path/to/omg/release", "/path/to/aistpp/release"]
  split: train
  label_policy: train_default
  alignment_policy: allow_estimated
  audio_context_sec: 0.25
  task_weights: {text: 0.5, audio: 0.5, video: 0.0}
  num_workers: 4
train:
  device: cuda:0
  precision: bf16
  batch_size: 8
  accumulation_steps: 2
  learning_rate: 0.0001
  warmup_steps: 1000
  schedule_steps: 100000
  steps: 100000
```

这是**目标配置示例**，不是当前 `src/msc/config.py` 已支持的结构；Codex 需要明确修改 `ModelConfig` / `DataConfig` / `factory.py`。具体 GPU Batch、Worker 数、T5 路径、V-JEPA Checkpoint 等以实际硬件和可用文件设置，不假设用户有 8 张 A100。

## 8. 与当前 GitHub 文件的一一对应修改

| 当前路径 | 应执行的改动 | 不应做的事情 |
|---|---|---|
| `src/msc/models/representation.py` | 保留 125D、标准化、`stats_sha256`；检查按最终 Label 的数据统计；必要时允许外置统计路径 | 改成 70D / 改 G1 表示 |
| `src/msc/models/motion_generator.py` | 6 层 512D，共享 Action Blocks：self-attn/cross-attn/flow-time conditioning、接受结构化 Memory/Mask/Temporal Bias | 为三个模态分别建 Transformer |
| `src/msc/models/flow_matching.py` | 保留 Flow 方向和采样；最小方式返回预测中间量用于附加 Loss，兼容旧标量 API | 改 Flow 符号/重复实现采样/本轮加强 RTC |
| `src/msc/models/msc_model.py` | `encode_conditions(batch)` + `forward()` + `compute_loss()` + `sample()`；一次条件编码，多次 Flow 调用共享 Memory | 将所有 Audio/Video 预处理和模型加载写进这个类 |
| `src/msc/models/conditioning/text.py` | 复用 frozen T5、可直接吃离线 Tokens、明确 fingerprint | 训练时默认更新 T5 参数 |
| `src/msc/models/conditioning/audio.py` | 替换固定 35D 逻辑，正式支持 768D MERT Adapter 与 Mask | 将 MERT 75 Hz 偷偷当成 30 FPS |
| `src/msc/models/conditioning/video.py` **NEW** | 接受 V-JEPA 预计算的 `[B,Lv,K,Dv]`、Mask、时段；轻量投影 | 首版重写视频提取算法/猜定 Dv |
| `src/msc/models/conditioning/fusion.py` **NEW** | `ConditionBridge`：独立 Projection、modality id、time positions、合并 Memory 与 Mask | 引入 Q-Former / Perceiver / Plugin registry |
| `src/msc/models/losses.py` **NEW** | 关节速度/加速度残差、有效帧与历史边界、归一化、metrics | 新增 Contact / 70D Head |
| `src/msc/data/release_dataset.py` **NEW** | MSC v1 sharded Motion + Features/Annotations/Links -> 125D 标准 batch | 依赖旧固定官方哈希 LeRobot 适配器 |
| `src/msc/data/temporal.py` **NEW** | 训练侧局部窗口时间查询与 feature token selection；**调用**共享 `TimeMap`，不重新定义时间映射 | 视频/音频用文件名或 Token 索引猜同步 |
| `src/msc/data/collator.py` | 对可变长 Token 和可缺失模态进行 pad+mask | 将缺失模态表示为全有效零向量 |
| `src/msc/data/synthetic.py` 和 `tests/conftest.py` | 从旧 `[30,35]` Synthetic Audio Fixture 迁移到 MERT `[La,768]` + 时间戳，并加入合成 Video `[Lv,K,Dv]`；必要时保留单独的 legacy fixture | 让旧测试因为维度不一致而静默失效 |
| `src/msc/config.py`, `src/msc/factory.py` | 增加目标配置和 `msc_release` backend，build_model 只构建需要的 Adapter | 无条件加载 T5/MERT/V-JEPA 模型 |
| `src/msc/training/trainer.py` | task-balanced deterministic sampler、AMP、每任务 loss、支持 Resume | 引入大型分布式训练框架 |
| `src/msc/training/checkpoint.py` | 记录 Release fingerprint、label policy、125D stats、各 frozen Encoder fingerprint、Adapter 配置、Loss 版本与任务采样 | 静默恢复不兼容的旧 35D / 12-layer checkpoint |
| `src/msc/train.py`, `src/msc/sample.py`, `src/msc/evaluate.py` | 升级新数据 backend、三模态输入 CLI/批处理、生成 125D，输出 qpos36 的推理路径 | 依赖原始 WAV/MP4 才能使用 features-only 包 |
| `src/msc/execute.py` | 保留已有 OMG125D → qpos36 → SONIC 路径，补充实测和回归测试 | 重写 SONIC 或正在跑的数据处理 |
| `tests/` 与 `configs/` | 增加 synthetic/真实小样本 smoke、模态/时间/损失/可逆性测试和新 YAML | 为了省事删除老的回归测试 |

保持文件数量克制：除列出的三四个确有职责的新文件，不再拆出抽象 BaseEncoder、TaskRegistry、ProcessorDAG、FusionEngine 等。

### 8.1 推荐的主模型内部接口（示意，实现可按既有风格简化）

```python
class MSCModel(nn.Module):
    def encode_conditions(self, batch):
        # Frozen encoder features or offline tokens -> trainable bridge.
        # Called exactly once per training forward / sampling request.
        return self.condition_bridge(batch)

    def forward(self, noisy_future, flow_time, batch):
        memory = self.encode_conditions(batch)
        return self.motion_generator(noisy_future, flow_time, memory, batch.get("motion_mask"))

    def compute_loss(self, batch, *, return_metrics=False):
        memory = self.encode_conditions(batch)
        # Reuse FlowMatching interpolation and velocity target.
        # Obtain predicted velocity, x_tau, tau, and valid mask.
        # Reconstruct x0_hat; denormalize; compute joint Vel/Acc losses.
        # Return scalar by default; optionally return named metrics.
        ...

    @torch.no_grad()
    def sample(self, motion_history, *, text_tokens=None, audio_tokens=None,
               video_tokens=None, num_steps=10, cfg_scale=1.0, **kwargs):
        # Build memory once, reuse for all Euler steps.
        # Null branch removes external modalities, keeps motion history.
        ...
```

该片段只规定调用方向和职责；实际函数签名应尽量保持仓库 `train.py`、`sample.py`、测试可持续工作。不得在 `sample()` 内进行每一步 Encoder 推理。`FlowMatching` 可以继续持有采样循环，但不能因新的 `ConditionMemory` 命名变化使采样无法读取 Batch/Device。

## 9. 实施顺序：四个可独立验收的 PR

### PR1 — MSC Release Dataset + OMG125D parity

- 新增 `release_dataset.py`、`temporal.py` 和可变长 collator。
- 对一个真实/合成 MSC Release 读 35 帧 qpos36 -> 125D；与当前 OMG Codec 的直接计算保持数值一致。
- 通过 Audio Link offset、Video interval、Text annotation、missing modality、split、actor/episode 边界测试。
- 不修改模型输出网络；默认不启动真实三模态训练。

### PR2 — Condition Bridge + Shared Flow Expert

- 将原 Text Cross-Attn / Audio FiLM 重构为**统一的 tokens + mask + temporal bias 条件接口**；Model 主干改为 6 层 512D；新增 Video Feature Adapter，但 synthetic 视频足以先测输入。
- 确保 Text-only、Audio-only、Video-only、History-only、Text+Audio 的 forward/backward 形状正确、loss finite；不同模态的 Adapter 与 Cross-Attention 参数在各自有效样本中得到非零梯度。
- `sample()` 保持 125D/30 帧、固定噪声可复现、CFG null 保留 History、Flow direction 正确。

### PR3 — Joint Dynamics Loss + Real Text/Audio Training

- 实现 125D joint angle 边界差分、速度/加速度监督、严格 mask 和训练数据统计归一化；保持 scalar `compute_loss` 兼容。
- 在相同数据/噪声/训练预算下跑 Flow-only、Flow+Vel、Flow+Vel+Acc 三组 A/B；比较验证 Loss、关节速度/加速度残差、输出 Jerk/SONIC tracking。
- 实施确定性 task sampler 与 checkpoint provenance，真实 OMG Text 与 AIST++ MERT 小样本完成 overfit / sampling。

### PR4 — Real Video Features Integration + Three-task Training

- 只有当真实 V-JEPA features、时间戳和身份映射已通过验证时，才接入真实 Video Task；使用本地数据小样本完成 Video-to-Motion。首版只读已有冻结特征，不触碰原数据处理进程。
- 配置 `task_weights={text:1/3,audio:1/3,video:1/3}` 做三任务联合训练；分别评估各模态，记录数据比例和 condition sensitivity。
- 若尚无 Video Feature，不要阻断 PR1–PR3 的完成；PR4 显式标记 pending 而非伪造结果。

## 10. 必须通过的测试与完成判定

**Tensor/Codec：** `history[B,5,125]`、`target[B,30,125]`、`velocity[B,30,125]`，`decode_125_to_qpos36` 保证有限、旋转规范；FK/Anchor 与旧 OMG 路径一致；不能跨 Episode。

**Temporal/Mask：** 验证 Audio 真实 PTS、`TimeMap` offset/scale、SONIC `1/30` origin 无双重补偿、Video interval、窗口边缘、估计对齐策略、部分/缺失模态；masked token 内即使含 NaN 也不影响有效输出（在进入线性投影前安全屏蔽），所有条件缺失时仍有有效 History Keys。

**Gradients：** 冻结 T5/MERT/V-JEPA 无梯度，三个 Modality Adapter 与 Expert 在对应任务有梯度；Joint Vel/Acc 梯度可达 Expert；权重置零时逐数值回归原 FM 标量。

**Loss：** 构造完全相同预测与真值时 Vel/Acc=0；构造交替帧高频误差时 Acc Loss 增大；构造匀速但正确的真实 Motion 不应仅因运动快而产生过大损失；验证 Future 第 0/1 帧历史边界、mask/短序列、tau=0/1、BF16 下 finite。

**Condition sensitivity：** 同一 Motion History / noise / seed，Text/Audio/Video 特征分别被打乱或置为缺失时对生成有可观察影响；训练后还需实际匹配评测，不能把网络有差异当成成功的语义对齐。

**Training：** Synthetic 两步 optimizer + overfit；真实 MSC Release 的 Text/MERT 短训练；checkpoint save/resume 精确且拒绝错误数据/编码器指纹；显存、参数数量和一步吞吐有报告，不能只给出“测试预期通过”。

**Motion Quality：** 不只展示 Flow Loss；采样连续动作 chunk，测 FK MPJPE / root drift / joint velocity & acceleration、Jerk、chunk 边界跳变、SONIC tracking error 与 failure rate；在可行时与同预算旧 125D OMG Flow 基线对照。不要将单段 motion 的低重建误差误认为长期执行稳定。

**Data Safety：** 任何模型训练代码不得修改正在运行的 Pipeline 的 Cache、Work、Release、原始文件或配置。只在新的 `src/msc`、`tests`、`configs`、`docs` 相关文件中修改，必要时仅读 `data_pipeline` 已发布的公共读取接口。

## 11. Codex 交付要求

1. 实施前先 `git status` 和确认 HEAD，与本规格的 `df5bda7` 对比；如果最新 HEAD 有接口变化，先说明差异和影响，不可默默覆盖。
2. 分四个里程碑实现；每个里程碑至少给出修改文件清单、选择理由、实际运行的测试命令及结果、未完成项。
3. 提供最终模型参数量、Peak GPU Memory（在可用 GPU 上）、最小真实训练配置和 sample/execute 命令；没有 GPU/真实 Video 权重时如实报告、不能虚报实测。
4. 不改 Motion Representation，不修改 MSC v1 Schema，不重算正在进行的 GMR/SONIC/MERT 数据，不加入 foot-contact 或 steering/闭环扩展。
5. 保留旧 OMG Flow 为可比较的 Baseline（通过代码版本/配置或轻量兼容路径），**避免为了保留兼容性而开发两套越来越复杂的框架**。

## 12. 最终完成标准

单个训练完成的共享 Action Expert，使用三个彼此独立的冻结 Encoder 的有效特征，在统一 `[B,30,125]` Flow Action Space 预测动作；能够分别完成 Text、Audio 和 Video 条件生成，所有条件只通过 Trainable Bridge + Cross-Attention 进入同一个网络。输出解码成 G1 qpos36，能够由 SONIC 跟踪。Joint Velocity/Acceleration 辅助监督被正确计算并得到实验对照，而不是仅在配置中出现。

**成功标准首先是可靠的三种独立条件生成和动作质量，不是本轮实现 Streaming / Steering / Closed Loop。**
