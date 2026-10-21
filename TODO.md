# Task: Build a Minimal MSC Framework from Ψ₀ and OMG

你现在位于一个新建的 `MSC/` 工作目录。

我正在研究 **Multimodal Streaming Humanoid Motion Generation**。希望以开源项目 Ψ₀ 和 OMG 为基础，构建一个简洁、模块化、能够实际训练和采样的 MSC 代码框架。

**请不要照搬两个仓库。我要的是最小、干净、可扩展的 PyTorch Research Framework。**

## 一、首先阅读现有项目

参考仓库：

- Ψ₀：/home1/gmh/wangyang/motion/Psi0
- OMG：/home1/gmh/wangyang/motion/OMG

先检查工作区是否存在本地 `Psi0/` 和 `OMG/`，尤其是我已经修改成 Flow Matching 的 OMG 分支。优先使用本地代码，保留其已完成的修改。

如果找不到本地项目，请先报告需要的路径，不要假定官方 OMG 已包含我的 Flow Matching 修改。

### Ψ₀ 重点阅读

- `src/psi/trainers/trainer.py`
- `src/psi/trainers/posttrain.py`
- `src/psi/models/psi0.py`
- `src/psi/config/`
- 数据加载、Checkpoint、训练入口和 Flow Matching 相关代码

目标是理解训练工程的组织结构，**不是复制完整 Ψ₀ 模型**。

### OMG 重点阅读

- `src/omg/generation/denoisers/transformer.py`
- Generation、Diffusion / Flow Matching 相关实现
- Motion Representation、Normalization、Dataset
- Motion History Conditioning
- Text Encoder 与 Audio Conditioning
- 模型训练和推理入口

确认每个模块的真实输入输出，不要根据文件名猜测接口。

## 二、项目目标

构建一个新的、独立的 `msc` Python Package。

核心设计：

**Training Framework：借鉴 Ψ₀**

**Motion Generator：flow matching 生成。目前omg是diffusion架构，psi0中有flow matching生成的代码**

**Motion Representation：完全使用 OMG 125D G1 Representation**

**Multimodal Encoders：每个模态拥有独立编码器和 Adapter**

第一阶段只支持：

1. Text-conditioned Motion Generation
2. Audio-conditioned Motion Generation
3. 可选的 Text + Audio 联合条件输入

Video 仅预留接口，不实现 Video Encoder，不加载视频模型。

## 三、模型设计

### 3.1 Motion Generator

采用 OMG 的 G1 motion representation：

- Root position：3D
- Root rotation：6D
- G1 joint angles：29D
- Link positions：87D
- Total：125D

默认配置：

- Motion FPS：30
- History Frames：10
- Future Frames：60
- Output Shape：`[B, 60, 125]`

保留 OMG Motion DiT 及其 History Conditioning。

### 3.2 Flow Matching

看psi0的flow matching的实现。

### 3.3 Text Encoder

第一版优先复用 OMG 的 T5 文本条件编码方式。

要求：

- 可冻结 Text Encoder
- 可以缓存预计算的文本特征
- 有独立 Text Adapter
- 输出 tokens 供 OMG DiT Cross-Attention 使用
- 支持 Text Conditioning Dropout

### 3.4 Audio Conditioning（优先兼容 OMG 训练数据）

本项目当前仍然使用 OMG 官方训练数据，因此第一阶段的 Audio Conditioning 必须优先兼容 OMG 已有的数据格式，不要引入额外的音频预处理依赖或重新提取全部音频特征。

#### 3.4.1 OMG Audio Features + Audio Adapter

默认配置：

audio_backend = "omg_precomputed"

直接读取 OMG Dataset 已有的音频特征：

数据字段：omg.audio.feature

对应条件有效性字段：omg.condition.has_audio

默认特征维度：35D

Audio FPS：与 OMG Motion FPS 对齐，默认 30 FPS

训练输入：audio_features: [B, 60, 35]

音频有效性掩码：audio_mask: [B, 60]

第一阶段不要对 OMG 数据集重新提取音频特征，也不要直接改变现有 35D 特征的数值含义。

注意：OMG 代码中存在不同的 35D 特征提取方案，例如 current35 和 aistpp_librosa35，它们并不等价。

请先检查官方 OMG 数据集、metadata 和相关预处理代码，确认已有 omg.audio.feature 的实际来源。如果无法确定，请明确说明，不能仅根据 audio_dim=35 猜测特征类型。

#### 3.4.2 Audio Adapter

第一阶段只实现一个轻量、可训练的 Audio Adapter，优先复用 OMG 已有设计：

AudioAdapter(
    input_dim=35,
    output_dim=motion_dit_hidden_dim,
)

推荐结构：

nn.Sequential(
    nn.LayerNorm(35),
    nn.Linear(35, hidden_dim * 2),
    nn.SiLU(),
    nn.Linear(hidden_dim * 2, hidden_dim),
)

输入：

[B, 60, 35]

输出：

[B, 60, hidden_dim]

将输出作为 Frame-wise Audio Conditioning 输入 OMG-Flow Motion DiT。

优先复用 OMG 原有的 per_layer_film 注入机制，不要为第一版重新设计复杂的 Audio Cross-Attention。

保留 OMG 原有的 Audio Conditioning Dropout 和 Audio Mask 逻辑。

#### 3.4.3 数据兼容性要求

使用现有 OMG Dataset Adapter 读取音频数据。

必须满足：

Audio Features 与 Motion Frames 按时间戳对齐。

保留原始 has_audio 掩码，不能把缺失音频当成真实的全零音频。

对短序列的 padding 使用有效性掩码。

支持 Text-only、Audio-only、Text+Audio 训练和推理。

不改变 OMG 官方数据已有音频特征的归一化约定。

不混用不同提取方法生成的 35D 特征。

在配置、checkpoint metadata 中记录 Audio Feature Dimension、Feature Type、FPS 和相关预处理版本（能从数据来源确认的部分）。

没有 Audio Condition 的样本仍然可以正常参与训练。

### 3.5 Video

只定义未来可扩展的接口。

暂不实现模型，不下载任何权重，不引入视频相关大型依赖。

### 3.6 Multimodal Model

提供统一入口，例如：

```python
MSCModel(
    motion_generator,
    text_encoder,
    audio_encoder,
    text_adapter,
    audio_adapter,
)
```

训练接口尽量简单：

```python
loss = model.compute_loss(batch)
```

推理接口：

```python
motion = model.sample(
    history=history,
    text=text,
    audio=audio,
    num_steps=10,
)
```

支持 text-only、audio-only 和 text+audio。

## 四、训练工程

借鉴 Ψ₀ 的代码组织，但只保留必要功能：

- PyTorch
- Config Management
- Trainer
- Dataset / DataLoader
- Optimizer / Scheduler
- Checkpoint Save / Resume
- Logging
- Evaluation Entry Point
- Random Seed
- BF16 / AMP（硬件支持时）

第一版不需要复杂的分布式训练封装，可以采用一个简单、容易扩展的 Trainer。

不要引入 Ψ₀ 的 Qwen、视觉语言模型训练、机械臂任务、VLM Collator 或无关依赖。

## 五、Dataset

优先复用 OMG 的数据读取和 motion normalization 逻辑。

统一 batch 字段，例如：

```python
batch = {
    "motion_history": ...,  # [B, 10, 125]
    "motion_future": ...,   # [B, 60, 125]
    "text": ...,            # optional
    "audio_features": ...,  # optional
    "motion_mask": ...,
    "audio_mask": ...,
}
```

具体数据字段应根据 OMG 真实 Dataset Schema 适配。

提供两个模式：

1. Synthetic Dataset：不需要下载真实数据，用于快速测试。
2. OMG Dataset Adapter：读取真实 OMG 数据，用于正式训练。

必须确保归一化方式与数据格式一致，不要重新发明 125D motion 格式。

## 六、建议代码结构

可以根据实际依赖微调，但尽量保持简单：

```text
MSC/
├── README.md
├── pyproject.toml
├── configs/
│   ├── model.yaml
│   ├── train.yaml
│   └── data.yaml
├── src/msc/
│   ├── models/
│   │   ├── msc_model.py
│   │   ├── motion_generator.py
│   │   ├── flow_matching.py
│   │   └── conditioning/
│   │       ├── text.py
│   │       ├── audio.py
│   │       └── fusion.py
│   ├── data/
│   │   ├── synthetic.py
│   │   ├── omg_dataset.py
│   │   └── collator.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── checkpoint.py
│   ├── train.py
│   └── sample.py
└── tests/
    ├── test_shapes.py
    ├── test_flow.py
    ├── test_conditioning.py
    └── test_train_step.py
```

## 七、本阶段明确不做

- 不使用 Qwen2.5-Omni、Qwen3-VL 或任何大语言模型作为统一 Backbone
- 不实现 Video Encoder
- 不实现 Ψ₀ 原有的机械臂任务和 70D Action Format
- 不实现机器人部署、ROS、ZMQ；数据预处理的离线执行使用 SONIC
- 不实现 Streaming Runner、RTC、异步重规划或 Execution Feedback
- 不训练新的低层 Tracking Policy
- 不自动下载大型预训练权重或数据集

这些都留到后续阶段。

## 八、必须通过的验收标准

完成后实际运行测试，而不是仅生成代码。

**Test 1：模型 Forward**

- 输入 `[B,10,125]` 历史动作
- 输出 `[B,60,125]` Flow Velocity
- 检查形状和 NaN / Inf

**Test 2：Flow Matching**

- 能计算 loss
- 能反向传播
- Euler 采样时间方向正确
- 能在小型 synthetic 数据上 overfit

**Test 3：Text / Audio**

- Text-only 可以 Forward
- Audio-only 可以 Forward
- Text+Audio 可以 Forward
- 两种模态均可关闭，不会触发空张量错误

**Test 4：Training**

- Synthetic Dataset 至少完成几个 optimizer steps
- Loss 正常记录
- Checkpoint 能保存和恢复

**Test 5：Sampling**

- 成功生成 `[1,60,125]` 动作
- 输出有限数值
- 不依赖机器人部署环境

真实 motion 的质量需要正式训练后验证，不能因为测试通过就宣称具有生成能力。

## 九、执行顺序

请先检查两个仓库并给出非常简短的模块复用方案，然后立即开始实现，不要只停留在架构文档。

优先级：

1. OMG Representation + Flow Motion Generator
2. Synthetic Data + Flow Training
3. Text Conditioning
4. Audio Conditioning
5. OMG Dataset Adapter
6. Checkpoint / Sampling / Tests

如有难以从源码判断的接口，请明确标记并给出最小兼容实现，不要静默猜测。

修改只发生在 `MSC/` 中，不要直接改动原 Ψ₀ 或 OMG 仓库。

移植代码时保留必要的版权声明和来源说明。

最终请提供：

- 项目目录树
- 从 Ψ₀ / OMG 复用的模块清单
- 安装命令
- Synthetic Training 命令
- 真实 OMG 数据训练命令（需要用户填写的数据路径）
- Sampling 命令
- 实际运行的测试结果
- 当前还没有实现的功能及下一阶段建议

**目标是得到一个能够真正运行、结构清楚、后续容易扩展 MSC Streaming 研究的最小代码基线，而不是一个包含大量空接口的脚手架。**
