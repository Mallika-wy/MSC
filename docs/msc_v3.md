# MSC v3 模型架构

MSC 只维护 v3 架构。运动契约固定为 G1 Canonical 125D、30 FPS、5 帧历史和 30 帧未来。
训练、采样和评估使用同一个 Flow Motion Expert；历史始终保留，每个样本只使用一种外部模态。

## 架构与参数

```text
Frozen online T5 / offline MERT / offline V-JEPA (one per sample)
    -> independent adapter -> ConditionMemory <- History [B,5,125]
                                         |
Noisy future [B,30,125] -> projection + position + flow time
    -> 12 x [gated Self-Attention -> gated Cross-Attention -> gated FFN]
    -> LayerNorm -> Linear -> velocity [B,30,125]
```

Hidden=1024，Heads=16，FFN=3072，Dropout=0.1。
Self 使用非 causal RoPE；Self/Cross 使用 FP32 per-head QK 归一化后转换回工作精度。
每个子层为 `x + gate * Dropout(branch(LayerNorm(x)*(1+scale)+shift))`。
每层 `SiLU -> Linear(1024,9216)` 生成九组调制量，Linear 权重和偏置零初始化。
初始 Block 对有效帧为恒等映射；首步允许分支和 Adapter 梯度为零，门控更新后验证有效梯度。
输出头保持普通初始化，因此初始整个模型不等于零预测。

实际实例化正式 Text+Audio 模型计数：

| 项目 | 参数量 |
| --- | ---: |
| 可训练总计 | **296,883,767** |
| Flow Motion Expert | 292,023,613 |
| 单个 MotionBlock | 24,138,768 |
| 冻结 T5 | 109,628,544 |
| 模型总计（含 T5） | 406,512,311 |

逐模块统计，`expert.layers.*` 为全部 12 层合计：

| 模块 | 可训练 | 冻结 |
| --- | ---: | ---: |
| `expert.time_embed` | 2,099,200 | 0 |
| `expert.input_proj` | 129,024 | 0 |
| `expert.layers.alpha` | 192 | 0 |
| `expert.layers.self_attn` | 50,380,800 | 0 |
| `expert.layers.cross_attention` | 50,380,800 | 0 |
| `expert.layers.ff` | 75,546,624 | 0 |
| `expert.layers.modulation` | 113,356,800 | 0 |
| `expert.norm` | 2,048 | 0 |
| `expert.output` | 128,125 | 0 |
| `condition_bridge.history` | 1,178,874 | 0 |
| `adapters.text` | 1,838,592 | 0 |
| `adapters.audio` | 1,838,592 | 0 |
| `condition_bridge.types` | 4,096 | 0 |
| `frozen_t5` | 0 | 109,628,544 |

MERT 是离线特征提取器，不驻留训练模型；V-JEPA 尚无真实特征，未实例化对应 Adapter。
两者均不计入上述 300M 可训练参数。启用真实 Video Adapter 后需按真实 K/D 重新审计。

## 条件与视频契约

ConditionMemory 保留 tokens、mask、relative_times、temporal_support、modality_ids。
训练样本在条件 Dropout 前恰好激活一种外部模态；不同样本可混合任务。
CFG Null 仅清除外部条件，History 五帧始终有效。文本是全局条件，不施加局部时间惩罚。
音频使用完整离线 MERT Token 序列与精确时间戳；媒体时间继续由既有 TimeMap/select_tokens 映射。
采样每个请求只编码一次条件。

Video 保留 `[B,L,K,D]` 特征、`[B,L,K]` Mask、`[B,L,2]` 区间与特征级采样。
真实 K/D 由 Release 元数据确定；配置禁用 Video 且不填写猜测值。
合成测试显式使用 `(K,D)=(2,6)` 和 `(3,10)`，验证前后向、空间嵌入梯度、
无效 Token、时间区间及有限值；这些尺寸不代表真实 V-JEPA。
部分空间 Token 无效时，其时间支持也被清零；Attention Mask 不会被时间偏置抵消。

混合 batch 的互斥外部模态共享补齐槽位，Memory 长度为
`5 + max(text_length, audio_length, flattened_video_length)`。
有效 token、Mask、模态标识、精确时间戳与 Video support 保持对应。

## Flow、监督与 RTC

保持 `x_tau=(1-tau)*x0+tau*noise`、`v=noise-x0`，以 Euler 从 tau=1 积分到 0；
时间缩放 1000，默认采样 10 步。Flow Loss 按有效帧和动作通道归一化。
关节速度、加速度辅助损失权重分别为 0.05、0.01，在第 1000 步后启用，
尺度由选定 Train 窗口精确统计，反标准化和残差计算使用 FP32。

RTC 默认关闭。训练侧支持逐样本延迟、clean prefix 和 Loss Mask；
训练式采样保留硬前缀，测试时采样保留 VJP 引导，均可与 CFG 组合。
条件在每次采样请求中只编码一次，CFG Null 保留全部五帧历史。

## 验证方式与限制

[模型测试](../tests/)覆盖零门控与梯度、缩小模型的 Text/Audio/合成 Video 前后向、
时间对齐、Video temporal support、RTC/CFG、混合 Memory 与拼接布局的前向及梯度对照。
真实在线 T5 和 CUDA 编译对照按环境条件执行；缺少环境时明确 skip。

[参数审计入口](../src/msc/models/audit.py)实例化当前配置，不读取数据集或动力学统计量。
训练和验证使用已发布的窗口索引，不重建 Release，也不执行离线媒体编码器。

Video 尚无真实 V-JEPA 特征，默认关闭；合成 K/D 只用于接口测试。
真实视频训练、MP4 编码和完整轨迹质量评估仍待具备真实输入与充分独立验证数据。

训练配置、启动和硬件测量见 [双 A6000 训练](msc_v3_2xa6000.md)。
