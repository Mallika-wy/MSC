本文保留早期设计讨论，仅作历史参考；当前实现与使用方式以 `msc_dataset_v1.md` 为准。

我认为，对于你的 MSC 项目，最优雅的 Pipeline 不是把代码拆成越来越多的文件，而是让每一层只回答一个问题。

我刚确认了 GitHub 最新版本，仍然是 `25567ab`（`data_pipeline3`）。你现在已经有了不错的模块划分，但有一个根本问题还没有解决：

整个 Pipeline 仍然以 Audio-to-Motion 任务为中心，而不是以 Motion 数据和它关联的多模态资产为中心。

如果是我来设计，我会采用：

# 一、最终架构：Dataset-centric + Independent Processors

核心只有五个概念：

1. Source：原始数据是什么，包含哪些模态？
2. Processor：怎样把某种数据变成标准表示？
3. Artifact：每个处理步骤产生了什么？
4. Alignment：不同模态如何在时间上对应？
5. Release：怎样打包成 MSC 能使用的数据集？

Source Datasets

OMG · AIST++ · FineDance · AIOZ-GDance · Future Datasets

Source Adapters

只做数据发现、解码、身份和配对元数据

统一返回 SourceSequence

Motion Processor

Decode → GMR → QC → Optional Tracker

Audio Processor

WAV → MERT → Cache

Video Processor

Crop → V-JEPA → Cache

Alignment & Release Builder

TimeMap · Quality Policy · Modality Links · Split

MSC Unified Dataset v1

G1 qpos36 + Text + Audio Features + Video Features + Metadata

MSC Training DataLoader

Window Sampling · OMG 125D · Temporal Alignment

其中三个 Processor 不是必须按顺序执行的。它们相互独立，只有在最后组装 Release 时才建立关联。

这才是真正适合你的多模态研究方向的架构。

## 二、我最想改掉的，是当前 Source 的定义

你现在有这样的目录：

```
sources/
├── audio/
│   ├── aistpp/
│   ├── finedance/
│   └── gdance/
├── text/
└── video/
```

我不太赞成这个分类。

原因是 AIST++ 不是一种 Audio 数据源，AIST++ 是一个数据集。它可以同时包含 Motion、Audio、Video。

如果将来处理 AIST++ 的视频，你是不是还要创建 `sources/video/aistpp/`？这样同一个数据集的身份识别、时间偏移和文件命名规则会重复实现。

我建议改成：

```
sources/
├── aistpp/
│   ├── discover.py
│   └── motion.py
├── finedance/
│   ├── discover.py
│   └── motion.py
├── gdance/
│   ├── discover.py
│   ├── motion.py
│   └── split.py
└── omg/
    └── discover.py
```

每个数据集只负责描述它自己，不负责调用 MERT、V-JEPA 或 Tracker。

### Source Adapter 统一返回什么？

我推荐一个 `SourceSequence`：

```
@dataclassclass SourceSequence:    sequence_id: str    source_dataset: str    split_group_id: str    motions: list[SourceMotion]    audios: list[SourceAsset]    videos: list[SourceAsset]    texts: list[SourceText]    alignments: list[SourceAlignment]
```

这只是数据发现时的描述对象，不是最终存储格式。

它可以自然表达：

- OMG：Motion + Text。
- FineDance：Motion + Audio。
- AIST++：Motion + Audio + 多机位 Video。
- GDance：多个演员的 Motion + 一份共享 Audio。

这样以后新增一个模态，不需要重新定义 Sample 的基本含义。

目前 [sources/sample.py](https://github.com/Mallika-wy/MSC/blob/main/data_pipeline/sources/sample.py?utm_source=chatgpt.com) 的 `Sample` 强制包含 Motion 和 Audio，这正是下一步需要改造的地方。

# 三、Processor 应该尽量简单

我不建议搞一套复杂的 `BaseProcessor`、`AbstractStage`、`PipelineNode`、`PluginRegistry`。

你目前只有几种非常明确的处理任务，普通函数和少量数据类就足够。

| Processor | 输入                 | 输出                         | 绝对不应该依赖     |
| --------- | ------------------ | -------------------------- | ----------- |
| Motion    | 原始人体动作或 G1         | G1 qpos36 + QC             | MERT、视频编码器  |
| Audio     | WAV                | MERT tokens + timestamps   | GMR、Tracker |
| Video     | MP4 + person track | V-JEPA tokens + timestamps | GMR、MERT    |
| Text      | Caption            | 文本标注 / 可选 T5 features      | 音视频处理器      |

例如：

```
motion = process_motion(source_motion, config)audio = process_audio(source_audio, config)video = process_video(source_video, config)
```

返回值统一是某种处理结果 Artifact，而不是嵌套很深的字典：

```
@dataclassclass FeatureArtifact:    asset_id: str    feature_path: Path    timestamps_path: Path    metadata: dict
```

Motion 还可以具有 `reference`、`tracker_executed` 两种 Artifact，但不应该规定每一个 Motion 都必须经过 Tracker。

### 我尤其建议：Tracker 是可选的 Motion Stage

当前 `pipeline.py` 把 `track()` 作为正式样本接受的必要步骤。

这在你目前的 Audio-to-Motion 实验里能够使用，但长期会限制 OMG Text-to-Motion 等数据源。

正确的设计应该由任务配置决定：

```
motion:
  target: g1_qpos36
  fps: 30

  retarget:
    enabled: auto

  tracking:
    enabled: true

  labels:
    publish:
      - retarget_reference
      - tracker_executed
    train_default: tracker_executed
```

`enabled: auto` 是建议的新配置语义：原生 G1 跳过 GMR，SMPL 数据运行 GMR。

注意发布标签和训练默认标签应该分离。保留两种 Motion 不代表训练时自动使用两种。

# 四、最优雅的 `pipeline.py` 应该非常短

我希望你最后的主文件可以接近下面这种程度：

```
def build_dataset(config):    source = create_source(config.source)    builder = ReleaseBuilder(config.output)    for sequence in source.discover():        try:            # 1. Motion processing            motions = [                process_motion(m, config.motion)                for m in sequence.motions            ]            # 2. Select eligible motion tracks            selected = select_motion_tracks(                motions, config.quality            )            # 3. Independent modality processing            assets = process_assets(                sequence=sequence,                selected_tracks=selected,                config=config,            )            # 4. Temporal alignment            links = align_modalities(                sequence=sequence,                motions=selected,                assets=assets,            )            # 5. Collect release records            builder.add(                sequence=sequence,                motions=motions,                assets=assets,                links=links,            )        except Exception as error:            builder.record_failure(sequence, error)    builder.validate()    return builder.publish()
```

这是表达目标职责的伪代码，错误处理策略、缓存状态和并发安全还需在实现中明确。

你看这个主流程，有三个明显好处。

第一，一眼就能理解数据怎样流动。

第二，如果音频编码换了模型，`pipeline.py` 不需要修改。

第三，如果将来增加 Video-to-Motion，只需要接入 Video Processor 和对应的资产关联，不需要大幅修改 Motion Processing。

其中 `process_assets()` 可以只用一个很简单的模态分派函数，没必要引入 DAG 调度框架。

## 五、我还会把三个概念彻底分开

这点特别重要，因为你以后会处理的数据越来越多。

Cache

存放昂贵计算的结果

GMR 输出、MERT 特征、V-JEPA 特征

Work / Reports

保存处理过程和失败信息

状态、QC、Tracker 回放、错误日志

Release

只保存训练所需的已验证数据

Parquet、FeatureRefs、TimeMap、Manifest

Cache 不应该由样本编号决定，而应由资产内容和处理配置决定。例如同一首 GDance 歌曲只计算一次 MERT；同一个视频在相同裁剪与编码器配置下也只计算一次视觉特征。

Release 不应该再次调用 MERT 或 GMR，只负责整理 Artifact、验证引用和发布。这会让重新打包数据非常便宜。

而训练 DataLoader 也不应该调用这些重型预处理器。

## 六、推荐的最终目录结构

我会控制在这个规模，不再无限增加层级。

```

data_pipeline/
├── cli.py
├── config.py
├── pipeline.py
│
├── sources/
│   ├── base.py           # SourceSequence 数据协议
│   ├── aistpp/
│   ├── finedance/
│   ├── gdance/
│   └── omg/
│
├── processors/
│   ├── motion/
│   │   ├── decode.py
│   │   ├── retarget.py
│   │   ├── tracking.py
│   │   └── quality.py
│   ├── audio/
│   │   └── mert.py
│   └── video/
│       ├── crop.py
│       └── vjepa.py
│
├── storage/
│   ├── cache.py
│   └── artifacts.py
│
├── schema/
│   ├── records.py
│   ├── manifest.py
│   ├── validation.py
│   └── time_map.py
│
├── release/
│   ├── builder.py
│   ├── package.py
│   └── validate.py
│
└── tests/
    ├── test_sources.py
    ├── test_motion.py
    ├── test_audio.py
    ├── test_video.py
    ├── test_alignment.py
    └── test_release.py

```

这里最大的变化是：

不要再有 `release/audio.py` 和未来的 `release/video.py` 这种以模态命名的发布器。

你最终只有一种 MSC Release 格式，就应该只有一个 `ReleaseBuilder`。

它负责把 `MotionArtifact`、`FeatureArtifact`、`ModalityLink` 等组装为你现有的六张 Parquet 表，不关心特征最初是 MERT 还是 V-JEPA 生成的。

现在的 MSC Dataset v1 六表结构可以直接保留。

# 七、真正让代码优雅的还有配置方式

我建议通过配置组合任务，而不是为每一种任务开发新的 Pipeline。

例如：

```
source:
  dataset: aistpp
  root: /data/AISTpp

motion:
  target_fps: 30
  representation: g1_qpos36
  retarget: true
  tracking: true

audio:
  enabled: true
  encoder: mert_v1_95m
  frozen: true

video:
  enabled: true
  crop: true
  encoder: vjepa2_1_base
  frozen: true

text:
  enabled: false

release:
  format: msc_v1
  label_policy: tracker_executed
  split_policy: song_disjoint
  alignment_policy: allow_estimated
  output: /data/msc/aistpp
```

这样 Audio-to-Motion 只启用 Audio，Text-to-Motion 只启用 Text，组合训练同时启用多个模态。

但有一个前提：启用某个模态不代表一定能找到可信的配对关系。例如 AIST++ 的文件名能对应同一条舞蹈序列，却不能自动保证视频、音频和 Motion 在毫秒级准确同步。Release Builder 应检查 `ModalityLink` 的覆盖范围和 `alignment_status`，并根据配置决定纳入、跳过或拒绝。

另外，我会把 `split_policy` 设置为可配置并且可审计的规则。对于严格 Music-to-Motion 泛化实验，`song_disjoint` 比当前允许共享歌曲跨 Split 的策略更稳妥。

## 八、如果继续让 Codex 改，我建议只分三步

| 顺序  | 本轮目标                                                                 | 验收标准                                                    |
| --- | -------------------------------------------------------------------- | ------------------------------------------------------- |
| PR1 | 重构 `SourceSequence` 和 Source Adapter，不再要求每条数据有 Audio                 | OMG Text-only、AIST++ Audio、GDance 多演员均可统一 Discover      |
| PR2 | 拆分 `process_motion` / `process_audio` / `process_video`，让 Tracker 可选 | 模态之间没有反向依赖；每个 Processor 可独立单元测试                         |
| PR3 | 将 `release/audio.py` 改成通用 `ReleaseBuilder`，统一缓存与发布                   | Text-only、Audio-only、Video-only、Combined 都能生成合法 Release |

当前 `audio/mert.py`、`schema/time_map.py`、现有 Video Crop 几何逻辑和运动学代码，除已知数值问题外，尽量不要修改算法本身。

这三步做完之后，就停止 Data Pipeline 的架构重构，转向 `src/msc/data/` 的统一训练加载器。

## 九、什么样的代码才算真正优雅？

我会用下面这五条来判断：

新增数据集不改主 Pipeline

只增加新的 Source Adapter，复用现有 Processors

新增模态不改 Motion 处理

Video Encoder 不应该修改 GMR 或 Tracker

换模型不重处理其他数据

换 MERT 只重新计算音频缓存，不触碰 G1 Motion

处理失败可追溯

Cache、QC Report、Release 三者边界明确，失败不进入训练集

磁盘格式不随模型架构改变

无论 OMG DiT 还是 Ψ₀-style Action Head，都读取同一 MSC Release

架构验收 0/5

&#x20;复制验收标准

我对你项目的最终推荐是：

\\[ \boxed{ \text{Source} \rightarrow \text{Independent Processors} \rightarrow \text{Artifacts + Alignment} \rightarrow \text{One Release Builder} } \\]

你现在已经有了其中约三分之二的基础，不需要第四次全面重写。主要是调整 `Source` 的抽象，解除 `pipeline.py` 对 Audio/GMR/Tracker 的强绑定，然后把 `release/audio.py` 泛化为唯一的 `ReleaseBuilder`。

做完这三件事，整个数据工程不仅会容易读，而且能自然服务于你最终的 Text / Audio / Video → G1 Motion 训练。
