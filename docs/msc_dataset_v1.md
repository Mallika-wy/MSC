# MSC 数据处理与共享分片发布

最终数据包的目录、全部表字段、动作/音频/视频内容及读取示例见 [Release 通用数据格式说明](Release数据格式说明.md)。

主流程为 `SourceSequence → Independent Processors → Artifacts / Alignment → ReleaseBuilder`。
支持 AIST++、FineDance、AIOZ-GDance 和本机清洗后的 OMG Motion+Text。
本轮只负责预处理和发布；`src/msc/data/` 的统一训练加载器尚未接入此 Release。
现有官方 OMG 训练适配器仍要求其固定官方版本，不能直接用于本地清洗版本。

## 模块与配置

- `sources/<dataset>/` 直接返回 SourceSequence，负责发现、身份、原始时间关系和解码。
- `sources/base.py` 定义 SourceSequence、SourceMotion、SourceAsset、SourceText、SourceAlignment。
- `motion/processor.py`、`audio/processor.py`、`video/processor.py` 分别提供独立处理入口。
- `motion/retarget.py`、`motion/tracking.py` 只负责计算并返回数组；动作缓存统一由 `motion/processor.py` 管理。
- `motion/tracking_video.py` 负责执行回放；`video/` 负责原始人物视频。
- `storage/` 保存带指纹的数组 Artifact，以及增量工作记录。
- `release/builder.py` 统一组装六张表；`release/export.py` 增量写出共享分片并原子发布。

配置采用 `source / motion / audio / video / text / quality / release`。
只支持上述配置格式，已移除旧 `dataset/root/retarget/tracker/output` 顶层字段。
所有配置路径相对于 YAML；只解析启用阶段所需的模型路径，不自动下载任何数据或模型。

```yaml
source:
  dataset: omg
  root: ../../../data/OMG-Data
motion:
  target_fps: 30
  representation: g1_qpos36
  retarget: {enabled: auto}
  tracking:
    enabled: true
    sonic_root: ../../third_party/sonic
    encoder: ../../model/sonic/model_encoder.onnx
    decoder: ../../model/sonic/model_decoder.onnx
    observation_config: ../../model/sonic/observation_config.yaml
    providers: CUDAExecutionProvider
    inference_batch_size: 32
audio: {enabled: false}
video: {enabled: false}
text: {enabled: true}
release:
  output: ../../data/processed/omg-text
  publish_labels: [source_g1, tracker_executed]
  train_default: tracker_executed
  max_shard_frames: 1000000
  split_policy: preserve
  alignment_policy: allow_estimated
```

OMG 读取当前 `meta/info.json`、episode 元数据和 `observation.state[36]`，不使用下一帧参考 `action`。
保留原始 episode 编号、上游来源、split 和清洗报告校验和；不要求未经修改的官方 manifest 哈希。
读取会校验 episode 身份、全局索引、局部帧号、30 FPS 时间戳和 qpos 数值。
OMG 原生 G1 跳过 GMR，但按上述配置交给 SONIC 重执行；原始 `source_g1` 仍是执行动作的父记录。
不启用 tracking 时，原始动作可直接引用源分片行范围和内容校验和。人体动作保留 GMR 和参考质检；FineDance/GDance 的 SMPL-X 近似不变。

`publish_labels` 与 `train_default` 分离；发布 tracker_executed 必须同时发布其参考父动作。
执行标签去掉初始化帧，从源时间 `1/30` 秒开始；资产映射和文本区间同时修正。

## 目录与文件数量

```text
output/
├── config.json
├── summary.json
├── cache/
│   ├── motion.sqlite               # NPZ 数组包作为数据库记录，非逐动作文件
│   ├── audio/features.sqlite       # 共享音频的 MERT 特征和时间戳
│   └── prepared/features-*/audio/shard-*.parquet
├── work/
│   ├── results.sqlite              # 按发现顺序索引的序列/演员结果、QC、错误和 traceback
│   ├── progress.json               # 有界频率更新的统计
│   └── diagnostics/                # 仅 --diagnostic 时保存详细中间文件/回放
└── release/
    ├── dataset.json
    ├── meta/
    │   ├── sequences.parquet
    │   ├── motions.parquet
    │   ├── assets.parquet
    │   ├── links.parquet
    │   ├── annotations.parquet
    │   ├── features.parquet
    │   └── provenance.json
    ├── motion/
    │   ├── train/shard-00000.parquet
    │   ├── val/shard-00000.parquet
    │   └── test/shard-00000.parquet
    ├── reports/
    │   ├── quality-00000.parquet
    │   └── dataset_summary.json
    └── features/                   # bundle 时包含
        ├── audio/shard-*.parquet
        └── video/shard-*.parquet
```

只有实际存在的 split/模态才创建目录。默认每个动作分片最多一百万帧，在动作边界切换；
单条超长动作独占一个分片。多条短动作共同组成 row group，避免每条动作一个 row group。
`motions.storage_uri + row_start + num_frames` 精确定位动作；每条动作时间从零开始。
G1 统一为根位置 xyz、根四元数 wxyz 和 29 个关节角，共 36 维，单位米/弧度，Z 向上。

QC 每行包含 `motion_id / accepted / report_json`。`quality_ref` 引用共享 Parquet，按 motion_id 查询。
旧式 JSON QC 仍可读取。新布局通过 manifest 的 `quality_storage`、`motion_storage` 显式标记，六表列不变。
不合格动作与错误记录仅留在 work；发布报告汇总处理数量及拒绝原因。

特征每行包含 `feature_id / shape / dtype / values / timestamps / timestamp_shape`，
按 feature_id 读取，`feature_uri` 和 `timestamp_uri` 指向同一共享文件。
当前特征以原生 FP16 存储和读取，时间戳保持 FP64；写入需要 PyArrow 15 或以上版本。
默认每个特征分片最多约 256 MiB 数组数据，单个超大资产可独占。
`release.features.read_feature` 同时支持旧 NPY 与新 Parquet 特征。
manifest 用 `feature_storage`、`package_mode`、`features_bundled` 标记特征布局与打包方式。

缓存采用 SQLite 事务完成标记、内容校验和和固定数量的文件锁，进程退出后锁自动释放。
音频数值和时间戳在首次生成时检查，缓存命中不再重复扫描；共享文件的指纹复用已有缓存。
不同资产共享内容和处理配置时复用同一记录；换 MERT 不影响动作缓存，改 Tracker 不重算 GMR。
每个 worker 复用 SMPL-X 模型、SONIC 的 MuJoCo 模型和配置，每条动作使用独立仿真状态。人体解码只计算所需的 55 个关节，避免逐帧网格蒙皮；GMR 使用显式 Mink 参数复用关节限位对象。这些优化保持 30 Hz 数据、50 Hz 控制和现有质量策略。
正常运行不会留下逐动作 NPZ、JSON 或视频。GMR 和参考质检直接传递数组；SONIC 使用临时 rollout，结束后清理。
启用诊断时才保留原始 rollout、报告和回放。
显式启用的人物视频裁剪按唯一视频/人物/配置保存媒体和空间时间映射；这类媒体缓存仍按视频保存，
不会因为同一视频关联多个动作而复制。

发布按序列增量写元数据和动作；并发任务数有界。最终验证仍需在内存中建立元数据引用索引，
但不会一次载入全部动作或特征数组。所有发布都验证完整元数据和 QC，默认深检前 32 条负载；
可用 validate 命令扩大深检范围。

## 运行与恢复

从原始 AIST++ 一次完成处理和最终打包（输出目录应尚无已发布 release）：

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m data_pipeline.cli run \
  --config configs/data_pipeline/aistpp_motion.yaml \
  --root /home1/gmh/wangyang/motion/data/AISTpp \
  --output data/processed/audio/aistpp-new --workers 16 --finalize
```

`--root` 指定原始数据目录。`--finalize` 在发布后把特征写入 release，逐条验证动作和特征，再清除输出目录中的 work、cache 和运行日志；最终只保留 release、config.json 和 summary.json。未完成打包时保留处理缓存以便恢复。原始数据和模型文件不受清理影响。
动作全量校验按 Parquet 分片默认使用 8 个进程，每个所需数据块只读取一次；小样本不启动进程池。元数据关联检查与原有数值、时间戳、四元数规则保持不变。独立校验可用 `validate --workers 8` 调整并行数；打包中断后可继续验证已完整生成的 bundle。

在 MSC 根目录使用 `.venv/bin/python`。GMR 源码和 G1 资产位于 `third_party/gmr`；SONIC 官方部署源码、配置和 G1 仿真模型位于 `third_party/sonic`，权重在 `model/sonic`。运行时不读取兄弟目录中的 GMR、SONIC 或 OMG 代码；源数据目录仍由配置指定。执行控制为 50 Hz，发布动作保持 30 Hz，首个执行标签对应源时间 `1/30` 秒。

```bash
python -m data_pipeline.cli run --config configs/data_pipeline/omg_text.yaml --limit 10 --dry-run
python -m data_pipeline.cli run --config configs/data_pipeline/omg_text.yaml --limit 10
.venv/bin/python -m data_pipeline.cli run --config configs/data_pipeline/aistpp_motion.yaml --workers 16 --diagnostic
python -m data_pipeline.cli run --config configs/data_pipeline/gdance.yaml --sequence <group-id>
python -m data_pipeline.cli validate --dataset <output>/release --deep-sample 100
python -m data_pipeline.cli publish --work <output>/work --out <new-release>
python -m data_pipeline.cli package --dataset <output>/release --out <bundle> --mode bundle
python -m data_pipeline.cli package --dataset <output>/release --out <training-bundle> --features-only
```

模型生成的 `[B,T,125]` 规范动作可直接离线执行。初始状态文件须包含与 batch 对应的 `qpos_36`，它提供规范坐标中缺失的世界高度和航向：

```bash
.venv/bin/python -m msc.execute --sample .runs/sample.pt --initial-state .runs/initial.npz --config configs/data_pipeline/omg_text.yaml --output .runs/sonic-execution
```

`--limit` 按序列计数；GDance 一条序列包含多名演员，共享音频只编码一次。
`--workers` 使用 spawn，完成顺序不影响发布顺序。演员失败单独记录，其他合格演员仍可发布。
OMG 的 GPU 配置由父进程中的共享推理线程合并请求；例如 64 个仿真 worker 共用 batch=32 的 SONIC 模型，保留每个 worker 独立的物理状态和历史。GPU 使用 FP32 并禁用 TF32；批量导出副本写入可清理的 cache，原始权重不变。CPU 配置不设置 `inference_batch_size`。

GPU 环境安装 `onnxruntime-gpu` 和 `onnx`，不要同时保留 CPU/GPU 两个 ONNX Runtime 包。当前环境已配置，可直接运行：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m data_pipeline.cli run \
  --config configs/data_pipeline/omg_text.yaml \
  --root /home1/gmh/wangyang/motion/data/OMG-Data \
  --workers 64 --resume --finalize
```

`--resume` 在同一原始数据集、同一输出目录内保留已完成序列和原有版本信息，只处理未完成及处理报错的序列。切换 CPU/GPU 后，旧结果保留真实版本，新计算使用新的 tracker 版本。
`--diagnostic` 可执行被参考质检拒绝的动作供检查，但不会发布该动作。
处理错误或演员错误使 CLI 返回非零状态；普通质检拒绝保留统计。

中断后重跑同一未发布输出会复用内容和配置匹配的缓存，并重新建立当前运行 ledger。
`publish` 仅读取 ledger 和已有数组，不加载任何处理模型，使用原运行的发布策略。
普通 Release 的外部特征引用依赖原缓存目录；移动使用 bundle，可整体迁移后验证。
训练包使用 `--features-only`：动作、特征和时间戳全部放入包内，保留原始媒体的身份、校验和、时长和对齐信息，但不复制 WAV/视频。此时 `assets.uri` 为空，表示未包含原始媒体。
已有 Release 不会被覆盖。不要在同一 output 上同时启动两个 run/publish；缓存内部允许多 worker 并发。

GDance 首次需要单人索引：

```bash
python -m data_pipeline.cli split-gdance --root <raw-gdance> --output <single-person-root>
```

## 对齐、划分和视频扩展

默认 `split_policy: preserve` 保留来源 split，将同源组和共享歌曲跨 split 情况写入 provenance。
`song_disjoint` 按歌曲内容校验和与来源组的连通分量划分，支持 `split_seed`（默认 0）、
`split_ratios`（默认 `[0.8, 0.1, 0.1]`），记录来源 split，避免共享歌曲跨集合。

默认 `alignment_policy: allow_estimated` 接受明确标记的 estimated，跳过 unknown；
`verified_only` 仅发布 verified 关联。所有链接都裁剪到动作与资产的实际交集。
MERT 保持现有 5 秒分块、最后一层 float16 特征和 token 中心时间，不将其当作动作帧。

现有数据集不根据文件名猜测视频同步。需要视频时，在 `source.asset_manifest` 指定补充 Parquet：
每行包含 `sequence_id, motion_id, asset_id, path, modality, time_scale, time_offset_sec,
alignment_status, alignment_source`，可选 `camera_id, track_id`。
路径相对于该清单；同一视频多演员要使用各自的 asset_id 和 track_id。
时间映射约定为 `asset_time = motion_time * time_scale + time_offset_sec`。
视频裁剪后自动扣除裁剪结果对应的原视频时间原点；不自动提升对齐可信度。
启用视频却没有声明视频资产会明确报错。本轮仅裁剪视频，不提取 V-JEPA 特征。

## 测试

```bash
.venv/bin/python -m pytest data_pipeline/tests -q
.venv/bin/python -m pytest tests -q -m 'not integration and not slow'
```

## 训练时在线编码文本

Release 只保存 `meta/annotations.parquet` 中的原始文本，不预计算或打包 T5 token。训练加载器按窗口选择描述，返回 `text` 字符串；Collator 将其组成 batch，并为空缺文本的音频样本保留空字符串。模型使用本地冻结 T5 按 batch 编码，空描述对应的文本 mask 全为 false。

```yaml
model:
  t5_path: model/t5-base-local
```

T5 处于 eval、无梯度模式，文本编码结果只供当前训练前向使用；生成时在 Flow 迭代前编码一次，各步复用同一条件 memory。模型输出维度从本地 T5 配置读取，不生成逐文本磁盘缓存。

## 音频包的 8:1:1 划分

AIST++、FineDance 的配置默认使用 `song_disjoint`、`split_ratios: [0.8, 0.1, 0.1]`、`split_seed: 0`。按共享音频内容及来源分组的连通分量整体分配；AIST++ 额外按原始音乐编号（如 mBR0）关联不同裁剪的同曲片段，数量比例按通过筛选后的序列计，不是按动作版本或窗口计。先将目标数量取整；分组允许时精确分配，无法整除时保留整组，优先避免音频跨 split。

已有包可重分，不重新仿真或提取特征：

```bash
.venv/bin/python -m data_pipeline.cli resplit \
  --dataset data/processed/audio/aistpp/release --ratios 0.8 0.1 0.1
```

重写动作分片和 split 元数据，特征文件复用原始字节；新包校验成功后替换旧包。当前 AIST++ 为 842/105/105，FineDance 为 75/9/9（train/val/test）。

## 与本地 OMG 训练缓存的区别

本地 OMG 的 `src/omg/data/datamodule.py` 当前 materialized 训练路径使用 ConcatDataset 和分片局部打乱，没有固定 text/audio 的 50:50 配比；最终条件出现频率由有效窗口及样本具备的模态决定。其 `EpisodeCachedG1MotionDataset` 读取预先保存的 `qpos_36.npy`、`body_pos_w.npy`、`body_quat_w.npy`，通过 mmap 和分片/episode 缓存取窗口，再按窗口 anchor 做 canonicalization。

MSC 当前离线保存 MERT 特征，T5 在训练时在线编码；训练读取动作时仍按窗口解压 Parquet 并计算 FK，仅有进程内有容量上限的缓存，尚无独立的持久化帧级 FK 训练缓存。因此两者不是同等程度的训练物化。正式训练提速可以另建按 shard 存储、可重建的帧级 qpos/FK mmap 缓存，不宜将大量重叠窗口的完整 125D 标签重复存盘。窗口的 canonical anchor 仍应在取样时统一计算。

128 个真实 35 帧窗口的局部测量中，直接读取并计算 FK 用时约 4.82 秒，预先准备帧级 qpos/body position 后仅复制窗口约 0.0023 秒。这只比较暖文件缓存下的读取/FK阶段，未计条件读取、canonicalization、collation 或模型计算，不能解释成完整训练的加速倍数。

## 训练数据遍历与缓存边界

`train.sh` 使用本地冻结 T5 在线编码。四个 release 通过 `data.roots` 统一读取；`data.tasks: [text, audio]` 仅选择条件类型，不分配 task 权重。训练默认采用 OMG `DistributedPermutationSampler` 的全局伪随机排列算法（SplitMix64、4 轮 Feistel 与 cycle-walking），覆盖全部有效窗口，每轮无放回遍历，不再按块打乱。排列由 seed 与轮次决定，按需计算索引，无需保存全量排列；支持直接从 batch 偏移续训。单卡末尾不完整 batch 保留；多卡为保证每个 rank 的 batch 等长，仅对末尾不足的全局 batch 重复填充。总更新数仍受 `train.steps` 限制。采样单位为窗口，数据集/模态的出现频率由有效窗口数量决定，不是按动作等概率。

`train.sh` 与 `train_overfit.sh` 共用该默认采样器，无需额外开关。新的 checkpoint 元数据记录 `global_permutation` 及算法；此前完成的块式采样实验结果保持原样，不能将其描述为全局随机训练结果。全局伪随机排列是确定性的双射，不等于从所有可能排列中均匀抽取。全局读取可能降低 Parquet 缓存命中率，实际完整训练速度需另行测量。

数据保存职责如下：release 中保存动作 qpos、原始文本、MERT 音频特征及必要索引；本地模型放在 model/；训练输出与统计放在 .runs/。尚未实现的帧级 FK 加速缓存可作为独立可重建派生物，不能代替 release。没有视频特征时不启用视频任务。

A6000 的局部对照：batch=8、历史5帧/未来30帧、动作模型6层/512维、BF16，T5 使用FP32、最多50 token。只测模型前向/反向/AdamW时，GPU 上已有文本 token 约61.3ms/步，在线T5约73.4ms/步，增加约12.1ms（19.7%）。未计入Dataset读取、FK、音频/视频编码，不能视为完整训练或所有模态在线编码的降速比例。
