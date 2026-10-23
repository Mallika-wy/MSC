# MSC Release 通用数据格式说明

本文说明 `data_pipeline` 当前生成的 **所有 release 文件类型**，适用于 OMG、AIST++、FineDance、AIOZ-GDance，以及同一框架下的动作、文本、音频和视频组合。依据 2026-10-23 的代码与四个实际数据包编写。

文中的“真实示例”取自现有 release；“结构示意”只用于说明可选格式。没有实际生成的内容会明确标注。原始数据、处理缓存和最终 release 的范围在第 12 节区分。

## 1. 总目录：必有文件与可选文件

```text
release/
├── dataset.json                       # 必有：格式约定、机器人、路径根目录
├── meta/                              # 必有：六张索引表，即使无记录也保留 schema
│   ├── sequences.parquet              # 序列、来源、train/val/test
│   ├── motions.parquet                # 动作版本、帧数、帧数据位置
│   ├── assets.parquet                 # 媒体资产：音频、视频等
│   ├── links.parquet                  # 动作与媒体的有效区间、时间映射
│   ├── annotations.parquet            # 文本及其有效区间
│   ├── features.parquet               # 特征位置、形状、编码器信息
│   └── provenance.json                # 必有：处理配置、来源、筛选统计
├── training/                          # 已声明 train_default 的训练包自动生成
│   ├── summary.json                   # 索引版本、配置、文件指纹、模态映射
│   ├── train-allow_estimated.arrow     # 紧凑窗口区间索引，按需映射
│   ├── val-allow_estimated.arrow
│   ├── test-allow_estimated.arrow
│   └── *-verified_only.arrow          # 存在 estimated link 时另建严格对齐索引
├── motion/                            # 必有：共享动作帧分片
│   ├── train/shard-00000.parquet ...
│   ├── val/shard-00000.parquet ...      # 仅有该 split 的动作时创建
│   └── test/shard-00000.parquet ...
├── reports/
│   ├── quality-00000.parquet           # 必有：已发布动作的共享质检表
│   └── dataset_summary.json           # 必有：发布/校验汇总
├── features/                          # 可选：打包实际存在的特征时创建
│   ├── audio/shard-00000.parquet ...   # 当前 MERT 音频特征的实际存储方式
│   └── video/shard-00000.parquet ...   # 通用写入器支持；当前未接入视频特征编码器
└── media/                             # 可选：bundle 且 include_media=True
    ├── audio/<内容SHA256>.<原扩展名>    # 如 .wav；保存音频文件本身
    └── video/<内容SHA256>.<原扩展名>    # 如 .mp4；保存视频文件本身
```

这个树包含不同模式下可能出现的文件，单个包不要求全部存在。六张元数据表始终存在；缺失模态对应的是 **0 行表**。`features/`、`media/`、`val/`、`test/` 等目录按实际内容创建。

当前 `run --finalize` 使用 bundle，但不携带原始媒体，即 `include_media=False`：保留动作、已有特征和必要元数据，资产表的 `uri` 为空。因此三个音频包有 MERT 特征，却没有 WAV；OMG 有文本和动作，没有音频特征。

Parquet 是带 schema 的压缩列式表格，使用 PyArrow 等工具读取；不是 JSON 文本。下文以 JSON 展示表中单行，便于阅读。

### training/：训练启动使用的派生索引

六表仍是发布与审计的完整元数据。训练直接映射当前 split 的 Arrow IPC 索引，避免逐次加载全量 Inventory。`validate` / 发布自动构建或刷新索引；`index --dataset <一个或多个 release>` 可独立重建，并在多库之间审计 split。没有 train_default 的参考导出可用 `--label-policy source_g1` 显式指定。

每条索引行表示同一动作、同一条件下连续的可用窗口区间，包含以下列：

| 列 | 内容 |
|---|---|
| motion_id, sequence_id, actor_id, label_type | 已选定训练标签的动作身份 |
| storage_uri, row_start, num_frames | 共享 motion shard 中完整动作的位置 |
| task | text、audio、video 或 motion_only |
| start, end | future 起始帧 k 的可用半开区间 [start,end)，每个窗口读取历史 5 帧与 future 30 帧 |
| text | annotation 原始字符串，仅 text 行非空；无 T5 特征 |
| link_id, feature_id | 音视频映射与共享特征的引用，仅对应模态行非空 |

字符串列采用字典编码，数值列为 int64。`summary.json` 保存版本、选定 label、音视频 context 长度、各 split 文件名/行数/窗口数/SHA256、六表与 manifest 的 SHA256、必要 link/feature 记录和 external_roots。训练只核对版本及配置；文件校验和在发布/validate 时核验。`verified_only` 与默认索引相同的包共用同一文件。Worker 启动时重新映射共享索引，避免序列化百万级 Python 对象。

索引可从六表重建。改变 metadata、标签或 context 后需重新构建；训练不会写入或自动重建 Release。它保存原始文本与位置，没有预计算文本特征。

## 2. 数据之间的关系

```text
sequence（一次来源序列 / 场景）
  ├── motion（人物 A 的参考动作）
  ├── motion（人物 A 的 SONIC 执行动作，parent 指向参考动作）
  ├── motion（人物 B 的动作，如多人数据）
  └── ...

motion ── storage_uri + row_start + num_frames ──→ motion 分片中的帧
motion ── motion_id ──→ annotations（文本）
motion ── motion_id ──→ quality（质检）
motion ── links（时间映射）──→ asset（音频 / 视频）──→ feature（特征矩阵）
```

一条序列可以有多个人、多个动作版本、多段文本及多个媒体资产。一个音频可以被同一序列的多个人、多个动作版本共同引用；特征按资产复用，不为每个人重复存一份。

| 概念 | 一行表示什么 | 是否保存数组本体 |
|---|---|---|
| sequences | 一个来源序列 | 否 |
| motions | 一个具体人物、具体版本的完整动作 | 否，是分片索引 |
| assets | 一个媒体资产 | 否，是媒体描述/路径 |
| links | 一条动作到一个资产的时间关联 | 否 |
| annotations | 一条文本及其有效时间范围 | 文本本身在此 |
| features | 某资产的一份编码特征 | 否，是特征索引 |
| motion shard | 一帧机器人 qpos | 是，每行一帧 |
| feature shard | 一份完整特征及时间戳 | 是，每行一份 feature |
| quality shard | 一条已发布动作的质检报告 | 是，报告保存在 JSON 字符串中 |

## 3. dataset.json：全包约定

下面是当前 AIST++ 包的实际内容，省略机器人校验和，关节列表在第 5 节完整列出：

```json
{
  "package_mode": "bundle",
  "features_bundled": true,
  "quality_storage": "parquet_by_motion_id_v1",
  "feature_storage": "parquet_by_feature_id_v1",
  "motion_storage": "split_shards_v1",
  "schema_version": "1.0",
  "kind": "release",
  "target_motion_fps": 30.0,
  "representation": "g1_qpos36_wxyz_zup_radians",
  "external_roots": {},
  "robot": {
    "name": "g1_29dof",
    "units": "m_rad",
    "up_axis": "z",
    "quaternion": "wxyz"
  }
}
```

| 字段 | 含义 |
|---|---|
| schema_version | 当前元数据格式版本，`1.0` |
| kind | 最终数据包为 `release`；内部清单也支持 `inventory` |
| target_motion_fps | 发布动作统一 30Hz |
| representation | G1 的 qpos36，四元数 wxyz、Z 轴朝上、弧度 |
| motion_storage | 当前 `split_shards_v1`，按 split 共享分片 |
| quality_storage | 当前 `parquet_by_motion_id_v1`，按 motion_id 查共享质量表 |
| feature_storage | 当前 `parquet_by_feature_id_v1`，按 feature_id 查共享特征表 |
| package_mode | `bundle` 独立打包，或 `reference` 引用外部特征/媒体 |
| features_bundled | 是否按可独立携带特征的模式打包；不代表特征表一定非空 |
| external_roots | 外部路径别名到真实目录的映射；当前四个最终包均为 `{}` |
| robot | 机器人名称、单位、坐标约定、关节顺序、运动学及 URDF 的 SHA256 |

### 3.1 路径如何解析

- `motion/train/shard-00000.parquet`：相对 release 根目录。
- `root://local_0/shard-00000.parquet`：通过 `external_roots.local_0` 找到外部根目录，再拼接文件名。
- `null`：未携带该文件，例如最终音频包的原始 WAV。

`root://...` 是本项目的本地路径别名，不是 HTTP 地址。移动独立 bundle 时只需移动整个 release；移动 reference 包时还需保证外部目录可用。

## 4. 六张元数据表

### 4.1 meta/sequences.parquet：来源序列与划分

一行是一条来源序列。所有动作版本和同一场景中的多个人通过 `sequence_id` 归属到此处。

| 字段 | 存储类型 | 含义 |
|---|---|---|
| sequence_id | string | 包内序列唯一 ID，其他表用它关联序列 |
| source_dataset | string | 来源数据集，例如 `omg`、`aistpp`、`finedance`、`gdance` |
| source_sequence_id | string | 源序列编号，例如 `episode-00000000` |
| split_group_id | string | 来源分组标识，用于划分审计，不是动作路径 |
| split | string | `train`、`val` 或 `test` |
| source_provenance | string | JSON 字符串，包含源 episode 元数据、来源路径及文件校验和 |

真实 OMG 第一条记录如下，`source_provenance` 内容仅展示部分字段，并为可读性展开成字典；磁盘上它是 **JSON 字符串**。

```json
{
  "sequence_id": "omg:episode-00000000",
  "source_dataset": "omg",
  "source_sequence_id": "episode-00000000",
  "split_group_id": "100style_train:Aeroplane_BR__seg0000__part0000",
  "split": "train",
  "source_provenance": {
    "source_root": "/home1/gmh/wangyang/motion/data/OMG-Data",
    "upstream_dataset": "100style_train",
    "upstream_sequence": "Aeroplane_BR__seg0000__part0000"
  }
}
```

`split` 存放在序列表，不在动作表。当前 OMG、GDance 保留来源划分；AIST++、FineDance 已按共享音频分组划分为 8:1:1，train/val/test 序列数分别为 842/105/105 和 75/9/9。`split_group_id` 是来源分组标识；具体跨歌曲的划分策略由生成时配置决定。

### 4.2 meta/motions.parquet：动作索引与版本

一行是一条具体动作。字段里的路径均相对于 release，或通过显式外部根别名解析。

| 字段 | 存储类型 | 含义 |
|---|---|---|
| motion_id | string | 动作版本的唯一 ID |
| sequence_id | string | 关联 `sequences.sequence_id` |
| actor_id | string | 序列内的人物标识，例如 `actor0` 或 `person_00` |
| group_id | string | 动作所属来源分组 |
| storage_uri | string | 相对于 release 根目录的动作分片路径 |
| row_start | int64 | 在该分片中的起始行，从 0 开始 |
| num_frames | int64 | 动作帧数 |
| fps | float64 | 当前统一发布帧率 30.0 |
| time_origin_sec | float64 | 该动作局部时间 0 对应的源序列时间，单位秒 |
| label_type | string | `source_g1`、`retarget_reference` 或 `tracker_executed` |
| retarget_version | string | 重定向版本；未重定向时为 `not_applicable` |
| tracker_version | string | 跟踪器版本；执行动作记录 SONIC 标识，原始动作为 `not_applicable` |
| quality_ref | string | 相对于 release 根目录的质量报告文件路径 |
| representation | string | 36 维动作表示的名称 |
| parent_motion_id | string，可为空 | 父动作 ID；SONIC 执行动作指向对应参考版本 |
| label_type | 含义 | 当前常见来源 |
|---|---|---|
| source_g1 | 原始数据已经是 G1 的动作状态 | OMG |
| retarget_reference | 人体动作经 GMR 重定向后的 G1 参考动作 | AIST++、FineDance、GDance |
| tracker_executed | 将参考动作交给 SONIC 仿真跟踪后的实际状态 | 四个数据集 |

真实 AIST++ 同一动作的两个版本：

```json
[
  {
    "motion_id": "aistpp:gBR_sBM_cAll_d04_mBR0_ch01:retarget_reference",
    "sequence_id": "aistpp:gBR_sBM_cAll_d04_mBR0_ch01",
    "actor_id": "actor0",
    "group_id": "gBR_sBM_cAll_d04_mBR0_ch01",
    "storage_uri": "motion/train/shard-00000.parquet",
    "row_start": 0,
    "num_frames": 360,
    "fps": 30.0,
    "time_origin_sec": 0.0,
    "label_type": "retarget_reference",
    "retarget_version": "gmr:reference:9e93f0f7a4db8f63ca0ba50b28a58d65528a8684343646cb54c591465364fafa",
    "tracker_version": "not_applicable",
    "quality_ref": "reports/quality-00000.parquet",
    "representation": "g1_qpos36_wxyz_zup_radians",
    "parent_motion_id": null
  },
  {
    "motion_id": "aistpp:gBR_sBM_cAll_d04_mBR0_ch01:tracker_executed",
    "sequence_id": "aistpp:gBR_sBM_cAll_d04_mBR0_ch01",
    "actor_id": "actor0",
    "group_id": "gBR_sBM_cAll_d04_mBR0_ch01",
    "storage_uri": "motion/train/shard-00000.parquet",
    "row_start": 360,
    "num_frames": 359,
    "fps": 30.0,
    "time_origin_sec": 0.03333333333333333,
    "label_type": "tracker_executed",
    "retarget_version": "gmr:reference:9e93f0f7a4db8f63ca0ba50b28a58d65528a8684343646cb54c591465364fafa",
    "tracker_version": "sonic:c06ce5fbc43aa906f3d8be17cbdab0ca7e2f36783211d7398ffc3668435c8af2",
    "quality_ref": "reports/quality-00000.parquet",
    "representation": "g1_qpos36_wxyz_zup_radians",
    "parent_motion_id": "aistpp:gBR_sBM_cAll_d04_mBR0_ch01:retarget_reference"
  }
]
```

`storage_uri + row_start + num_frames` 唯一确定帧范围；`row_start` 从 0 开始，是 **帧行偏移**，不是字节位置。比如 `row_start=90, num_frames=89` 表示读取分片的 `[90, 179)` 行。

`parent_motion_id` 关联参考版本。当前推荐训练标签为 `tracker_executed`，在 provenance 的 `train_default` 中记录。参考版本和执行版本可能有不同帧数、时间原点，不应假定逐帧长度一致。

### 4.3 meta/assets.parquet：音频与视频资产

资产表描述媒体本身；特征与动作均通过 ID 引用它。当前音频按内容 SHA256 去重复用，所以资产数可能小于序列数。AIST++ 的划分还使用来源音乐编号，将同一首歌的不同裁剪保持在同一个 split。

| 字段 | 存储类型 | 含义 |
|---|---|---|
| asset_id | string | 资产唯一 ID |
| modality | string | `audio` 或 `video` |
| uri | string，可为空 | 原始媒体位置；未随包携带媒体时可为空 |
| source_asset_id | string | 来源资产标识 |
| checksum | string | 原始资产内容校验和 |
| size_bytes | int64 | 原始文件大小 |
| license_ref | string | 来源使用条款说明 |
| duration_sec | float64，可为空 | 时长，秒 |
| sample_rate_hz | float64，可为空 | 音频采样率 |
| frame_rate | float64，可为空 | 视频帧率 |
| camera_id | string，可为空 | 相机标识 |
| person_track_id | string，可为空 | 视频人物轨迹标识 |

真实 AIST++ 音频资产：

```json
{
  "asset_id": "audio:37054864e47f540589e6f4ad54f4dae07f238f3ced8766a745efa9eddbc04456",
  "modality": "audio",
  "uri": null,
  "source_asset_id": "audio:37054864e47f540589e6f4ad54f4dae07f238f3ced8766a745efa9eddbc04456",
  "checksum": "37054864e47f540589e6f4ad54f4dae07f238f3ced8766a745efa9eddbc04456",
  "size_bytes": 2302030,
  "license_ref": "aistpp upstream terms; source media not redistributed",
  "duration_sec": 11.989333333333333,
  "sample_rate_hz": 48000.0,
  "frame_rate": null,
  "camera_id": null,
  "person_track_id": null
}
```

此处 `sample_rate_hz: 48000` 描述原始 WAV，不是 MERT 的 75Hz token 频率，也不是动作的 30Hz 帧率。`uri: null` 表示没有携带 WAV，但通过 `asset_id` 仍能找到已打包的音频特征。

视频使用相同表，`modality=video`，填写 `frame_rate`、`camera_id`、`person_track_id` 等适用字段。当前 pipeline 启用视频预处理时，资产对应处理后的 512×512 人物裁剪视频；它不是 SONIC 机器人回放视频。媒体未被打包时 uri 同样可为空。

文本主要存放在 `annotations`；虽然资产校验契约允许 `modality=text`，当前文本流程不额外生成 text 资产文件。

### 4.4 meta/links.parquet：动作与媒体怎样对齐

同一动作可以关联音频和不同视角视频；一条关联保存一个资产的有效范围与映射。

| 字段 | 存储类型 | 含义 |
|---|---|---|
| link_id | string | 关联唯一 ID |
| motion_id | string | 对应动作 |
| asset_id | string | 对应媒体资产 |
| modality | string | 媒体类型 |
| motion_valid_start_sec | float64 | 动作局部时间的有效起点，包含 |
| motion_valid_end_sec | float64 | 动作局部时间的有效终点，不包含 |
| time_scale | float64 | 动作时间到媒体时间的缩放 |
| time_offset_sec | float64 | 动作时间到媒体时间的偏移，秒 |
| alignment_status | string | 对齐状态，例如 `verified`、`estimated` |
| alignment_source | string | 对齐依据 |
| view_id | string，可为空 | 视角标识 |
| actor_id | string，可为空 | 人物标识 |

真实 AIST++ 第一条 SONIC 执行动作关联：

```json
{
  "link_id": "aistpp:gBR_sBM_cAll_d04_mBR0_ch01:tracker_executed:8f4fd0f5ae508d4b",
  "motion_id": "aistpp:gBR_sBM_cAll_d04_mBR0_ch01:tracker_executed",
  "asset_id": "audio:37054864e47f540589e6f4ad54f4dae07f238f3ced8766a745efa9eddbc04456",
  "modality": "audio",
  "motion_valid_start_sec": 0.0,
  "motion_valid_end_sec": 11.956,
  "time_scale": 1.0,
  "time_offset_sec": 0.03333333333333333,
  "alignment_status": "estimated",
  "alignment_source": "matched source excerpt; offset not independently calibrated",
  "view_id": null,
  "actor_id": "actor0"
}
```

时间约定：

```python
asset_time_sec = time_scale * motion_local_time_sec + time_offset_sec
```

有效区间为 `[motion_valid_start_sec, motion_valid_end_sec)`，起点包含、终点不包含。`actor_id` 表明对应人物；视频可同时记录 `view_id`。

`alignment_status` 可为 `verified`、`estimated`、`unknown`。当前发布器不发布 unknown 关联；若配置 `alignment_policy=verified_only`，也会排除 estimated。现有三个音频包的配对关系标为 `estimated`，表示来源片段匹配，偏移未独立标定，不应解释成严格标定的同步数据。

### 4.5 meta/annotations.parquet：文本描述

一行是一段文本及适用区间；序列可有多段标注。

| 字段 | 存储类型 | 含义 |
|---|---|---|
| annotation_id | string | 标注唯一 ID |
| sequence_id | string | 所属序列 |
| text | string | 文本描述 |
| language | string | 语言代码，例如 `en` |
| start_sec | float64 | 对应动作局部时间中的起点，包含 |
| end_sec | float64 | 对应动作局部时间中的终点，不包含 |
| source | string | 文本来源，例如 `OMG episode tasks` |
| confidence | float64 | 标注置信值；当前 OMG 的 1.0 是写入值，不是模型评分 |
| motion_id | string，可为空 | 对应动作 ID；当前发布器绑定具体动作版本 |

真实 OMG 的 SONIC 执行动作文本：

```json
{
  "annotation_id": "omg:episode-00000000:tracker_executed:text:0",
  "sequence_id": "omg:episode-00000000",
  "text": "raise arms to sides and initiate body rotation; style: smooth",
  "language": "en",
  "start_sec": 0.0,
  "end_sec": 2.966666666666667,
  "source": "OMG episode tasks",
  "confidence": 1.0,
  "motion_id": "omg:episode-00000000:tracker_executed"
}
```

时间区间使用对应动作的局部时间。`motion_id` 可为空的契约允许序列级标注，但当前发布器生成的标注绑定具体动作。音频包没有文本时，这张表是 0 行，而不是额外生成空 TXT 文件。

### 4.6 meta/features.parquet：特征索引

这里保存特征描述与位置。实际矩阵存放在 `features/<modality>/shard-*.parquet`。

| 字段 | 存储类型 | 含义 |
|---|---|---|
| feature_id | string | 特征唯一 ID |
| asset_id | string | 对应资产 |
| encoder_name | string | 特征编码器名称 |
| encoder_checkpoint_hash | string | 编码器权重标识 |
| preprocess_hash | string | 预处理配置标识 |
| feature_uri | string | 特征分片路径 |
| timestamp_uri | string | 时间戳路径；共享分片时通常与 feature_uri 相同 |
| dtype | string | 特征数值类型，例如 `float16` |
| shape | list<int64> | 原始特征矩阵形状 |
| token_time_semantics | string | token 时间语义，例如 `point` |
| aggregation_method | string | 特征选层或聚合方式 |
| extraction_version | string | 特征提取流程版本 |
| input_asset_checksum | string | 对应输入媒体的校验和 |
| token_rate_hz | float64，可为空 | 特征 token 频率 |

真实 AIST++ 第一份 MERT 特征索引：

```json
{
  "feature_id": "audio:37054864e47f540589e6f4ad54f4dae07f238f3ced8766a745efa9eddbc04456:9fe5e40b893a7422ab2a6ba86ecf2607ab7f3a83a05898d1e89c99b4624e0c3c",
  "asset_id": "audio:37054864e47f540589e6f4ad54f4dae07f238f3ced8766a745efa9eddbc04456",
  "encoder_name": "m-a-p/MERT-v1-95M",
  "encoder_checkpoint_hash": "a2b8b747f72c06e0595aeae41ae5473f4364938c6b39b2c58be38c48e6bd3fcd",
  "preprocess_hash": "610ce91fd5505448c3e59a0c3bc5cf1a115124d8c9bbc1d758cbbf3ce2d44b80",
  "feature_uri": "features/audio/shard-00000.parquet",
  "timestamp_uri": "features/audio/shard-00000.parquet",
  "dtype": "float16",
  "shape": [
    896,
    768
  ],
  "token_time_semantics": "point",
  "aggregation_method": "last_transformer_layer",
  "extraction_version": "mert_chunked_v1",
  "input_asset_checksum": "37054864e47f540589e6f4ad54f4dae07f238f3ced8766a745efa9eddbc04456",
  "token_rate_hz": 75.0
}
```

该记录说明：特征来自 MERT-v1-95M，最后一层隐藏状态，原始矩阵形状为 `[896, 768]`，FP16。`feature_uri` 和 `timestamp_uri` 相同，因为矩阵和时间戳都在同一个共享分片的同一行。

同一资产如果使用不同编码器或预处理，可以有不同 feature_id。不能只按文件名判断特征身份；应通过 asset_id 选择需要的编码器版本，再按 feature_id 读取。

## 5. motion 分片：每帧是什么样的

所有动作版本均采用同一种 schema：

| 列 | Arrow 类型 | 一行内容 |
|---|---|---|
| qpos36 | fixed_size_list<float32>[36] | 一帧 G1 的根位置、根姿态及 29 个关节角 |
| timestamp_sec | float64 | 所属动作的局部时间，秒 |

真实 OMG 第一帧，数值保留 6 位小数：

```json
{
  "qpos36": [
    0.0,
    0.0,
    0.789987,
    0.997659,
    0.014762,
    -0.022059,
    -0.063024,
    -0.123104,
    0.104521,
    0.020546,
    0.432828,
    -0.231269,
    -0.098258,
    0.139099,
    -0.066978,
    -0.302464,
    0.071361,
    -0.175339,
    -0.012541,
    0.004745,
    0.139947,
    0.195591,
    -0.014102,
    0.020392,
    -0.299045,
    1.372084,
    0.062627,
    0.121441,
    -0.114837,
    -0.223114,
    -0.27217,
    0.37607,
    1.43411,
    -0.509527,
    -0.051105,
    -0.639722
  ],
  "timestamp_sec": 0.0
}
```

```python
root_xyz = qpos[:, 0:3]       # 世界坐标 x/y/z，米，Z 轴朝上
root_wxyz = qpos[:, 3:7]      # 根姿态，单位四元数，w/x/y/z
joint_angles = qpos[:, 7:36]  # 29 个关节角，弧度
```

36 维表示位置状态，不含关节速度、力矩、接触标签或控制网络输入。矩阵读取后为 `[T, 36]` 的 float32 数组。

| qpos 列索引，从 0 开始 | 关节名称 |
|---:|---|
| 7 | `left_hip_pitch_joint` |
| 8 | `left_hip_roll_joint` |
| 9 | `left_hip_yaw_joint` |
| 10 | `left_knee_joint` |
| 11 | `left_ankle_pitch_joint` |
| 12 | `left_ankle_roll_joint` |
| 13 | `right_hip_pitch_joint` |
| 14 | `right_hip_roll_joint` |
| 15 | `right_hip_yaw_joint` |
| 16 | `right_knee_joint` |
| 17 | `right_ankle_pitch_joint` |
| 18 | `right_ankle_roll_joint` |
| 19 | `waist_yaw_joint` |
| 20 | `waist_roll_joint` |
| 21 | `waist_pitch_joint` |
| 22 | `left_shoulder_pitch_joint` |
| 23 | `left_shoulder_roll_joint` |
| 24 | `left_shoulder_yaw_joint` |
| 25 | `left_elbow_joint` |
| 26 | `left_wrist_roll_joint` |
| 27 | `left_wrist_pitch_joint` |
| 28 | `left_wrist_yaw_joint` |
| 29 | `right_shoulder_pitch_joint` |
| 30 | `right_shoulder_roll_joint` |
| 31 | `right_shoulder_yaw_joint` |
| 32 | `right_elbow_joint` |
| 33 | `right_wrist_roll_joint` |
| 34 | `right_wrist_pitch_joint` |
| 35 | `right_wrist_yaw_joint` |

多个动作首尾拼接到一个分片，没有逐动作文件，也没有逐帧 motion_id 列。动作边界由 `motions.parquet` 定义。各段时间戳均从 0 开始，因此整个分片的 timestamp 不全局递增。

当前目标为每个动作分片最多约 1,000,000 帧，整条动作写入一个 shard。一个动作可能跨 Parquet 内部 row group；row group 不等于动作边界。

### 5.1 动作时间原点和帧率

动作时间戳为 `np.arange(T) / 30`。最后一帧时刻是 `(T-1)/30`，而帧覆盖时长按 `T/30` 计算。

```python
source_time_sec = motion_local_time_sec + motion.time_origin_sec
```

当前 SONIC 控制与筛选使用 50Hz，release 动作统一为 30Hz。执行结果排除初始状态，第一帧通常对应来源时间 `1/30` 秒；以记录中的 `time_origin_sec` 和 `num_frames` 为准。

## 6. features 分片：音频特征的实际内容

与动作分片的“每行一帧”不同，特征分片是 **每行一份完整特征**，其 schema 为：

| 列 | Arrow 类型 | 内容 |
|---|---|---|
| feature_id | string | 与 meta/features.parquet 对应的 ID |
| shape | list<int64> | values 还原后的数组形状 |
| dtype | string | 当前为 `float16` |
| values | large_list<float16> | 展平的特征数值；物理存储也是 FP16 |
| timestamps | list<float64> | 展平的 token 时间戳 |
| timestamp_shape | list<int64> | 时间戳还原后的形状 |

真实 AIST++ 特征行，`values` 和 `timestamps` **只展示前 8 个元素**：

```json
{
  "feature_id": "audio:37054864e47f540589e6f4ad54f4dae07f238f3ced8766a745efa9eddbc04456:9fe5e40b893a7422ab2a6ba86ecf2607ab7f3a83a05898d1e89c99b4624e0c3c",
  "shape": [
    896,
    768
  ],
  "dtype": "float16",
  "values": [
    0.0192718505859375,
    -0.080322265625,
    0.08404541015625,
    -0.057769775390625,
    0.27685546875,
    -0.41064453125,
    -0.24072265625,
    0.287353515625
  ],
  "timestamps": [
    0.0083125,
    0.021645833333333333,
    0.034979166666666665,
    0.0483125,
    0.06164583333333333,
    0.07497916666666667,
    0.0883125,
    0.10164583333333334
  ],
  "timestamp_shape": [
    896
  ]
}
```

该完整记录有 `896×768=688,128` 个 values 和 896 个时间戳。还原方法：

```python
values = flat_values.reshape(shape)                 # [896, 768]
timestamps = flat_timestamps.reshape(timestamp_shape)  # [896]
```

当前 writer 按模态分别写 shard，以未压缩数组合计约 256MiB 为目标轮换。完整 feature 不拆成多个记录，压缩后的文件大小不必等于这个目标。

### 6.1 音频时间戳与 75Hz

当前 MERT 将输入音频混为单声道、重采样到模型采样率，按配置分块编码，保存最后一层 FP16 特征。现有模型使用 24kHz 输入，token 名义频率为 75Hz；第一个 token 中心在约 0.0083125 秒。

时间戳使用 **资产的时间轴**，不是某条动作的局部时间轴。由于分块和尾部处理，不应通过 `token_index / 75` 重建整个时间轴，也不要假设每个 30Hz 动作帧恰好对应固定整数个 token。读取时使用已保存的 timestamps，再通过 links 映射动作区间。

`token_time_semantics=point` 表示每个 token 的一个时间点，对应时间戳 `[N]`；契约也支持 `interval` 区间语义，对应 `[N, 2]`，筛选时按区间相交判断。当前 MERT 生成 point。

### 6.2 视频特征支持到什么程度

共享特征写入器按 modality 分目录，因此具备写 `features/video/` 的能力，但当前视频流程只处理人物裁剪，没有接入视频特征编码器；现有四包均无视频特征。不要把目录示意理解为已经生成了视频 embedding，也不要将视频特征的维度和频率直接假定为 MERT 的 768/75Hz。

### 6.3 文本采用在线编码

文本直接保存在 `meta/annotations.parquet`，Release 不保存 T5 特征分片，也不需要 `features/text/`。训练时按窗口取原始描述，由本地冻结 T5 按 batch 在线编码；文本 token 仅在当前计算中使用。生成时在迭代开始前编码一次并复用条件。

MERT 音频特征继续离线保存为 FP16。音频波形预处理与长音频编码成本较高，与文本采用不同的准备方式。

## 7. media：音频和视频文件本身

当使用 `package --mode bundle` 且不加 `--features-only` 时，打包器复制已有媒体资产到：

```text
media/audio/<SHA256>.wav
media/video/<SHA256>.mp4
```

文件名使用内容 SHA256，扩展名保留来源文件扩展名；WAV/MP4 是常见示例，不是额外的固定命名要求。`assets.uri` 指向这些路径。

音频文件内部是波形采样：采样率、声道数、PCM 位深从 WAV 头读取，不能从特征维度推断。资产表记录 duration 和 sample_rate；MERT 单声道处理不意味着媒体文件本身变成单声道。

当前视频预处理生成 512×512 人物裁剪 MP4，资产表记录对应的 frame_rate 和人物标识。打包器复制视频本身，不自动打包裁剪组件旁边的 metadata.npz、settings.json、detections.json 或 quality.json；这些属于处理中间产物，详见第 12 节。

**当前四个 finalize 包都不携带媒体**，所以没有 media/，assets.uri 为 null。特征可以正常读取，若要回听音频或播放原视频则需另行取得媒体。

## 8. reports：质量报告和统计

### 8.1 quality-00000.parquet

当前质量写入器生成一个共享文件，所有已发布动作均在其中有一条报告。

| 列 | Arrow 类型 | 内容 |
|---|---|---|
| motion_id | string | 对应动作 |
| accepted | bool | 是否接受；最终发布记录为 true |
| report_json | string | 详细报告的 JSON 字符串 |

真实 OMG SONIC 报告如下，已将 report_json 展开成字典：

```json
{
  "motion_id": "omg:episode-00000000:tracker_executed",
  "accepted": true,
  "report_json": {
    "accepted": true,
    "rejection_reasons": [],
    "screening": {
      "frames": 149,
      "screening_fps": 50.0,
      "thresholds": {
        "min_root_height_m": 0.2,
        "max_root_tilt_deg": 85.0,
        "low_root_height_m": 0.35,
        "low_root_tilt_deg": 60.0,
        "consecutive_fall_frames": 10,
        "strict_joint_limits": false,
        "joint_limit_tolerance_rad": 0.0001
      },
      "accepted": true,
      "rejection_reasons": [],
      "root_height_min_m": 0.7435262203216553,
      "root_height_max_m": 0.787953794002533,
      "root_tilt_max_deg": 6.866276077926073,
      "fall_frames": 0,
      "longest_fall_run_frames": 0,
      "longest_fall_run_s": 0.0,
      "joint_limit_violation_values": 0,
      "joint_limit_violation_frames": 0,
      "joint_limit_max_violation_rad": 0.0,
      "quaternion_max_norm_error": 3.513853430447966e-08,
      "max_joint_speed_rad_s": 6.068216264247894
    },
    "tracking": {
      "joint_rmse_rad": 0.10449829697608948,
      "global_body_mpjpe_mm": 206.96739852428436
    },
    "production_ready": false,
    "diagnostic_only": false,
    "motion_id": "omg:episode-00000000:tracker_executed"
  }
}
```

报告的内部字段按阶段变化，不要求所有 label_type 的 JSON 完全一样：

| 字段/分组 | 含义 |
|---|---|
| motion_id / accepted | 动作身份及接受状态 |
| stage / processing | 原始或参考动作的处理阶段信息，按阶段存在 |
| rejection_reasons | 本阶段的拒绝原因列表 |
| screening | 筛选帧率、阈值、根高度、倾斜、倒地持续帧、关节越界等 |
| tracking | 与参考动作的误差指标，例如 joint_rmse_rad、global_body_mpjpe_mm |
| diagnostic_only | 是否为诊断执行结果，按报告存在 |
| production_ready | 当前流程写入的状态标记，不等同于 accepted |

参考动作的报告还可能包含重定向或手臂扭曲指标，应查看该报告实际键名；不要假设它拥有 SONIC 的 tracking 内容。`accepted=true` 表示通过当前筛选，不保证视觉动作完美。最终包主要保留通过的动作，拒绝动作的逐条报告不自动全部纳入其中。

### 8.2 dataset_summary.json

当前 AIST++ 的实际统计：

```json
{
  "valid": true,
  "processing_summary": {
    "accepted": 1052,
    "motion_accepted": 1052,
    "rejected": 355,
    "motion_rejected": 355,
    "error": 1,
    "motion_error": 1
  },
  "rejection_reasons": {
    "arm_twist_reference": 244,
    "persistent_fall": 191
  },
  "sequences": 1052,
  "motions": 2104,
  "assets": 100,
  "features": 100,
  "duration_sec": 23655.00000000021,
  "sources": {
    "aistpp": 1052
  },
  "modality_combinations": {
    "audio": 2104
  },
  "deep_checked_motions": 2104,
  "deep_checked_features": 100
}
```

| 字段 | 含义 |
|---|---|
| valid | 这次校验是否通过 |
| sequences / motions / assets / features | 四类记录数量 |
| processing_summary | accepted、partial、rejected、error 等处理计数；具体键按运行存在 |
| rejection_reasons | 拒绝原因及数量汇总 |
| duration_sec | 所有动作记录的 `num_frames/fps` 之和 |
| sources | 按来源数据集统计序列数量 |
| modality_combinations | 按动作实际关联的模态组合统计 |
| deep_checked_motions / deep_checked_features | 实际深入检查数值、时间戳的记录数量 |

同一来源动作有参考和执行两个版本时，duration_sec 会累加两份，不能当作去重后的独立采集总时长。不能只凭 valid 判断是否全量深入检查，还应查看 deep_checked_* 与记录总数。

package 命令的返回值还可能带 mode、portable、copied_files，具体以实际输出为准；这些不是每份 summary 都必有的字段。

## 9. meta/provenance.json：处理策略与来源

这是独立 JSON。以下为当前 FineDance 的实际内容：

```json
{
  "pipeline": "dataset_centric_v3",
  "release_policy": {
    "output": "../../data/processed/audio/finedance",
    "publish_labels": [
      "retarget_reference",
      "tracker_executed"
    ],
    "train_default": "tracker_executed",
    "max_shard_frames": 1000000,
    "split_policy": "preserve",
    "alignment_policy": "allow_estimated"
  },
  "train_default": "tracker_executed",
  "quality_storage": "parquet_by_motion_id_v1",
  "feature_storage": "parquet_by_feature_id_v1",
  "split_audit": [],
  "processing_summary": {
    "accepted": 93,
    "motion_accepted": 93,
    "rejected": 110,
    "motion_rejected": 110
  },
  "rejection_reasons": {
    "arm_twist_reference": 80,
    "persistent_fall": 30
  },
  "package_mode": "bundle",
  "portable": true,
  "media_bundled": false
}
```

主要用途：查看处理管线版本、发布哪些 label、推荐训练标签、分片目标、split 策略、对齐策略、处理数量、拒绝原因及打包模式。

- `release_policy.publish_labels`：允许发布的动作版本。
- `train_default`：推荐训练版本，当前通常为 tracker_executed。
- `split_audit`：来源分组/共享音频的划分审计记录；为空表示没有记录到冲突。
- `portable`、`package_mode`：是否独立打包。
- `media_bundled`：是否按携带媒体的模式打包，不表示一定有某种模态。

来源绝对路径、生成时 output 配置和校验和用于溯源，不是独立 bundle 的读取依赖。详细逐序列来源放在 sequences.source_provenance 中。

## 10. 动作、音频特征和文本怎样一起读取

### 10.1 动作 + 文本

在 MSC 根目录，用项目 `.venv/bin/python` 执行：

```python
from pathlib import Path
import pyarrow.parquet as pq
from data_pipeline.schema.records import MotionRecord
from data_pipeline.release.export import read_motion

root = Path('data/processed/omg-text/release')
motion_id = 'omg:episode-00000000:tracker_executed'
row = pq.read_table(root / 'meta/motions.parquet',
                    filters=[('motion_id', '=', motion_id)]).to_pylist()[0]
motion = MotionRecord(**row)
qpos, motion_times = read_motion(root, motion)
texts = pq.read_table(root / 'meta/annotations.parquet',
                      filters=[('motion_id', '=', motion_id)]).to_pylist()
print(qpos.shape, qpos.dtype)  # (89, 36) float32
print(texts[0]['text'])
```

### 10.2 动作 + 对应区间的音频特征

下例从实际 AIST++ 包选择第一条 SONIC 动作，找对应音频，再读取与动作有效区间相匹配的 token。代码会同时处理动作到音频的时间偏移。

```python
from pathlib import Path
import json
import pyarrow.parquet as pq
from data_pipeline.schema.records import MotionRecord, FeatureRecord
from data_pipeline.schema.time_map import TimeMap
from data_pipeline.release.export import read_motion
from data_pipeline.release.features import read_feature

root = Path('data/processed/audio/aistpp/release')
manifest = json.loads((root / 'dataset.json').read_text())

# 从索引中流式寻找第一条执行动作。
found = None
for batch in pq.ParquetFile(root / 'meta/motions.parquet').iter_batches(batch_size=128):
    found = next((r for r in batch.to_pylist() if r['label_type'] == 'tracker_executed'), None)
    if found is not None:
        break
motion = MotionRecord(**found)
qpos, motion_times = read_motion(root, motion)

links = pq.read_table(root / 'meta/links.parquet', filters=[
    ('motion_id', '=', motion.motion_id), ('modality', '=', 'audio'),
]).to_pylist()
link = links[0]  # 示例选第一个音频关联；实际可按需要遍历。
feature_rows = pq.read_table(root / 'meta/features.parquet',
    filters=[('asset_id', '=', link['asset_id'])]).to_pylist()
feature = FeatureRecord(**feature_rows[0])  # 有多个编码器时应显式选版本。
values, audio_times = read_feature(root, feature, manifest['external_roots'])

mapping = TimeMap(link['time_scale'], link['time_offset_sec'])
start = link['motion_valid_start_sec']
end = link['motion_valid_end_sec']
audio_mask = mapping.select(audio_times, start, end,
                            semantics=feature.token_time_semantics)
motion_mask = (motion_times >= start) & (motion_times < end)
qpos_valid = qpos[motion_mask]
audio_values_valid = values[audio_mask]
audio_times_on_motion = mapping.to_motion(audio_times[audio_mask])

print(qpos.shape, qpos.dtype)
print(values.shape, values.dtype)  # 第一份特征为 (896, 768) float16
print(qpos_valid.shape, audio_values_valid.shape)
```

输出的动作帧数与 token 数通常不同，这是 30Hz 与约 75Hz 的正常差异。需要逐帧融合时，应明确选取插值、池化或注意力等策略；release 不预先强制两者长度相等。

### 10.3 只查看某个文件的 schema 和前两行

```python
from pathlib import Path
import pyarrow.parquet as pq

root = Path('data/processed/audio/aistpp/release')
file = pq.ParquetFile(root / 'meta/features.parquet')
print(file.schema_arrow)
print('行数:', file.metadata.num_rows)
for batch in file.iter_batches(batch_size=2):
    print(batch.to_pylist())
    break
```

查看特征本体时不要直接打印完整 values，单行就可能包含数十万至数百万元素；只打印 shape 和少量数值即可。批量训练应预建索引并按 shard 组织读取，以上过滤查询适合说明格式和单条查看。

## 11. 当前四个 release 的实际区别

以下数量直接从当前文件的 Parquet 元数据统计，不是格式的固定上限。

| 数据集 | 序列 | 动作记录 | 文本标注 | 媒体资产 | 关联 | 特征 |
|---|---:|---:|---:|---:|---:|---:|
| omg | 731,673 | 1,463,346 | 1,463,346 | 0 | 0 | 0 |
| aistpp | 1,052 | 2,104 | 0 | 100 | 2,104 | 100 |
| finedance | 93 | 186 | 0 | 91 | 186 | 91 |
| gdance | 1,608 | 11,474 | 0 | 1,608 | 11,474 | 1,608 |
| 内容 | OMG | AIST++ / FineDance / GDance |
|---|---|---|
| 参考动作类型 | source_g1 | retarget_reference |
| SONIC 执行动作 | 有 | 有 |
| 英文文本 | 有 | 当前无 |
| MERT 音频特征 | 无 | 有，FP16、768 维、名义 75Hz |
| 原始媒体文件 | 未打包 | 未打包 |
| 视频 / 视频特征 | 当前无 | 当前无 |
| split | train / val / test | 三个数据集都已有 train / val / test |

目录位置：

```text
data/processed/omg-text/release
data/processed/audio/aistpp/release
data/processed/audio/finedance/release
data/processed/audio/gdance/release
```

## 12. 哪些文件不属于最终 release

### 12.1 release 的同级文件

`config.json`、`summary.json` 可能保留在处理输出目录，位于 release 外；它们是该次运行的配置和总结果，不是这份格式要求的训练内容。

`work/`、`cache/`、SQLite、锁文件、运行日志、临时 `.release-bundle/` 和 `.release-reference/` 属于运行/恢复/打包过程。finalize 成功后删除 work、cache 及打包旧副本；原始数据目录不会因此改写。

### 12.2 人物视频组件的中间文件

启用人物视频处理时，缓存目录可能有以下文件，它们不会自动成为最终 release 的独立文件：

| 文件 | 内容 |
|---|---|
| person.mp4 | 512×512 人物裁剪视频；可作为媒体资产被复制进 media/video |
| metadata.npz | 输出时间、源 PTS、取帧索引、裁剪框、正逆仿射矩阵、检测掩码与置信度等 NumPy 数组 |
| settings.json | 裁剪、编码及输入参数 |
| detections.json | 人物检测/身份轨迹缓存及检测设置 |
| quality.json | 检测覆盖率、平均置信度、采样误差等裁剪质量信息 |

metadata.npz 的当前键：`output_timestamps`、`source_requested_pts`、`source_frame_indices`、`source_pts`、`crop_xyxy`、`original_to_crop_affine`、`crop_to_original_affine`、`detection_valid`、`confidence`、`track_id`、`source_size`、`output_size`、`fps`。当前 release 表没有这些裁剪矩阵的独立索引；如果后续训练需要映射回原图坐标，需要额外设计其发布方式。

SONIC 诊断回放、原始 SMPL/SMPL-X、模型权重也不属于当前 release 的必备文件。reference 的 G1 qpos 已经是发布后的动作表示，不是人体模型参数。

## 13. reference 包及读取器兼容的其他形态

当前四个 finalize 包均使用上文的共享 Parquet bundle。以下是代码仍支持的其他形态，读取时以 manifest 和记录中的 URI 为准：

| 形态 | 文件/内容 | 如何定位 |
|---|---|---|
| reference 包 | 动作和质量文件复制进包内；特征、媒体可以保持外部引用 | external_roots + root://alias/path |
| NPY 特征读取兼容 | 特征矩阵 .npy 与时间戳 .npy，矩阵的 shape/dtype 在元数据中记录 | feature_uri 与 timestamp_uri 分别指向数组 |
| JSON 质量读取兼容 | 一个 JSON 对象的质量报告 | quality_ref 指向对应 JSON |

当前特征 writer 不再把 MERT FP16 扩成 FP32。NPY/JSON 兼容读取能力也不代表当前 finalize 会重新生成逐样本小文件。路径布局应按当前包实际 URI 读取，不要将某个示例文件名或具体 shard 编号写死。

实现依据：`data_pipeline/schema/records.py`、`schema/manifest.py`、`release/export.py`、`release/features.py`、`release/package.py`、`release/quality.py`、`audio/mert.py` 和 `pipeline.py`。OMG 的更具体记录可参阅 [OMG Release 实例说明](OMG_Release数据格式说明.md)。
