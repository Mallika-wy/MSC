# OMG Release 数据格式说明

本文对应实际目录 `/home1/gmh/wangyang/motion/MSC/data/processed/omg-text/release`，按 2026-10-23 的已生成数据编写。下文标注“真实记录”的内容来自该目录；省略或四舍五入会单独说明。

> 文本以原始描述保存在 annotations.parquet，训练时使用本地冻结 T5 在线编码；Release 不保存 T5 特征缓存。通用格式见 [Release 通用说明](Release数据格式说明.md)。

## 1. 这个包保存的是什么

这是一个独立的 Motion + Text 数据包，保存 G1 机器人的动作、对应的英文描述、动作索引和处理信息。读取包内数据不依赖原始 OMG 目录。

它采用“索引表 + 共享动作分片”的结构：一条动作在 `motions.parquet` 中占一行，但动作的所有帧存放在共享 shard 中。不会为每条动作单独建立文件。

| 内容 | 当前数量 |
|---|---:|
| 序列 sequence | 731,673 |
| 动作记录 motion | 1,463,346 |
| 文本标注 annotation | 1,463,346 |
| 音频 / 视频资产 asset | 0 |
| 音视频关联 link | 0 |
| 音视频特征 feature | 0 |
| 动作分片文件 | 209 |

一个序列有两个动作版本：`source_g1` 是 OMG 输入的原始 G1 动作，`tracker_executed` 是 SONIC 在仿真中跟踪执行后的动作。当前推荐的训练版本是 `tracker_executed`，记录在 `meta/provenance.json` 的 `train_default` 中。两份动作不是两条独立采集的样本；划分数据或统计独立样本数时应按序列理解。

## 2. 目录中都有什么

```text
release/
├── dataset.json                      # 格式版本、机器人定义、关节顺序
├── meta/
│   ├── sequences.parquet             # 序列索引，包含 train/val/test
│   ├── motions.parquet               # 动作索引，定位到 shard 的行范围
│   ├── annotations.parquet           # 文本及对应动作、有效时间段
│   ├── assets.parquet                # 音视频资产索引；本包为空表
│   ├── links.parquet                 # 动作与音视频的时间关系；本包为空表
│   ├── features.parquet              # 音视频特征索引；本包为空表
│   └── provenance.json               # 处理配置、来源及质量筛选汇总
├── motion/
│   ├── train/shard-00000.parquet ...  # 166 个分片
│   ├── val/shard-00000.parquet ...    # 21 个分片
│   └── test/shard-00000.parquet ...   # 22 个分片
└── reports/
    ├── quality-00000.parquet          # 每条已发布动作的质量报告
    └── dataset_summary.json          # 数量及最终校验汇总
```

目前没有 `features/` 目录，也没有音频、视频、回放 MP4、ONNX 模型或原始人体参数文件。空表保留了统一的字段结构，便于与音频数据集使用同一种 release 格式。

整个目录共 219 个文件，文件大小合计 36,045,206,167 字节，约 33.57 GiB。Parquet 是压缩的列式二进制表格，不能当普通文本直接打开；下面的 JSON 是表格记录转成字典后的展示形式。

## 3. dataset.json：全包的数据约定

这是普通 JSON。下面是实际内容的主要字段；完整关节顺序在第 6 节列出。

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
    "quaternion": "wxyz",
    "joint_order": "见第 6 节，原文件为 29 个关节名称组成的列表",
    "kinematics_sha256": "4699efa6f6b2dca3bd32edc9ab43eac19ee7268beb1ee95e800dbd031fe32d90",
    "urdf_sha256": "e1dc89366bf96aa33ba4ef211884f6d81320ddd2ac6f8b1ad82986e64008be31"
  }
}
```

`package_mode: bundle` 和 `external_roots: {}` 表示动作及已有特征从包内解析，不需要外部路径。`features_bundled: true` 表示打包模式允许特征随包携带，不表示这个 OMG 包实际含有特征；实际数量要看 `features.parquet`。

`target_motion_fps` 是发布帧率 30Hz。`representation` 表示每帧采用 G1 的 36 维位置状态，根四元数为 wxyz，Z 轴向上，关节角为弧度。`robot` 中的两个 SHA256 用于标识机器人运动学和 URDF 版本。

## 4. meta/sequences.parquet：序列索引

一行表示一个原始 episode；当前共 731,673 行。

| 字段 | 存储类型 | 含义 |
|---|---|---|
| sequence_id | string | 包内序列唯一 ID，其他表用它关联序列 |
| source_dataset | string | 来源数据集，本包为 `omg` |
| source_sequence_id | string | 源序列编号，例如 `episode-00000000` |
| split_group_id | string | 来源分组标识，用于划分审计，不是动作路径 |
| split | string | `train`、`val` 或 `test` |
| source_provenance | string | JSON 字符串，包含源 episode 元数据、来源路径及文件校验和 |

真实第一条记录如下。为方便阅读，只展开 `source_provenance` 中的部分内容；磁盘上该字段是字符串，需要 `json.loads()` 才能成为字典。

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
    "upstream_sequence": "Aeroplane_BR__seg0000__part0000",
    "local_cleaned_release": true,
    "episode（仅部分字段）": {
      "episode_index": 0,
      "length": 90,
      "omg/split": "train",
      "tasks": [
        "raise arms to sides and initiate body rotation; style: smooth"
      ]
    }
  }
}
```

`source_root` 只是溯源信息，即使原始目录移动了，也不影响读取这个 bundle 中的动作和文本。

| split | 序列数 | 动作记录数 | 动作分片数 |
|---|---:|---:|---:|
| train | 583,029 | 1,166,058 | 166 |
| val | 73,790 | 147,580 | 21 |
| test | 74,854 | 149,708 | 22 |

本包保留上游 split。`motions.parquet` 本身没有 split 列，需要通过 `sequence_id` 关联本表。

## 5. meta/motions.parquet：动作在哪里、属于哪个版本

一行描述一段完整动作，共 1,463,346 行。它保存索引，不直接存储每帧的 36 维数组。

| 字段 | 存储类型 | 含义 |
|---|---|---|
| motion_id | string | 动作版本的唯一 ID |
| sequence_id | string | 关联 `sequences.sequence_id` |
| actor_id | string | 序列内的人物标识，本包示例为 `actor0` |
| group_id | string | 动作所属来源分组 |
| storage_uri | string | 相对于 release 根目录的动作分片路径 |
| row_start | int64 | 在该分片中的起始行，从 0 开始 |
| num_frames | int64 | 动作帧数 |
| fps | float64 | 发布帧率，本包为 30.0 |
| time_origin_sec | float64 | 该动作局部时间 0 对应的源序列时间，单位秒 |
| label_type | string | `source_g1` 或 `tracker_executed` |
| retarget_version | string | 重定向版本；OMG 原生 G1 此处为 `not_applicable` |
| tracker_version | string | 跟踪器版本；执行动作记录 SONIC 标识，原始动作为 `not_applicable` |
| quality_ref | string | 相对于 release 根目录的质量报告文件路径 |
| representation | string | 36 维动作表示的名称 |
| parent_motion_id | string，可为空 | 父动作 ID；SONIC 执行动作指向对应 `source_g1` |

实际第一组序列的两条动作记录：

```json
[
  {
    "motion_id": "omg:episode-00000000:source_g1",
    "sequence_id": "omg:episode-00000000",
    "actor_id": "actor0",
    "group_id": "100style_train:Aeroplane_BR__seg0000__part0000",
    "storage_uri": "motion/train/shard-00000.parquet",
    "row_start": 0,
    "num_frames": 90,
    "fps": 30.0,
    "time_origin_sec": 0.0,
    "label_type": "source_g1",
    "retarget_version": "not_applicable",
    "tracker_version": "not_applicable",
    "quality_ref": "reports/quality-00000.parquet",
    "representation": "g1_qpos36_wxyz_zup_radians",
    "parent_motion_id": null
  },
  {
    "motion_id": "omg:episode-00000000:tracker_executed",
    "sequence_id": "omg:episode-00000000",
    "actor_id": "actor0",
    "group_id": "100style_train:Aeroplane_BR__seg0000__part0000",
    "storage_uri": "motion/train/shard-00000.parquet",
    "row_start": 90,
    "num_frames": 89,
    "fps": 30.0,
    "time_origin_sec": 0.03333333333333333,
    "label_type": "tracker_executed",
    "retarget_version": "not_applicable",
    "tracker_version": "sonic:9f38d74cdef4d2677840fc57229c3d12b8844761e7a07f847d457e94a3c860fe",
    "quality_ref": "reports/quality-00000.parquet",
    "representation": "g1_qpos36_wxyz_zup_radians",
    "parent_motion_id": "omg:episode-00000000:source_g1"
  }
]
```

这两条动作都存放在 `motion/train/shard-00000.parquet`：

- 原始动作读取行区间 `[0, 90)`，得到 `[90, 36]` 数组。
- SONIC 执行动作读取行区间 `[90, 179)`，得到 `[89, 36]` 数组。

`row_start` 是分片内的帧行偏移，不是字节偏移，也不是 `motions.parquet` 的行号。完整定位公式为：

```python
frames = shard.slice(row_start, num_frames)
```

SONIC 执行动作通常比输入少一帧：初始状态不作为执行结果发布，第一帧从源时间 `1/30` 秒开始。不要假设原始版本和执行版本帧数、时间原点完全一致，应以各自记录为准。

## 6. motion/*/shard-*.parquet：真正的动作数值

每行是一帧，只有两列：

| 列 | Arrow 类型 | 含义 |
|---|---|---|
| qpos36 | fixed_size_list<float32>[36] | 这一帧的 36 维机器人位置状态 |
| timestamp_sec | float64 | 当前动作内部的局部时间，单位秒 |

同一个 shard 顺序拼接了多段动作。分片中没有 `motion_id` 列，动作边界由 `motions.parquet` 给出。每段动作的时间戳都从 0 开始，所以整个分片的时间戳并不全局递增。

分片目标上限为 1,000,000 帧，写入时尽量保留整条动作；单条动作不会因为达到这个目标而被切到两个 shard 中。当前第一份 train shard 有 999,773 帧。Parquet 内部还划分 row group，它是读取单位，不代表一条动作的边界；一条动作可能跨越 row group。

第一帧真实数值如下（为阅读方便保留 6 位小数）：

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

36 维的切片定义如下，索引均从 0 开始：

```python
root_position = qpos[:, 0:3]      # 世界坐标 x、y、z，单位米
root_quaternion = qpos[:, 3:7]    # w、x、y、z，单位四元数
joint_positions = qpos[:, 7:36]  # 29 个关节角，单位弧度
```

这里没有关节速度、控制力矩、接触标签或 SONIC 网络输入向量。36 维 qpos 表示机器人的姿态状态。

关节顺序必须以 `dataset.json → robot.joint_order` 为准。本包的实际顺序为：

| qpos 列索引 | 关节名称 |
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

动作内部时间戳为 `0, 1/30, 2/30, ..., (T-1)/30`。常用的帧覆盖时长为 `T/30` 秒，最后一帧的采样时刻为 `(T-1)/30` 秒，两者不要混淆。

局部时间转换到源序列时间：

```python
source_time_sec = timestamp_sec + time_origin_sec
```

SONIC 仿真控制为 50Hz，但 release 输出统一重采样为 30Hz。质量报告里的 `screening_fps: 50.0` 与动作表的 `fps: 30.0` 分别描述筛选和发布的时间网格。

## 7. meta/annotations.parquet：动作对应的文本

一行是一条文本标注，当前共 1,463,346 行。

| 字段 | 存储类型 | 含义 |
|---|---|---|
| annotation_id | string | 标注唯一 ID |
| sequence_id | string | 所属序列 |
| text | string | 文本描述 |
| language | string | 语言代码，本包为 `en` |
| start_sec | float64 | 对应动作局部时间中的起点，包含 |
| end_sec | float64 | 对应动作局部时间中的终点，不包含 |
| source | string | 文本来源，本包为 `OMG episode tasks` |
| confidence | float64 | 标注置信值；本包的 1.0 是写入值，不是模型评分 |
| motion_id | string，可为空 | 对应动作 ID；本包真实记录均绑定具体动作版本 |

真实第一组标注：

```json
[
  {
    "annotation_id": "omg:episode-00000000:source_g1:text:0",
    "sequence_id": "omg:episode-00000000",
    "text": "raise arms to sides and initiate body rotation; style: smooth",
    "language": "en",
    "start_sec": 0.0,
    "end_sec": 3.0,
    "source": "OMG episode tasks",
    "confidence": 1.0,
    "motion_id": "omg:episode-00000000:source_g1"
  },
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
]
```

两条文本相同，因为它们描述同一个序列的不同动作版本。有效时间区间经过各自动作的时间原点和时长裁剪，因此结束时间不同。直接通过 `motion_id` 查文本，能避免把另一个动作版本的时间区间误用进来。

## 8. 三张空表：assets、links、features

以下三张表在本 OMG 包中都是 **0 行**，但仍保存完整 Arrow schema，因此可以正常打开并查看列名。它们在 AIST++、FineDance、GDance 这样的音频动作包中承载实际数据。

### 8.1 meta/assets.parquet

一行表示一个原始音频或视频资产，而不是一段机器人动作。

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

### 8.2 meta/links.parquet

一行把一个动作与一个媒体资产对应起来，并明确两条时间轴的关系。

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

```python
asset_time_sec = time_scale * motion_local_time_sec + time_offset_sec
```

这也是为什么不能单凭动作与音频的总时长是否相等来判断对齐，应使用关联表中的有效区间及时间映射。

### 8.3 meta/features.parquet

一行描述一个资产的一份特征矩阵，它是特征的索引，不是特征数值本体。

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

有音频特征的包中，`features/audio/shard-*.parquet` 才是数值本体。一行保存一个 feature，列为 `feature_id: string`、`shape: list<int64>`、`dtype: string`、`values: large_list<float16>`、`timestamps: list<float64>`、`timestamp_shape: list<int64>`。`values` 是展平数组，读取时按 `shape` 还原；时间戳按 `timestamp_shape` 还原。当前 OMG 包没有这些文件。

## 9. meta/provenance.json：怎么处理出来的

这是普通 JSON，下面是当前完整内容：

```json
{
  "pipeline": "dataset_centric_v3",
  "release_policy": {
    "output": "../../data/processed/omg-text",
    "publish_labels": [
      "source_g1",
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
    "accepted": 731673,
    "motion_accepted": 731673,
    "rejected": 25727,
    "motion_rejected": 25727
  },
  "rejection_reasons": {
    "persistent_fall": 25727
  },
  "package_mode": "bundle",
  "portable": true,
  "media_bundled": false
}
```

`publish_labels` 说明发布了哪些动作版本；`train_default` 指定推荐训练版本；`split_policy: preserve` 说明保留来源划分。`output` 是生成时的配置记录，不是读取时必须存在的依赖路径。

处理了 757,400 个序列，731,673 个通过、25,727 个拒绝。拒绝原因为 `persistent_fall`。被拒绝序列的动作没有写入最终动作分片，其逐条报告也没有全部保留在这个最终包里；这里保留的是拒绝原因汇总。

`portable: true` 表示独立打包；`media_bundled: false` 表示没有打包原始音视频。

## 10. reports/quality-00000.parquet：每条已发布动作的质检报告

共 1,463,346 行，使用以下三列：

| 字段 | 存储类型 | 含义 |
|---|---|---|
| motion_id | string | 关联动作 ID |
| accepted | bool | 是否通过发布筛选；本包已发布记录均为 true |
| report_json | string | JSON 字符串，保存该动作的详细报告 |

真实 SONIC 执行动作的报告如下。为展示内部结构，已将 `report_json` 从字符串解析成字典：

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

`screening` 中的帧数和指标来自 50Hz 执行轨迹；`tracking` 包含与参考动作比较的关节 RMSE（弧度）及全局身体 MPJPE（毫米）。原始 `source_g1` 报告更简单，主要记录来源阶段和接受状态。

`accepted: true` 表示通过当前筛选，不保证动作视觉完美。`production_ready: false` 是当前流程写入的状态标记，不表示该记录被拒绝。`strict_joint_limits: false` 表示此处记录关节越界指标，但未启用严格越界拒绝规则。

质量报告是共享表，`quality_ref` 指向报告文件；必须再按 `motion_id` 找到对应行，不能把整个文件当作某一条动作的报告。

## 11. reports/dataset_summary.json：最终汇总

当前真实内容：

```json
{
  "valid": true,
  "processing_summary": {
    "accepted": 731673,
    "motion_accepted": 731673,
    "rejected": 25727,
    "motion_rejected": 25727
  },
  "rejection_reasons": {
    "persistent_fall": 25727
  },
  "sequences": 731673,
  "motions": 1463346,
  "assets": 0,
  "features": 0,
  "duration_sec": 6926005.633342877,
  "sources": {
    "omg": 731673
  },
  "modality_combinations": {
    "text": 1463346
  },
  "deep_checked_motions": 1463346,
  "deep_checked_features": 0
}
```

`valid: true` 表示这次数据包校验通过，`deep_checked_motions` 表示实际检查了多少条动作的数值与时间戳。此包为全部 1,463,346 条。

`duration_sec` 是所有动作记录的时长之和，包含原始与执行两个版本，不能直接理解为独立采集总时长。`modality_combinations.text` 统计带文本的动作记录数。

## 12. 文件如何关联

```text
sequences.sequence_id
    └── motions.sequence_id
            ├── storage_uri + row_start + num_frames → 动作分片的具体帧
            ├── motion_id → annotations.motion_id → 文本
            ├── motion_id → quality.motion_id → 质量报告
            ├── parent_motion_id → 原始/父动作
            └── motion_id → links.motion_id（本包为空）
                                └── asset_id → assets / features（本包为空）
```

训练时一般先通过序列确定 split，再选择 `label_type == 'tracker_executed'`，最后按 `motion_id` 取文本、按分片路径和行范围取动作。不要不加筛选地把两个版本都作为同一种训练标签。

## 13. 实际读取一条动作和文本

在 MSC 根目录执行下面代码，使用项目 `.venv/bin/python`。示例选择真实第一条序列的 SONIC 执行动作；使用项目已有的 `read_motion()`，只读取与动作相交的 row group，避免加载整个动作分片。

```python
from pathlib import Path
import json
import pyarrow.parquet as pq
from data_pipeline.schema.records import MotionRecord
from data_pipeline.release.export import read_motion

root = Path('/home1/gmh/wangyang/motion/MSC/data/processed/omg-text/release')
motion_id = 'omg:episode-00000000:tracker_executed'

row = pq.read_table(
    root / 'meta/motions.parquet',
    filters=[('motion_id', '=', motion_id)],
).to_pylist()[0]
record = MotionRecord(**row)
qpos, timestamps = read_motion(root, record)

sequence = pq.read_table(
    root / 'meta/sequences.parquet',
    filters=[('sequence_id', '=', record.sequence_id)],
).to_pylist()[0]
texts = pq.read_table(
    root / 'meta/annotations.parquet',
    filters=[('motion_id', '=', motion_id)],
).to_pylist()
quality = pq.read_table(
    root / record.quality_ref,
    filters=[('motion_id', '=', motion_id)],
).to_pylist()[0]
quality_report = json.loads(quality['report_json'])

print(sequence['split'])         # train
print(qpos.shape, qpos.dtype)     # (89, 36) float32
print(timestamps[:3])             # [0.         0.03333333 0.06666667]
print(record.time_origin_sec)    # 0.03333333333333333
print(texts[0]['text'])
# raise arms to sides and initiate body rotation; style: smooth
print(quality_report['accepted']) # True
```

这是单样本查看示例。批量训练应预先建立 split、motion_id、文本之间的索引，并按 shard 组织读取，避免每取一个样本都扫描元数据和重新打开文件。

## 14. 查看任意 Parquet 的结构和前几行

下面的读取方法只取一个小批次，适合先了解文件，不会把整张表转成 Python 列表。

```python
from pathlib import Path
import pyarrow.parquet as pq

root = Path('/home1/gmh/wangyang/motion/MSC/data/processed/omg-text/release')
file = pq.ParquetFile(root / 'meta/motions.parquet')
print(file.schema_arrow)
print('行数:', file.metadata.num_rows)
print('row group 数:', file.num_row_groups)
for batch in file.iter_batches(batch_size=2):
    print(batch.to_pylist())
    break
```

对于空表，仍可查看 `schema_arrow` 和行数，循环不会产生记录。若要查看音频数据集，替换 `root` 即可；基本的六表结构相同，实际模态和 label 类型以各包内容为准。
