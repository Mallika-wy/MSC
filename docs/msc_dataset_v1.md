# MSC Dataset v1

## 安装与构建

Python >=3.10。离线工具不自动下载原始数据或模型。

```bash
python -m pip install -e '.[pipeline,test]'
msc-data discover --source omg --root /path/OMG-Data --out /work/inventory --limit 1
msc-data build --config /work/build.yaml --stages motion,text --limit 1
msc-data validate --dataset /work/release --deep-sample 20
msc-data package --dataset /work/release --out /work/portable --mode bundle
```

`/work/build.yaml`（所有路径相对 YAML 所在目录）：

```yaml
inventory: inventory
output: release
tracker:
  enabled: false
```

AIST 使用 `discover --source aistpp --root /path/AIST --out /work/inventory --cameras c01 c02`，
支持 `archives/motions_complete.zip`、已解压 `motions/*.pkl`，以及本机
`data/raw_audio/AISTpp/edge_aistpp/{motions,wavs}`。同名 WAV 记录为源资产，
不自动伪造已验证同步或 MERT 缓存；视频可缺失。构建配置见
[`aistpp_motion.yaml`](../configs/data_pipeline/aistpp_motion.yaml)；`split_assignments` 优先按 sequence_id 指定 split，也兼容旧 split_group_id 配置。

GMR 直接使用本机 `/home1/gmh/wangyang/motion/GMR`，示例 YAML 的
`gmr_root: ../../../GMR` 指向这个目录。使用已装齐 GMR/SMPL 依赖的离线 Python 环境：

```bash
python -m pip install -e '.[pipeline]'
msc-data build --config configs/data_pipeline/aistpp_motion.yaml --stages motion
```

流程：`source decoder → HumanMotion → shared run_gmr() → G1 qpos36`。
`stages/retarget.py` 直接导入该目录的 `GeneralMotionRetargeting`，不复制 GMR，
不再维护每个数据集一份 backend/子进程。数据源只负责格式、单位、坐标和时间的解码；
共用接口为关节名、全局位置（米/Z-up）、wxyz朝向、30Hz秒时间戳、身高和 GMR 骨架配置名。
新人体格式在 `HUMAN_DECODERS` 注册解码器，之后复用同一求解逻辑；
同格式的新数据源可复用解码器。OMG 原生 G1 直接跳过重定向。
已接入 AIST SMPL、FineDance SMPL-H 及拆分后 GDance SMPL 的解码；FineDance 和
GDance 使用中性 SMPL-X 作为人体近似，时间和音频对齐仍需进一步验证。

启用 tracker 必须设置 `tracker.enabled: true`、`tracker.onnx` 和显式 `tracker.version`。
每条已执行轨迹在工作目录的 `tracking/<缓存键>/execution.mp4` 保存 50Hz MuJoCo
实际状态回放；即使质检拒绝该轨迹也会保留。质量报告的 `execution_video` 指向该文件，
视频保存在工作目录而非发布包中；需要保留工作目录以便后续检查。
官方 OMG split 保留，不从旧35D特征推断原始音频。
当前支持 motion/text build；真实音视频特征提取和统一训练 loader 尚未完成。

原 gmr 环境本身可用。普通启动会被 `~/.local` 中旧 NCCL 污染；
使用 `python -s` 或 `PYTHONNOUSERSITE=1` 即可排除用户级包，无需重装 Torch。
本项目 `.venv-gmr` 是继承原 gmr 环境的轻量本地 venv，只补装 PyArrow 和 MSC，
不修改原 Conda 环境。运行方式：

```bash
PYTHONNOUSERSITE=1 .venv-gmr/bin/python -s -m data_pipeline.cli build \
  --config configs/data_pipeline/aistpp_motion.yaml --stages motion --limit 1
```

已验证原环境中的真实 GMR 求解器正常。模型来自 HMR2.0 的公开固定版本文件，源 URL、SHA256、结构和原模型条款说明保存在
`model/smpl/source.json`；只缓存本地模型，不随代码或数据包分发。
数据目录与模型路径已写入 `configs/data_pipeline/aistpp_motion.yaml`。

## 数据契约

```text
release/
  dataset.json
  meta/{sequences,motions,assets,links,annotations,features}.parquet
  meta/provenance.json
  motion/shard-*.parquet
  reports/{quality-*,dataset_summary}.json
  features/…       # 可选缓存
  media/…          # bundle 时复制
```

六类实体字段以 [`schemas/records.py`](../data_pipeline/schemas/records.py) 为准；
原始待解码动作单独用 `SourceMotion` 写入库存 `meta/source_motions.json`。
原生 release 只存每条完整 G1 qpos36，不生成重叠窗口和冗余 action 列。
Parquet：`qpos36: fixed_size_list<float32>[36]`，`timestamp_sec: float64`。

- qpos36 = xyz3 + quat_wxyz4 + joint29；米/弧度，Z-up，30Hz。
- manifest 包含 schema_version=1.0、机器人关节顺序、URDF/FK资产 SHA256 和 external_roots。
- Motion 本地时间从零开始；`time_origin_sec` 保留其在公共源序列中的起点。
- 条件映射：`t_asset = scale*t_motion + offset`，scale>0。
  旧 AIST `sequence=asset+old_offset` 转换为 `offset=motion_origin-old_offset`。
- 区间均为 `[start,end)`。音频 point 时间戳为 `[T]`；视频 tubelet 支持区间为 `[T,2]`，均为 float64 秒。
- 文字可以有多段有效区间；不强制任何样本具有 audio 或 video。
- AIST 按源动作序列独立分配 split；共享歌曲或编舞允许跨集合，`split_group_id` 不再强制同组同 split。
- 参考与执行动作分别为 `retarget_reference` 和 `tracker_executed`，后者关联参考 ID。
  50Hz 第 i 个保存状态在 `(i+1)/50`；SLERP 回30Hz后排除 t=0 初始化，保留真实源起点。

每条轨迹关联独立质量报告。representation/arm-twist/tracking 检查的适用范围明确，
通过不代表完整物理或语义质量合格。Neutral SMPL 需显式 opt-in；
AIST scale、Y-up→Z-up、首帧水平平移和整段常量地面高度处理不随重构改变。

URI 使用根相对路径或 `root://alias/path`；外部 alias 显式写入 external_roots。
禁止路径穿越及逃逸根目录的 symlink。特征按资产共享，记录输入校验和、encoder checkpoint、
预处理与提取代码指纹，不按维度推断兼容性。

## 验证与缓存

`validate` 默认检查 metadata、外键、split leakage、文件存在性、Parquet schema/row ranges 和质量报告。
`--deep-sample N` 额外抽查前 N 个 motion/asset/feature 的数值、SHA256、shape/dtype、时间轴及覆盖。
该检查不代表原始 WAV/MP4 全量解码或真实 motion 质量评估。

release/package 拒绝覆盖已存在目标，验证后原子发布。AIST stage 缓存键包含原始输入、SMPL模型、
GMR版本及本地差异、机器人契约、处理代码与配置指纹。失败或改变配置时使用新的 output/work 目录。

bundle 按文件身份复制媒体和特征，改写为相对路径，移动后可读；reference 保留媒体 root aliases，
不宣称可携带。原始数据、旧成品与模型权重不会被删除或自动重算。

## 人物裁剪

```bash
msc-data crop-video --video /path/input.mp4 --output-root /work/crops \
  --config configs/data_pipeline/person_video.yaml --device cpu
msc-data validate-video --directory /work/crops/input --decode
```

先配置真实本地 YOLO 权重。相对权重路径以 YAML 目录为基准。
输出 `person.mp4`（512×512）、`metadata.npz`（真实源PTS、采样索引、裁剪框、正逆仿射）、
检测缓存和质量报告；不生成调试预览或机器人回放。多人场景可传 `--track-id`。

## 回归测试

```bash
python -m pytest data_pipeline/tests tests -q -m 'not integration and not slow'
```

测试目录中的 fixture 覆盖 Text/Audio/Video/Combined、多演员/相机、75Hz 时间偏移、
tracker 初始化排除和 bundle 移动读取。Synthetic features 不是 MERT 或真实视频编码结果。
本机真实 OMG 90帧样本已验证。SMPL Male 已下载并校验，配置指向
`model/smpl/SMPL_MALE.pkl`。
