# 数据处理入口与阅读顺序

当前处理 AIST++、FineDance、AIOZ-GDance 的音频—动作对。OMG 汇总文本入口已移除；
后续文本原始数据集分别放到 `sources/text/<dataset>/`。

## 从这几个文件开始

1. `cli.py`：解析命令，调用 pipeline。
2. `config.py`：读取 YAML，将路径解析为绝对路径。
3. `pipeline.py`：逐条组织 GMR、质检、tracker/视频、MERT 和导出。
4. `sources/audio/<dataset>/dataset.py`：找动作与配对 WAV。
5. `sources/audio/<dataset>/motion.py`：解码该数据集的人体动作。
6. `motion/retarget.py`、`motion/tracking.py`：共用 GMR 与 OMG tracker。

新数据集只需提供 `samples(root)` 和 `decode_motion(path, settings)`，
并在 `sources/audio/__init__.py` 的数据集名单注册。

## 目录职责

| 目录/文件 | 负责什么 |
|---|---|
| `sources/audio/aistpp/` | AIST++ SMPL 60Hz → 人体关节 30Hz |
| `sources/audio/finedance/` | FineDance 315D、52个关节的6D旋转；按矩阵行解码 |
| `sources/audio/gdance/` | 读取拆分人物的索引、SMPL参数和共享音频；含拆分脚本 |
| `motion/decoding.py` | SMPL-X前向运动学及Y-up → Z-up转换 |
| `motion/retarget.py` | 人体关节 → 共用GMR → G1参考动作；处理缓存 |
| `motion/tracking.py` | 50Hz MuJoCo执行、执行缓存、30Hz标签 |
| `motion/arm_twist.py` | 参考动作的手臂扭转启发式检查 |
| `motion/quality.py` | 执行动作的跌倒、限位检查与时间重采样 |
| `audio/mert.py` | WAV预处理与MERT编码，只返回数组与时间戳 |
| `audio/cache.py` | 特征指纹、去重、保存和复用；共享音乐只编码一次 |
| `video/execution.py` | 实际执行状态的MuJoCo回放，保存每条已执行动作的视频 |
| `video/processor.py` | 独立的原始人物RGB视频裁剪入口 |
| `video/detection.py`、`geometry.py`、`media.py`、`config.py` | 人物检测、裁剪几何、视频读写、参数 |
| `schema/records.py` | 六类发布记录。保留文本字段以兼容格式，当前流程不生成文本 |
| `schema/manifest.py` | 元信息与Parquet表的读写 |
| `schema/validation.py` | 字段、引用、时间覆盖和数据契约检查 |
| `schema/time_map.py` | 动作与模态时间映射 |
| `release/audio.py` | 将合格样本转换为发布记录 |
| `release/export.py`、`validate.py`、`package.py` | 正式导出、检查和搬运打包 |
| `io.py`、`robot.py` | JSON、指纹、路径解析和G1关节约定 |
| `examples/process_one_sample.py` | 调试单条样本，调用同一个pipeline |

FineDance原始数据是SMPL-H；FineDance和GDance目前均使用中性SMPL-X人体近似。
G1动作统一为36维：根位置3 + 根四元数wxyz 4 + 关节角29，单位米/弧度，Z向上。

## 运行

本机使用 `.venv-gmr`，继承已有GMR依赖，并在该环境安装MERT依赖与本地OMG包。
代码不再把其他项目的site-packages插入sys.path。新环境需安装MSC的pipeline依赖、
GMR依赖及本地OMG tracker包。

在MSC根目录：

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:$PWD/src"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4
export LP_NUM_THREADS=4

# 只看即将处理的配对，不加载模型，不产生输出
.venv-gmr/bin/python -s -m data_pipeline.cli run \
  --config configs/data_pipeline/aistpp_motion.yaml --limit 3 --dry-run

# 三个数据集分别运行一条；去掉--limit可处理全量
.venv-gmr/bin/python -s -m data_pipeline.cli run \
  --config configs/data_pipeline/aistpp_motion.yaml --limit 1
.venv-gmr/bin/python -s -m data_pipeline.cli run \
  --config configs/data_pipeline/finedance.yaml --limit 1
.venv-gmr/bin/python -s -m data_pipeline.cli run \
  --config configs/data_pipeline/gdance.yaml --limit 1

# 和批处理完全共用逻辑的调试示例
.venv-gmr/bin/python -s -m data_pipeline.examples.process_one_sample finedance \
  --output data/processed/debug/finedance
```

`--sequence` 可选单个样本或GDance组；`--output` 可指定新的输出目录。
`--diagnostic` 允许执行未通过参考质检的动作供人工检查，仍不发布失败样本。
一条数据报错会写出error.json，继续下一条；命令最终在有错误时返回非零状态。

GDance需要先有单人索引。本机已拆好，重做时：

```bash
.venv-gmr/bin/python -s -m data_pipeline.cli split-gdance \
  --root data/raw_audio/AIOZ-GDance \
  --output data/processed/AIOZ-GDance-single-person
```

## 输出和恢复

原始数据在 `data/raw_audio/`，模型在 `model/`，结果在 `data/processed/audio/<dataset>/`。

```text
output/
  config.json
  progress.json
  summary.json
  samples/<sample>/
    status.json
    error.json                     # 该样本报错时产生
    reference/<fingerprint>/       # GMR参考动作和报告
    tracking/<fingerprint>/
      tracker_rollout.npz           # 真实50Hz执行状态
      paired_motion.npz             # 对齐到30Hz，首帧初始化状态无效
      labels.npz                   # 发布时排除初始化帧
      quality.json
      execution.mp4                # 512×512、50Hz，包含质检拒绝的执行
  audio/<fingerprint>/
    values.npy
    times.npy
    features.json
  release/                         # 只包含通过参考与执行质检的样本
    dataset.json
    meta/
    motion/
    reports/
```

失败后可对同一输出目录重试，已有指纹匹配的计算会复用。
成功发布后，重跑请使用新的 `--output`；不会覆盖已有release。
发布按条读取动作，不一次把全部动作放进内存。
AIST++和FineDance默认train；GDance保留官方train/val/test。共享歌曲或编舞不限制跨集合。
同一GDance片段的演员共享同一音频资源和特征。

MERT采用5秒分块，输出最后一层float16特征；每块重新建立上下文。
时间戳为卷积感受野中心，不能当作30Hz动作帧。
音频与动作匹配按原始片段身份建立，未独立校准的时间偏移标为estimated。
执行标签从t=1/30秒起，发布记录和音频链接都保存这个偏移。

回放视频保留在工作输出目录。移动结果或人工检查时应一起保留该目录。
人物RGB裁剪和MuJoCo回放是不同功能；当前音频流程不运行人物裁剪。
当前发布格式尚未接入统一训练loader。

## 检查

```bash
python -m data_pipeline.cli validate --dataset <output>/release --deep-sample 20
python -m data_pipeline.cli package --dataset <output>/release --out <bundle> --mode bundle
python -m pytest data_pipeline/tests tests -q -m 'not integration and not slow'
```
