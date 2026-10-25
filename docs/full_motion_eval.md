# 完整动作评测

`eval.sh` 调用 `python -m msc.eval_full`，加载指定 checkpoint，在指定 Release 中按 seed 随机抽取 **10 条不同动作**，逐条连续生成至 GT 最后一帧。支持推理时 RTC 开关、模型生成历史反馈，以及可选 SONIC 实际执行历史反馈。

## 直接使用

默认沿用 checkpoint 中的数据集与子集清单，适合当前 100 条 Text 过拟合实验：

```bash
CHECKPOINT=.runs/msc-v3-overfit-text100/checkpoints/step_1000.pt \
SEED=42 RTC=false OUTPUT_DIR=.runs/text100-eval-off bash eval.sh

CHECKPOINT=.runs/msc-v3-overfit-text100/checkpoints/step_1000.pt \
SEED=42 RTC=true OUTPUT_DIR=.runs/text100-eval-rtc bash eval.sh
```

两次使用相同数据、seed、步长、推理参数和 feedback，将得到相同抽样动作及每轮初始噪声。RTC 开启使用现有 `test_time` 引导；无需 checkpoint 曾经启用 RTC 训练。

指定整个数据集：

```bash
CHECKPOINT=/path/to/step_60000.pt \
DATA_ROOT=data/processed/omg-text/release \
SPLIT=train TASK=text SEED=123 RTC=true \
OUTPUT_DIR=.runs/omg-full-eval bash eval.sh
```

显式 `DATA_ROOT` 会清除 checkpoint 继承的子集限制，避免指定新数据集却仍只评测训练子集。若需要指定清单，增加 `MOTION_MANIFEST=configs/overfit_text100_manifest.json`；使用 `MOTION_MANIFEST=none` 可单独清除继承的清单。

## 可修改参数

所有参数位于 `eval.sh` 顶部，可用环境变量覆盖；末尾 CLI 参数优先。

| 参数 | 默认值 / 含义 |
|---|---|
| `CHECKPOINT` | 必填，指定模型 |
| `CUDA_VISIBLE_DEVICES` / `DEVICE` | `4` / `cuda:0` |
| `DATA_ROOT` | 留空继承模型；指定后读取对应 Release |
| `MOTION_MANIFEST` | 留空按上述规则继承，路径选子集，`none` 清除 |
| `TASK` / `SPLIT` | `text` / `train`；支持 audio 及具备真实特征的 video |
| `SEED` / `NUM_MOTIONS` | `42` / `10` |
| `RTC` | `false`，改为 `true` 启用推理时 RTC |
| `FEEDBACK` | `generated` 或 `sonic` |
| `REPLAN_FRAMES` | `15`；每轮预测 30 帧、执行前 15 帧再重新规划 |
| `INFERENCE_STEPS` / `CFG_SCALE` | `20` / `1.0` |
| `RTC_ALPHA` / `RTC_MASK` | `0.9` / `exponential` |
| `PRECISION` | CUDA 默认 BF16；CPU 使用 FP32 |
| `VIDEO` / `PANEL_SIZE` | `true` / `512`；`VIDEO=false` 只保存轨迹和指标 |
| `CAPTION_FONT` | 可选 TTF/TTC 字体；默认优先 Noto CJK，显示中英文 |
| `SONIC_CONFIG` | `configs/data_pipeline/omg_text.yaml`，提供仿真/渲染资源 |
| `OUTPUT_DIR` | `.runs/msc-full-eval`；不同实验请使用不同输出目录 |

CLI 支持重复 `--data-root` 指定多个 Release；完整选项可通过 `PYTHONPATH=src:. .venv/bin/python -m msc.eval_full --help` 查看。

## 完整时长与输入契约

- 按 motion 抽样，不按窗口抽样，不按 loss 挑选。不因动作更长、滑窗更多而提高其被抽中的概率。
- 仅纳入从第 5 帧到 GT 结尾都有外部条件覆盖的完整动作。有覆盖缺口的动作被排除，绝不截短后冒充完整评测；不足指定数量时报错。
- 最初 **5 帧 GT 是已知启动历史**。其后没有任何 GT 运动历史或未来动作被重新注入模型；只有外部文本/音频/视频条件按时间读取。
- 每轮生成未来 30 帧，按指定步长提交前一部分；最后不足一块时只提交剩余帧。音频/视频尾部直接读取对应时间的特征，不越界读取不存在的完整 GT 窗口。
- 输出数组和视频均保持 **GT 的全部 N 帧、30 FPS**。例如 300 帧 GT 的视频仍为 10 秒，包含最初 5 帧启动历史及其后的 295 帧连续生成。GT 不是整数秒时同样逐帧对齐。
- RTC 开/关保持相同重规划步长；开启后将前一计划未执行部分变换到新的 Canonical 坐标，保留全部 125D 条件目标。首轮没有上一计划，普通采样。
- 不使用固定噪声挑最好种子；seed 同时控制抽样与每条动作的独立噪声序列，RTC 开关不改变噪声序列。
- 这是离线同步推理，模拟推理延迟为 0，不声明具备在线实时性能。

## 视频与指标

文本任务的视频下方有独立字幕区，完整显示传入的 caption，自动换行，不遮挡机器人；若文本随片段变化，字幕同步更新。字幕保留原文，不作摘要。模型仍使用 checkpoint 的 T5 token 上限，保存在 `results.json` 的 `text_max_tokens` 中；字幕显示全文不代表更改了模型 tokenizer。

`FEEDBACK=generated` 时视频为 **GT / 生成参考** 两栏，历史来自已提交的生成帧，用于直接诊断模型。`FEEDBACK=sonic` 时增加 **SONIC 实际执行** 第三栏，仿真器及策略历史全程保留，下一轮历史来自测量状态：

```bash
CHECKPOINT=/path/to/model.pt SEED=42 RTC=true FEEDBACK=sonic \
OUTPUT_DIR=.runs/full-eval-sonic bash eval.sh
```

SONIC 的重规划步长需为 3 的倍数，使 30 FPS 与 50 Hz 控制同步。仅末尾不足一块时允许控制多运行不足 20ms，再插值到精确 GT 时间点；不增加输出视频长度。

每次输出：

- `selection.json`：数据根目录、子集、seed、10 条动作、条件时间范围及逐动作噪声种子。
- `results.json`：参数、RTC 模式、逐动作指标、每轮耗时/条件/反馈来源、视频帧数。
- `index.html`：字幕与视频浏览入口。
- `01/` 至 `10/`：`comparison.mp4`、预览 JPG、`trajectories.npz`、`metrics.json`。

指标排除共享的前 5 帧，包括关节 RMSE、世界位置误差、FK MPJPE、运动幅度、速度误差、终点漂移，以及保持末姿态的对照。SONIC 模式另报实际执行指标。所有视频均检查帧数、30 FPS，并完整解码验证；原速、无循环或慢放，音频任务暂不附原始音轨。相机跟随各自 root，绝对位移应查看指标和轨迹。

训练集结果是重建诊断，不是泛化结论；运动幅度或文本敏感性也不能单独证明语义忠实。真实 Video 特征仍不可用，其真实评测为 Pending，不以合成特征测试替代。
