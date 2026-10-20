# 视频 / 动作数据流水线

```text
数据源根目录 → 对应 adapter → pairs.csv
                                  ↓
                  动作解码 / 重定向 → tracker 执行
                                  ↓
                  人物裁剪512×512 → 同时间轴配对 → OMG导出
```

通用 pipeline 只读 CSV，不需要了解 AIST++ / FreeMan 等其他数据源的目录结构。

本文是统一使用入口；[AIST动作约定与已知问题](aistpp.md)、
[人物裁剪组件](components/person_video/README.md)补充格式、质检、实验结论和裁剪限制。
官方训练数据的加载/缓存契约仍见[训练数据说明](../docs/data.md)，与原始数据处理分开。
处理后肘腕组合筛查的两档规则、开关及已知误报边界见[AIST质检说明](aistpp.md#肘腕组合筛查2026-10-19新增)。

## 1. 生成配对 CSV

```bash
PYTHONPATH=.:src .venv/bin/python -m data_pipeline.cli prepare \
  --source aistpp --root /home1/gmh/wangyang/motion/OMG/data/AIST++
```

生成 `data/AIST++/pairs.csv`：每行一个“视频机位 + 动作”，应用官方 ignore list。
当前 c01 对应1,363行。原始动作在 ZIP 内，直接记录路径和 member，不重复解压。
重新 prepare 会重建 CSV；手工编辑的版本应另存文件，用 `run --csv` 指定。

| 列 | 含义 |
|---|---|
| source / sequence_id / camera_id | 数据源、动作序列、视频机位，必填 |
| video_path / motion_path | 文件位置，必填；相对路径相对 CSV 所在目录 |
| motion_format | 动作解码器名称，必填，不通过扩展名猜骨架/坐标 |
| motion_member | ZIP 内文件名；普通动作文件留空 |
| motion_fps | 源动作帧率；AIST为60 |
| video_time_offset_s / motion_time_offset_s | `sequence_time = asset_time + offset`；默认0表示已同步 |
| split_group | 防止同源序列、机位和重叠片段跨训练/验证集合 |
| source_root | 可选源资源目录，AIST用来加载旧SMPL依赖 |
| camera_path | 可选原始相机标定；裁剪图需要再应用像素变换 |

## 2. 处理 CSV

检测额外依赖仍位于独立目录，不升级 OMG 训练环境：

```bash
# 先小批量调试；subset_1 单独存放，不覆盖全量结果
PYTHONPATH=.:src:data/processed/person_crop/runtime_deps YOLO_AUTOINSTALL=false \
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=4 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python -m data_pipeline.cli run \
  --config data_pipeline/configs/aistpp_full.yaml --video-device 4 --limit 1

# 全量（GPU编号和渲染设备按实际空闲资源选择，不要与已有全量任务重复启动）
PYTHONPATH=.:src:data/processed/person_crop/runtime_deps YOLO_AUTOINSTALL=false \
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=4 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python -m data_pipeline.cli run \
  --config data_pipeline/configs/aistpp_full.yaml --video-device 4 --workers 12
```

`--until inventory` 只读取CSV并检查路径；`--until process` 不导出；
`--csv /path/to/pairs.csv` 覆盖配置入口。配置中的其他路径相对 OMG 根目录。
默认复用已完成动作和视频；更改配置/输入内容后，用新输出目录或 `--force` 重新处理。
不扫描内容哈希，因此在原路径替换动作文件时必须主动 `--force`。
失败保留已完成输出和错误，修好后重复命令。默认 `--workers 1` 便于单步调试；
全量当前使用 `--workers 12`，由一个主进程分配互不重叠的序列目录并统一更新进度、导出。
不要另外启动第二个命令写同一输出目录。并行度不写入数据配置，改变worker数可继续复用结果。
每个worker复用YOLO权重，但在新视频前重置ByteTrack状态，避免人物身份跨视频串联。
视频解码限制为2线程，减少多进程时的CPU线程争抢；分辨率、采样和质检保持不变。
12路是本机小批量验证过的档位，不是硬件上限；早期4/8/12路计时未包含执行视频渲染，
不能直接用于当前全量ETA。24路尚未完成对照验证；不要只根据显存余量增加进程。

## 检查进度和导出

查看`<output.root>/results.csv`与启动任务时指定的日志；`complete`、`rejected`、`failed`
应分别统计。命令本身不创建后台会话，不要把“已启动”当成“已完成”。

```bash
PYTHONPATH=.:src .venv/bin/python -m data_pipeline.cli validate \
  --dataset data/processed/aistpp_full_prototype/release
```

有部分运行失败时，成功部分仍可能已经导出；最终命令会抛出汇总异常，不能据此忽略
失败项，也不能把部分release称作全量成功。质检拒绝的动作不进入release，原始文件不删除。

### 打包为独立数据集

默认`release/`只有动作和索引，人物视频在`sequences/`中，不能只搬走release后删除其视频。
发布或整理最终数据时使用独立打包入口（输出目录必须不存在）：

```bash
PYTHONPATH=.:src .venv/bin/python -m data_pipeline.cli package \
  --release data/processed/aistpp_full_prototype/release --output data/AIST++
```

打包复制动作、`videos/<source>/<sequence>/<camera>/person.mp4`及裁剪映射，
并将存在的执行回放复制到`execution_videos/`。重叠episode共享同一视频，不重复切片。
`meta/video_index.json`中的`uri`、`crop_metadata_uri`、`execution_video_uri`均相对数据集根目录，
不是相对meta目录或当前工作目录；移动整个包后仍可验证。原片只保留名字，不要求原片存在。
源处理目录保持不动；确认独立包验收完成前，不要删除尚被外部索引引用的派生视频。
打包后manifest身份值会更新，仍以`validate`输出为准，不代表已实现视频条件训练加载器。
若输出命名为`data/AIST++`，该目录此后是处理好的成品，不是adapter所需的原始数据根目录；
重新重定向需要另行提供原始动作、原片和对应CSV，不能对成品目录执行`prepare`。

### 重定向后肘腕筛查

正常流水线在重定向后立即筛查参考动作，命中就标为rejected，不再运行控制器、
生成执行回放、裁剪RGB或导出。复用参考缓存也会重新筛查；不再在执行后做肘腕筛查。

## 3. 输出（名称直接可读）

```text
<output.root>/
  config.json                     一份运行参数，不是审计
  results.csv                     每条序列 complete / rejected / failed 和错误
  sequences/<source>/<sequence>/
    reference/motion.npz           30Hz G1参考动作
    reference/quality.json         重定向后肘腕筛查，命中就停止该条后续处理
    tracker_rollout.npz            50Hz执行结果
    execution.mp4                  512×512执行回放（50FPS），仅实际执行过的序列
    paired_motion.npz              30Hz动作与有效性掩码
    quality.json / video_index.json
    videos/<camera>/person.mp4     512×512、30FPS
                    metadata.npz   PTS / 裁剪框 / 正逆坐标变换
                    detections.json / settings.json / quality.json
  release/                        OMG动作Parquet、episode元数据和视频索引
```

整段动作先处理再切片，多机位共享动作处理，重叠episode共享视频。原始视频不修改。
每条序列默认保存动作执行视频，即使动作质检拒绝也保留倒地过程；视频来自保存的
tracker执行状态，不是参考动作。重复运行会给旧结果补视频，无需重新仿真。
训练人物RGB视频仍为512×512、30FPS；execution.mp4按原生50Hz回放，仅用于诊断。
保留必要的数值、跌倒、限位、时间覆盖、尺寸、split和加载检查；这些不是下载审计。
`release/meta` 是训练格式必需元数据；小型 `omg_manifest.json` 仅为兼容现有 OMG
加载器，不含原始文件哈希清单或来源审计。视频条件训练读取分支尚未实现。

## 4. 接新数据源

1. **文件组织不同**：在 `adapters/` 写根目录→CSV的 builder，加入 `BUILDERS`。
2. **动作表示不同**：在 `stages/motion.py` 的 `DECODERS` 加解码器；必须明确单位、
   骨架、坐标、旋转和时间语义。不能把任意 `.npy/.pkl` 改名就当作SMPL。
3. 后面的执行、裁剪、配对和导出共用，不复制另一套pipeline。

当前支持 `aist_smpl`（原始AIST SMPL PKL/ZIP）与 `g1_qpos_wxyz`（已规范化的G1
NPZ，含 `qpos_36[T,36]`、`timestamps[T]`，米制/Z-up/wxyz/官方29关节顺序/30Hz）。
只有视频、没有动作的互联网数据，应先估计动作伪标签，再生成同样的CSV。

当前AIST配置显式使用可用 neutral SMPL，仍是原型，不应当作正式男体模型标签。
正式使用需换合法可用的male模型并验证重定向质量。人物裁剪仅支持单连续镜头。

## 代码阅读与调试

`cli.py → adapters/aistpp.py → core/pairs.py → core/runner.py → stages/motion.py /
stages/video.py → components/person_video/ → exporters/omg.py`。
动作分支的GMR后端在`backends/aist_smpl_gmr.py`，执行质检和时间插值在
`stages/motion_quality.py`，对应测试都在`data_pipeline/tests/`。

`/home1/gmh/wangyang/.vscode/launch.json` 已对应更新。先运行00生成CSV，再用01读单条
代码；02/03分别是全量处理和导出。这些配置不会自动启动任务。

```bash
PYTHONPATH=.:src .venv/bin/python -m pytest -q data_pipeline/tests tests/tracking
```
