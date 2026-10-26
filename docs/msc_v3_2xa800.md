# 双 A800 80GB 训练

使用 `bash train_a800.sh`。该脚本复用 `train.sh` 的所有环境变量与
`section.key=value` 覆盖机制，原有 A6000 启动方式保持可用。
配置文件为 `configs/msc_v3_2xa800.yaml`。

默认使用 GPU 0、1，每卡 batch 2048，全局 batch 4096，梯度累积 1。
默认采样每轮先打乱 episode 顺序，同时维护 64 个活跃 episode；episode 内的
窗口也打乱，每次取 8 个后轮换，耗尽后补入下一个。按现有 dataset index
无放回覆盖全部窗口，保留所有 annotation/task 的原始窗口重数，不引入任务配比。
全局 batch 4096 按 rank 切成两份，每卡 2048，大约各包含四轮 64×8 的访问。
采样顺序与原全局随机不同，不声明训练效果等价。checkpoint 记录采样 version=5
和 64/8 参数，并拒绝将旧采样方式的 checkpoint 当作同一实验恢复。

保留 12 层、hidden 1024、16 heads、FFN 3072 的模型结构，BF16、fused AdamW、
Expert 编译和 BF16 梯度通信。峰值学习率 1e-4，warmup 1000，cosine 下限 1e-6，
共 60000 optimizer steps。批量增大后保持原学习率，未自动按 batch 放大学习率。
训练样本预算随全局 batch 增大；步数保持当前单卡脚本的 60000。

每卡 8 个 DataLoader worker，每 worker 缓存 512 MiB，CPU 算子线程 4。
每 10 步记录指标，每 2000 步验证，每 10000 步保存，并保存最终 checkpoint。
使用 Text 和离线 Audio 条件，关闭 Video、RTC 和动力学辅助损失。
NCCL 启用此次容器已验证的 PCIe P2P，关闭 IB；容器 `/dev/shm` 为 120 GiB。
600 MiB BF16 AllReduce 实测：P2P 约 28.9 ms，共享内存路径约 90.7 ms。

## 环境

宿主机 OMG 环境 `/HOME/zju_jchli/zju_jchli_1/HDD_POOL/home/user/envs/omg`
实际指向 `/XYAIFS00`，此次容器只挂载 `/XYFS02`，无法直接执行原环境。
通过 conda-pack 创建项目专用副本 `.runs/envs/omg`，解包后运行 `conda-unpack`。
原 OMG 环境不作修改。打包时跳过失效的 editable 安装引用和缺失文件检查：
原环境 pip 26.1.2 的 conda 元数据记录了不存在的 `python3.1` 路径；
实际使用 `python -m pip` 与项目 `PYTHONPATH`。

环境检查保存在 `.runs/setup/versions.json`，准备日志位于
`.runs/setup/prepare_container.log`。启动脚本将此环境的 `bin` 加入 PATH，
使用系统 GCC 11.4 编译 Triton 扩展，所需 CUDA 运行库由 PyTorch 环境提供。
容器提供的 CUDA module 仅有旧版 CUDA 9.0，因此不加载该 module。
本地 T5、Hugging Face 离线模式和 W&B offline 日志均无需下载或登录。

```bash
# 以下命令在容器内、项目根目录执行。
bash train_a800.sh --cache
bash train_a800.sh --normalization-stats
bash train_a800.sh --stats
bash train_a800.sh

# 临时覆盖参数：
BATCH_SIZE=1536 OUTPUT_DIR=.runs/my-a800-run bash train_a800.sh
```

GPU 数量由 `CUDA_VISIBLE_DEVICES` 推导，可用 `NUM_GPUS` 覆盖。
默认输出 `.runs/msc-v3-flow-only-2xa800-b2048`，编译缓存
`.runs/compiler-cache-a800`。动力学统计 `.runs/msc-v3-dynamics-stats.json`
只与训练数据池相关，改变数据池后需重新生成。
125D 归一化使用 `.runs/msc-v3-normalization-stats.json`，按实际 Train 窗口、
5 帧历史、参考帧 4、30 帧未来精确计算。新统计带有数据池指纹，训练时自动校验。
保留旧内置统计供历史 checkpoint 使用；新训练不复用其 10/60 帧统计。

## 本次容量测试（2026-10-26）

容器 `deeplearni-2693439`，两张 NVIDIA A800 80GB PCIe，无 NVLink，
驱动 595.45.04；Python 3.10.20、PyTorch 2.6.0+cu124、Triton 3.2.0、
Transformers 4.57.6、PyArrow 21.0.0、W&B 0.26.1。

容量测试使用真实窗口的 motion/音频，重复样本并将音频扩展到 113 tokens
（75Hz、最长 1.5 秒条件窗口向上取整），文本截断到 50 tokens。
分别执行纯音频、各半混合、纯文本输入，各 3 次完整前向、反向、AdamW 更新。
测试保留冻结 T5；重复文本会经过现有文本去重逻辑。
这些是单 GPU 容量测量，双卡实际训练指标另行记录。

| 每卡 batch | 长音频峰值 allocated | 混合峰值 allocated | 长文本峰值 allocated |
| --- | --- | --- | --- |
| 1792 | 61.60 GiB | 62.34 GiB | 50.39 GiB |
| 2048（采用） | 69.78 GiB | 70.63 GiB | 56.97 GiB |

2048 混合输入 reserved 峰值 71.11 GiB，给 DDP、CUDA context 和长度变化保留余量。
预热后的单 GPU 更新约 0.82–0.87 秒（音频）、0.85 秒（混合）、0.65 秒（文本）；
重复样本、无 DataLoader，不能作为真实随机采样双卡训练的吞吐。
原始日志：`.runs/capacity_a800_b1792.log`、`.runs/capacity_a800_b2048.log`；
容量脚本：`.runs/capacity_a800.py`。

当前 Train 池为 68,302,262 窗口（Text 62,777,855，Audio 5,524,407）。
60000 steps、全局 batch 4096 处理约 2.458 亿样本，即约 3.60 轮。


数据缓存 `.runs/data-cache` 是原 Parquet 数据的 NPY 派生副本，覆盖 Train 和 Val。
按源文件大小和修改时间在首次打开时校验，源不匹配时回退原始读取；不改变数据池指纹或采样顺序。
缓存前后相同 2048 个真实样本的输入张量逐项一致。
单 worker 准备该 batch 的 cProfile 总时间由 29.39 秒降至 2.72 秒。
未命中缓存时，新的窗口读取器仍保证同一 batch 的每个 row group 只解码一次，
即使 LRU 容量为零，也不会因随机窗口顺序重复解码。


固定动作评测使用 `.runs/fixed-eval/text.json` 和 `audio.json`：各 10 条 Val 完整动作，
seed 42，逐动作噪声种子固定。`eval_a800.sh` 对生成结果的 selection 进行核对。
使用同等训练量的基线与新采样 checkpoint 分别评测，比较数值指标及完整动作视频；
当前初始化阶段的短训练仅用于性能与链路验证，不能据此证明质量未退化。

```bash
CHECKPOINT=/path/to/checkpoints/step_10000.pt TASK=text \
  OUTPUT_DIR=.runs/fixed-text-eval bash eval_a800.sh
CHECKPOINT=/path/to/checkpoints/step_10000.pt TASK=audio \
  OUTPUT_DIR=.runs/fixed-audio-eval bash eval_a800.sh
```

环境副本另外安装了同版本 `opencv-python-headless==4.11.0.86`，以适配容器无 libGL
的情况；FFmpeg/FFprobe 链接到共享目录中已有的静态二进制。原 OMG 环境保持不变。


## 数组缓存与交错采样的吞吐（2026-10-26）

固定两卡、每卡 batch 2048、64/8、同一 seed 和同一批次序列，每组运行 32 步，
使用日志 step 16–32 的五个区间平均值；不启用 CUDA profiler。

| 每卡 workers | 稳定秒/step | 全局样本/秒 | 每卡 allocated 峰值 |
| --- | --- | --- | --- |
| 4 | 1.0147 | 4037 | 75.73 GiB |
| 8（默认） | 1.0046 | 4077 | 75.73 GiB |
| 16 | 1.0065 | 4069 | 75.73 GiB |

4 与 8 的 step loss 最大差异约 8.94e-8；只改变并发读取数，不改变采样序列。
16 workers 没有稳定吞吐收益，增加启动开销。该短测量不保证全程速度，
正式训练继续记录 interval 吞吐与每卡显存。

全局随机对照在数组缓存和 P2P 下稳定约 1.40 秒/step；原读取方式长测中，
预取耗尽后单个区间可达到 7–12 秒/step，首次 6 步的 1.47 秒不能代表稳态。
原始日志 `.runs/bench_a800_interleaved_w{4,8,16}.log`，逐步指标及完整 checkpoint
位于相应 `.runs/bench-a800-interleaved-w*/`。

完整 CPU 回归：255 passed、6 skipped。CUDA BF16 和编译由真实双卡训练验证，
完整动作质量仍须使用训练后 checkpoint 与相同训练量的基线进行固定评测。


最终 64/8 + 8 workers 设置的低开销 CUDA event 计时（最后 16 步平均）：
H2D 61.2 ms，T5（含分词）27.6 ms，条件桥 33.7 ms，Expert 前向和 Flow loss
269.9 ms，反向含 DDP 592.0 ms，梯度裁剪 4.9 ms，AdamW 5.7 ms。
事件区间是当前 CUDA stream 的经过时间，包含其依赖等待；通信与反向可能重叠，
不要再次将独立 AllReduce 时间加到该反向区间。详见
`.runs/profile-a800-final-events/events_rank{0,1}.json`。

正式后台输出：`.runs/msc-v3-flow-only-2xa800-b2048`，日志 `train.log`，
进程号 `launcher.pid`，容器标识 `container.txt`。通过 `nohup` 启动，SSH 断开后继续；
容器被回收会终止训练。checkpoint 每 10000 步保存并保留，最后一步也保存。

```bash
tail -f .runs/msc-v3-flow-only-2xa800-b2048/train.log
```


训练启动验证：容器 `deeplearni-2693439`，launcher PID 11731，已确认正常完成前 30 步；
step 20–30 区间约 1.02 秒/step、4014 样本/秒，每卡 allocated 峰值 75.73 GiB。
新采样器在 epoch 内无放回；DDP 沿用轮末补齐规则，当前末批补齐 2634 个窗口
（约占该轮 0.0039%），仅为两 rank 保持相同 batch 大小，补齐后再进入下一轮。
