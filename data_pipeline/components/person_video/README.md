# 人物视频组件：RGB → 512×512

输入普通单镜头视频；YOLO11n检测、ByteTrack固定主体、平滑中心、整段固定缩放。
不需要人体/动作标注。不分割镜头、不做ReID，长遮挡拒绝，不偷偷切换人物。

```python
from pathlib import Path
from data_pipeline.components.person_video import PersonVideoConfig, preprocess_person_video

config = PersonVideoConfig(weights=str(Path('models/person_detection/yolo11n.pt').resolve()), device='4')
result = preprocess_person_video(Path('/path/input.mp4'), Path('/path/output/clip'), config)
```

第二个参数就是该视频的输出文件夹，不再产生哈希路径、locks、attempts或manifest。
输出 `person.mp4`（512×512/默认30FPS）、`metadata.npz`（PTS、帧号、裁剪框、正逆仿射）、
`detections.json`、`settings.json` 和简单质量指标。预览可选，不是训练输入。

配对pipeline会传入动作对应的源视频PTS，共用一份时间轴。默认采用原片PTS最近邻，
不会按视频帧号假定精确60FPS；未知同步关系需要调用方先解决。

已有输出默认复用；参数改变需要新目录或 `force=True`。仅调整裁剪参数时复用检测；
源视频/权重正常修改会通过路径及mtime发现，不读取全文件哈希。不能并发写同一目录。
源动作在原路径被替换时，pipeline仍需要显式 `--force`。

CLI（从OMG根目录）：

```bash
PYTHONPATH=.:src:data/processed/person_crop/runtime_deps YOLO_AUTOINSTALL=false \
.venv/bin/python -m data_pipeline.cli crop-video \
  --video /path/input.mp4 --output-root data/processed/person_video --device 4
```

CLI会在output-root下按视频名建文件夹。配置见 `configs/person_video.yaml`。
额外依赖保持独立，不安装/升级训练环境。旧审计和多级缓存代码已删除。

裁剪变换改变的是图像坐标，不改变3D动作；投影需 `K_crop=A_frame@K_source`。
输出不带音频、使用离线未来帧平滑，原片出画或被遮挡的身体无法恢复。

## 裁剪策略与限制

默认`imgsz=640`、person类别0、置信度0.1，在每个输出时刻检测；默认30FPS。
选择出现最多的固定track ID，多人场景应显式指定主体；最长轨迹不保证是动作演员。
中心按0.15秒Gaussian平滑，正方形边长采用整段最大检测尺寸×1.3，避免逐帧缩放抖动。
窗口约束优先包含检测框和安全余量，越出原图时补黑边；这不保证包含全部人体轮廓。
超过0.5秒的跟踪缺口、首尾漏检会失败，不自动合并不同ID或删除动作首尾。
缓慢变焦、镜头切分和ReID尚未实现，互联网多镜头视频需要额外预处理。

## AIST人物占比与压缩依据

历史统计覆盖全部本地1,408个c01视频，分辨率均为1920×1080；排除45条ignore后，
1,356条共有964,945个有效标注帧。使用官方COCO-17关键点置信度≥0.3、至少12点、
至少一个头部点及双脚踝有效作为门槛。包围框只是关键点范围，不是完整人体检测框或分割面积。

| 人物关键点框/原图 | P10 | 中位数 | P90 |
|---|---:|---:|---:|
| 宽度 | 5.89% | 8.75% | 13.39% |
| 高度 | 29.35% | 33.98% | 38.89% |
| 面积 | 1.90% | 2.94% | 4.75% |

中位框约168×367像素；有效帧双手腕同时可信的比例82.8%，仍可能低估手部范围。
另7条breaking ch10未满足统计门槛，不代表原片无人，也不能据此删除序列。
用有效框并集×1.2估算固定方形裁剪时，1,270/1,356条（93.7%）可放入原图，
边长P10/P50/P90约461/523/705像素。这只是关键点统计，不是当前YOLO裁剪的验收通过率。
原始统计产物目录已清理，可用`data_pipeline.visualization.analyze_aist_person_scale`
重新统计（需要`archives/keypoints2d.zip`，`--output`指定新目录）。

最初RGB-only实测样本`gJB_sFM_c01_d07_mJB4_ch05`约32秒：原片41,107,736字节，
旧版512×512/30FPS裁剪989,346字节，主体track覆盖959/959帧，平均置信度0.884。
这是历史单样本结果，不是当前版本的统一压缩率或速度基准；体积下降同时来自裁剪、
降帧率、分辨率和H.264参数。原片约6.13秒处有人体出画，裁剪不能恢复缺失身体。
1920×1080→512×512的像素数减少87.4%，不等于文件体积减少87.4%。

## 模型与环境

权重为`models/person_detection/yolo11n.pt`（约5.6 MB），
[下载地址](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)，
使用时遵守[Ultralytics许可](https://www.ultralytics.com/license)。
当前隔离依赖中的主要版本为`ultralytics==8.3.221`、`lap==0.5.12`、
`torchvision==0.21.0+cu124`；OMG训练环境Torch未因此升级。
每个worker复用网络权重，新视频前重置ByteTrack状态；不把上一条视频的ID和运动状态带入下一条。
