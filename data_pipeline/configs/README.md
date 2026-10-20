# 配置

- `aistpp_full.yaml`：读取AIST根目录生成的pairs.csv，全量原型处理。
- `aistpp_smoke.yaml`：同一CSV里选一条，验证/阅读代码。
- `person_video.yaml`：独立RGB人物裁剪组件，硬性输出512×512。

先 `prepare --source aistpp --root data/AIST++`，再 `run --config ...`。
新来源只需提供同格式CSV及相应motion_format解码器，不要求下载审计或历史manifest。
原始数据只读，处理输出独立存放。当前neutral SMPL配置不是正式生产标签。
