"""各数据源的目录 → CSV 适配器；原生动作格式的解码在 stages/motion 中扩展。"""
from data_pipeline.adapters.aistpp import build_csv

BUILDERS = {'aistpp': build_csv}
