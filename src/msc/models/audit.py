"""Count actual parameters of the configured model, without reading a dataset."""
import argparse
import json
from pathlib import Path
from msc.config import load_config
from msc.factory import build_model


def parameter_report(model):
    groups = {}
    for name, parameter in model.named_parameters():
        parts = name.split('.')
        if parts[0] == 'text_encoder':
            group = 'frozen_t5'
        elif parts[:2] == ['motion_generator', 'layers']:
            branch = parts[3]
            group = 'expert.layers.' + ('cross_attention' if branch in {'query', 'key', 'value', 'cross_out'} else branch)
        elif parts[0] == 'motion_generator':
            group = 'expert.' + parts[1]
        elif parts[:2] == ['condition_bridge', 'adapters']:
            group = 'adapters.' + parts[2]
        else:
            group = '.'.join(parts[:2])
        entry = groups.setdefault(group, dict(total=0, trainable=0))
        entry['total'] += parameter.numel()
        entry['trainable'] += parameter.numel() if parameter.requires_grad else 0
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total=total, trainable=trainable, frozen=total-trainable, modules=groups,
                offline_encoders=['MERT-v1-95M', 'V-JEPA (Pending; disabled)'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/msc_v3_2xa6000.yaml')
    parser.add_argument('--output')
    args = parser.parse_args()
    config = load_config(args.config)
    config.model.dynamic_stats_path = None  # Statistics do not affect parameter counts.
    report = parameter_report(build_model(config))
    assert 285_000_000 <= report['trainable'] <= 315_000_000, report['trainable']
    result = json.dumps(report, indent=2)
    print(result)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result + '\n')


if __name__ == '__main__':
    main()
