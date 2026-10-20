"""Debug one sample using exactly the same pipeline as the batch command."""
import argparse
import json
from pathlib import Path

from data_pipeline.pipeline import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['aistpp', 'finedance', 'gdance'])
    parser.add_argument('--sequence')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    config_name = 'aistpp_motion.yaml' if args.dataset == 'aistpp' else f'{args.dataset}.yaml'
    config = project / 'configs/data_pipeline' / config_name
    result = run(config, limit=1, sequence=args.sequence, output=args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
