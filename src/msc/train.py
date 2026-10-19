import argparse
from msc.config import load_config
from msc.factory import seed_everything, build_dataset, build_model
from msc.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train independent MSC flow matching")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="section.key=value")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.train.seed)
    dataset = build_dataset(cfg)
    model = build_model(cfg, dataset.audio_metadata)
    Trainer(model, dataset, cfg).fit()


if __name__ == "__main__":
    main()
