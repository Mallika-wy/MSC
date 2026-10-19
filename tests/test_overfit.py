import json
import pytest
import torch
from msc.config import load_config
from msc.factory import build_dataset, build_model, seed_everything
from msc.data.collator import collate_motion


@pytest.mark.slow
def test_synthetic_overfit(tmp_path):
    cfg = load_config("configs/overfit.yaml")
    seed_everything(42)
    dataset = build_dataset(cfg)
    model = build_model(cfg, dataset.audio_metadata)
    batch = collate_motion([dataset[i] for i in range(4)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=0)
    fixed_noise = torch.randn(4, 30, 125)
    fixed_times = torch.tensor([0.2, 0.4, 0.6, 0.8])
    model.eval()
    with torch.no_grad():
        initial = model.compute_loss(batch, noise=fixed_noise, times=fixed_times).item()
    for step in range(600):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        model.compute_loss(batch).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final = model.compute_loss(batch, noise=fixed_noise, times=fixed_times).item()
        motion = model.sample(batch["motion_history"][:1])
    report = dict(initial=initial, final=final, ratio=final / initial, steps=600)
    (tmp_path / "overfit.json").write_text(json.dumps(report))
    print("OVERFIT", json.dumps(report))
    assert final < initial * 0.2
    assert torch.isfinite(motion).all()
