"""Opt-in local tests. No weights or datasets are downloaded."""
import os
from pathlib import Path
import pytest
import torch
from msc.data.collator import collate_motion
from msc.factory import build_model
from msc.models.conditioning.text import T5TextEncoder

pytestmark = pytest.mark.integration


def local_path(name):
    value = os.environ.get(name)
    if not value or not Path(value).exists():
        pytest.skip(f"Set {name} to enable local integration")
    return value


def test_t5_unfreeze_gradients(tmp_path):
    # A tiny local T5 model verifies trainability without downloading another model.
    from transformers import T5Config, T5EncoderModel
    import shutil
    source = Path(local_path("MSC_TEST_T5_PATH"))
    local = tmp_path / "tiny-t5"
    local.mkdir()
    shutil.copyfile(source / "spiece.model", local / "spiece.model")
    tiny = T5EncoderModel(T5Config(vocab_size=32128, d_model=16, d_kv=8, d_ff=32, num_layers=1, num_heads=2))
    tiny.save_pretrained(local)
    encoder = T5TextEncoder(local, frozen=False, max_length=8)
    encoder.train()
    features, mask = encoder(["walk forward"], torch.device("cpu"))
    features.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.encoder.parameters())
    with pytest.raises(ValueError, match="Static text cache"):
        T5TextEncoder(local, frozen=False, cache_dir=tmp_path / "cache")


def test_frozen_t5_and_manifest_cache(tmp_path,monkeypatch):
    """Real frozen T5 correctness; the captions are fixtures, not a real Release."""
    import json
    import sys
    from msc.models.msc_model import MSCModel
    from msc import cache_text
    encoder=T5TextEncoder(local_path('MSC_TEST_T5_PATH'),frozen=True,max_length=8)
    captions=tmp_path/'captions.txt'
    captions.write_text('walk forward\nraise the left arm\nwalk forward\n')
    cache=tmp_path/'cache'
    monkeypatch.setattr(cache_text,'T5TextEncoder',lambda *a,**k:encoder)
    monkeypatch.setattr(sys,'argv',['msc-cache-text','--t5-path',local_path('MSC_TEST_T5_PATH'),
        '--captions',str(captions),'--cache-dir',str(cache),'--max-length','8'])
    cache_text.main()
    manifest=json.loads((cache/'manifest.json').read_text())
    assert manifest['encoder']['output_dim']==encoder.output_dim
    assert len(manifest['entries'])==2
    model=MSCModel(hidden_dim=32,num_layers=1,num_heads=4,text_dim=encoder.output_dim,
                   text_encoder=encoder,use_audio=False,text_dropout=0)
    model.train()
    assert not encoder.encoder.training
    batch=dict(motion_history=torch.randn(2,5,125),motion_future=torch.randn(2,30,125),
               motion_mask=torch.ones(2,30,dtype=torch.bool),text=['walk forward','raise the left arm'])
    loss=model.compute_loss(batch)
    loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in encoder.encoder.parameters())
    assert model.text_adapter[1].weight.grad.abs().sum()>0
