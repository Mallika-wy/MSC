"""Isolated compute candidates; never imported by the training launcher.

Run parity tests first. GPU timings require an otherwise idle GPU:
PYTHONPATH=src:. python experiments/benchmark_motion_block.py --device cuda:0
"""
import argparse
import copy
import json
import time

import torch
from torch.nn import functional as F

from msc.models.attention import normalize_qk
from msc.models.conditioning.fusion import ConditionMemory
from msc.models.motion_generator import MotionBlock


class CandidateBlock(MotionBlock):
    def __init__(self, *args, packed_kv=False, all_frames_valid=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.packed_kv = packed_kv
        # A full-model implementation must compute this once per batch, outside
        # the layer loop. Do not add a CUDA tensor -> bool sync in each layer.
        self.all_frames_valid = all_frames_valid

    def cross_attention(self, h, memory):
        if not self.packed_kv:
            return super().cross_attention(h, memory)
        batch, horizon, dim = h.shape
        def split(value):
            return value.reshape(batch, -1, self.heads, dim // self.heads).transpose(1, 2)
        # Keep key/value modules and state_dict names for checkpoint parity.
        kv = F.linear(memory.tokens, torch.cat((self.key.weight, self.value.weight)),
                      torch.cat((self.key.bias, self.value.bias)))
        k, v = kv.chunk(2, dim=-1)
        q, k, v = normalize_qk(split(self.query(h))), normalize_qk(split(k)), split(v)
        result = F.scaled_dot_product_attention(q, k, v,
            attn_mask=self.attention_bias(memory, horizon).to(q.dtype),
            dropout_p=self.dropout.p if self.training else 0.)
        return self.cross_out(result.transpose(1, 2).reshape(batch, horizon, dim))

    def forward(self, x, time, memory, mask):
        a, b, c, d, e, f, g, h, i = self.modulation(time).chunk(9, -1)
        normalized = self.norm_self(x) * (1 + b) + a
        padding = None if self.all_frames_valid else ~mask
        x = x + c * self.dropout(self.self_attn(normalized, key_padding_mask=padding))
        normalized = self.norm_cross(x) * (1 + e) + d
        x = x + f * self.dropout(self.cross_attention(normalized, memory))
        normalized = self.norm_ff(x) * (1 + h) + g
        return (x + i * self.dropout(self.ff(normalized))).masked_fill(~mask[..., None], 0)


def inputs(batch, dim, length, *, device='cpu', dtype=torch.float32, partial=False, time_frames=1):
    x = torch.randn(batch, 30, dim, device=device, dtype=dtype, requires_grad=True)
    time = torch.randn(batch, time_frames, dim, device=device, dtype=dtype, requires_grad=True)
    tokens = torch.randn(batch, length, dim, device=device, dtype=dtype, requires_grad=True)
    valid = torch.ones(batch, length, device=device, dtype=torch.bool)
    valid[:, -3:] = False
    times = (torch.arange(length, device=device, dtype=dtype)/75-.25).expand(batch, -1)
    ids = torch.full_like(valid, 2, dtype=torch.long)
    ids[:, :5] = 0
    memory = ConditionMemory(tokens, valid, times, torch.stack((times, times), -1), ids)
    mask = torch.ones(batch, 30, device=device, dtype=torch.bool)
    if partial:
        mask[0, -4:] = False
    return x, time, memory, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    if not args.device.startswith('cuda'):
        parser.error('Use pytest for CPU correctness; performance requires CUDA')
    torch.cuda.set_device(args.device)
    free, total = torch.cuda.mem_get_info()
    if total-free > 2*1024**3:
        raise SystemExit('GPU is occupied. Run this benchmark on an idle GPU; active training remains untouched.')
    torch.set_num_threads(4)
    base = MotionBlock(1024, 16, .1, 3072).to(args.device)
    with torch.no_grad():
        base.modulation[-1].weight.normal_(std=.01)
    state = copy.deepcopy(base.state_dict())
    del base
    for length in (50, 113):
        for name, packed, unmasked in [('baseline', False, False), ('packed_kv', True, False),
                                       ('self_unmasked', False, True), ('combined', True, True)]:
            block = CandidateBlock(1024,16,.1,3072,packed_kv=packed,all_frames_valid=unmasked).to(args.device)
            block.load_state_dict(state)
            if args.compile:
                block.compile(dynamic=True)
            data = inputs(args.batch_size,1024,length,device=args.device)
            def step():
                block.zero_grad(set_to_none=True)
                for value in (data[0],data[1],data[2].tokens):
                    value.grad = None
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss = block(*data).float().square().mean()
                loss.backward()
            for _ in range(10):
                step()
            torch.cuda.synchronize()
            started = time.monotonic()
            for _ in range(30):
                step()
            torch.cuda.synchronize()
            print(json.dumps(dict(variant=name,length=length,batch=args.batch_size,compiled=args.compile,
                                  forward_backward_ms=(time.monotonic()-started)*1000/30)),flush=True)
            del block, data
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
