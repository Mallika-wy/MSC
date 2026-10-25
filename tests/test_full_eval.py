"""Whole-clip correctness checks; synthetic fixtures do not establish model quality."""
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
import torch

from data_pipeline.release.training_index import TrainingMotion
from msc.config import Config
from msc.eval_full import configure_dataset, check_feature_contract
from msc.evaluation.rollout import (
    MotionCase, covers_full_motion, select_motions, generate_full_motion,
    reanchor_previous, trajectory_metrics,
    condition_arguments,
)
from msc.motion.codec import G1MotionFeatureCodec
from msc.motion.kinematics import G1Kinematics
from msc.motion.rotations import quaternion_apply, quaternion_to_matrix


def motion(name, frames):
    return TrainingMotion(name, name, 'actor0', 'source_g1', 'unused.parquet', 0, frames)


class SelectionDataset:
    def __init__(self, reverse=False):
        self.records = {}
        rows = []
        for i in range(20):
            item = motion(f'motion-{i:02d}', 65 + i)
            spans = [(5, item.num_frames - 29)]
            if i == 19:
                spans = [(5, 6), (40, item.num_frames - 29)]  # condition gap
            records = [(0, item, 'text', lo, hi, ('text', '', f'action {i}')) for lo, hi in spans]
            self.records[item.motion_id] = records
            rows += [dict(release_index=0, motion_id=item.motion_id, task='text',
                          start=lo, end=hi, num_frames=item.num_frames) for lo, hi in spans]
        table = pa.Table.from_pylist(rows[::-1] if reverse else rows)
        for name in ('motion_id', 'task'):
            table = table.set_column(table.schema.get_field_index(name), name, table[name].dictionary_encode())
        self.rollout_spans = SimpleNamespace(table=table)

    def motion_spans(self, ri, mid, task):
        return self.records[mid]


def test_seeded_selection_uses_distinct_full_motions_and_rejects_gaps():
    a = select_motions(SelectionDataset(), 10, 42)
    b = select_motions(SelectionDataset(reverse=True), 10, 42)
    ids = [case.motion.motion_id for case in a]
    assert ids == [case.motion.motion_id for case in b]
    assert len(set(ids)) == 10
    assert 'motion-19' not in ids
    assert ids != [case.motion.motion_id for case in select_motions(SelectionDataset(), 10, 43)]
    assert covers_full_motion([(5, 18, None)], 47)
    assert not covers_full_motion([(5, 6, None), (40, 55, None)], 84)
    with pytest.raises(ValueError, match='Only 19'):
        select_motions(SelectionDataset(), 20, 42)


class ConditionOnlyDataset:
    roots = ['unused']

    def _init_codec(self):
        pass

    def _condition(self, sample, ri, task, frame, condition):
        sample['text'] = condition[2]


class RecordingModel:
    def __init__(self):
        self.calls = []

    def sample(self, history, **kwargs):
        self.calls.append(dict(history=history.clone(), noise=kwargs['initial_noise'].clone(),
                               rtc=kwargs['rtc_mode'], prior=kwargs['previous_motion']))
        pred = history[:, -1:].expand(-1, 30, -1).clone()
        pred[..., :3] = 0
        pred[..., 0] = torch.arange(1, 31) / 100
        pred[..., 9] += torch.arange(1, 31) / 50
        return pred


@pytest.mark.parametrize('frames', [35, 47, 300])
def test_complete_duration_tail_seed_and_no_future_gt_injection(monkeypatch, frames):
    import msc.evaluation.rollout as module
    gt = np.zeros((frames, 36), np.float32)
    gt[:, 3] = 1
    gt[:, 2] = .8
    gt[:, 0] = np.arange(frames) / 500
    gt[:, 7] = np.sin(np.arange(frames) / 10)
    monkeypatch.setattr(module, 'read_motion', lambda *args: (gt.copy(), np.arange(frames) / 30))
    case = MotionCase(0, motion('test', frames), 'text', [(5, frames - 29, ('text', '', 'move'))])
    off, on = RecordingModel(), RecordingModel()
    a, chunks = generate_full_motion(off, ConditionOnlyDataset(), case, seed=42, device='cpu')
    b, rtc_chunks = generate_full_motion(on, ConditionOnlyDataset(), case, seed=42, device='cpu', rtc=True)
    assert a['predicted'].shape == b['predicted'].shape == gt.shape
    np.testing.assert_array_equal(a['predicted'][:5], gt[:5])
    assert sum(c['generated_frames'] for c in chunks) == frames - 5
    assert chunks[-1]['generated_frames'] == ((frames - 6) % 15) + 1
    assert all(not c['rtc'] for c in chunks)
    assert not rtc_chunks[0]['rtc'] and all(c['rtc'] for c in rtc_chunks[1:])
    for first, second in zip(off.calls, on.calls):
        torch.testing.assert_close(first['noise'], second['noise'])
    gt[5:, :3] += 10  # changing all GT future positions cannot affect generation
    c, _ = generate_full_motion(RecordingModel(), ConditionOnlyDataset(), case, seed=42, device='cpu')
    np.testing.assert_allclose(a['predicted'], c['predicted'], atol=0, rtol=0)
    assert trajectory_metrics(a['predicted'], a['target'])['evaluated_frames'] == frames - 5


def test_reanchor_preserves_world_rotation_joints_and_predicted_links():
    codec = G1MotionFeatureCodec(G1Kinematics(), num_prev_states=5,
                                canonical_frame_idx=4, rotation_representation='rot6d')
    features = torch.randn((1, 30, 125), generator=torch.Generator().manual_seed(7))
    old_pos, new_pos = torch.tensor([[[1., 2., .8]]]), torch.tensor([[[-1., -3., .7]]])
    old_quat = torch.tensor([[[.9238795, 0., 0., .3826834]]])
    new_quat = torch.tensor([[[.7071068, 0., 0., -.7071068]]])
    result = reanchor_previous(codec, features, 15, old_pos, old_quat, new_pos, new_quat)
    shifted = torch.cat((features[:, 15:], features[:, -1:].expand(-1, 15, -1)), 1)
    before, after = codec.split_features(shifted), codec.split_features(result)
    a = codec.decode_to_world_qpos36(before, old_pos, old_quat)
    b = codec.decode_to_world_qpos36(after, new_pos, new_quat)
    torch.testing.assert_close(a[..., :3], b[..., :3], atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(a[..., 7:], b[..., 7:])
    torch.testing.assert_close(quaternion_to_matrix(a[..., 3:7]), quaternion_to_matrix(b[..., 3:7]), atol=3e-6, rtol=3e-6)
    def world_links(comp, pos, quat):
        _, heading, _ = codec._canonical_anchor(pos, quat)
        return quaternion_apply(heading.unsqueeze(-2).expand(1, 30, 29, 4), comp.body_link_pos_local) + pos.unsqueeze(-2)
    torch.testing.assert_close(world_links(before, old_pos, old_quat), world_links(after, new_pos, new_quat), atol=3e-6, rtol=3e-6)


def test_explicit_dataset_does_not_inherit_an_unrelated_training_manifest(monkeypatch):
    import msc.eval_full as module
    monkeypatch.setattr(module, 'build_dataset', lambda cfg: cfg.data)
    cfg = Config()
    cfg.data.motion_manifest = 'training-subset.json'
    cfg.data.max_motions_per_task = 100
    args = SimpleNamespace(data_root=['different-release'], motion_manifest=None, split='train', task='text')
    data = configure_dataset(cfg, args)
    assert data.roots == ['different-release'] and data.motion_manifest is None
    assert data.max_motions_per_task is None and data.max_windows_per_motion is None
    args.motion_manifest = 'selected.json'
    assert configure_dataset(cfg, args).motion_manifest == 'selected.json'


def test_video_caption_wrap_preserves_full_english_and_chinese_text():
    from msc.evaluation.render import wrap_caption, caption_panels
    image_font = pytest.importorskip('PIL.ImageFont')
    font = image_font.truetype('DejaVuSans.ttf', 16)
    text = 'A person raises the left arm and turns around. ' * 12 + '抬起左手，然后转身。'
    lines = wrap_caption(text, font, 280)
    assert len(lines) > 1 and all(font.getlength(line) <= 280 for line in lines)
    assert ''.join(''.join(lines).split()) == ''.join(text.split())
    panels, height = caption_panels([text, 'Another caption'], 512, .5)
    assert height % 2 == 0
    assert panels[text].shape == panels['Another caption'].shape == (height, 512, 3)


def test_audio_tail_reads_conditions_without_requiring_a_full_gt_window():
    class AudioDataset:
        initialized = False

        def _init_codec(self):
            self.initialized = True

        def _condition(self, sample, ri, task, frame, condition):
            assert self.initialized and frame == 46 and task == 'audio'
            sample.update(audio_tokens=torch.zeros(2, 768), audio_mask=torch.ones(2, dtype=torch.bool),
                          audio_times=torch.tensor([-.1, 0.]))
    case = MotionCase(0, motion('audio', 47), 'audio', [(5, 18, ('audio', 'link', 'feature'))])
    model = SimpleNamespace(audio_metadata={'features': ['verified-placeholder']})
    args, condition = condition_arguments(AudioDataset(), case, 46, model, 'cpu')
    assert args['audio_tokens'].shape == (1, 2, 768)
    assert args['audio_metadata'] is model.audio_metadata
    assert condition == ('audio', 'link', 'feature')


def test_feature_contract_allows_a_known_dataset_subset_but_not_another_encoder():
    descriptor = ['audio', 'mert', 'checkpoint-hash', 'preprocess-hash', [768]]
    data = SimpleNamespace(feature_metadata={'features': [descriptor]})
    model = SimpleNamespace(feature_metadata={'features': [descriptor]})
    check_feature_contract(data, model, 'audio')
    model.feature_metadata = {'features': [['audio', 'different', 'other-hash', 'preprocess-hash', [768]]]}
    with pytest.raises(ValueError, match='provenance'):
        check_feature_contract(data, model, 'audio')
