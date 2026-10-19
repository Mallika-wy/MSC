"""Public adapter for the pinned official OMG LeRobot release."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from msc._omg.lerobot_dataset import LeRobotG1MotionDataset, LEROBOT_REVISION, LEROBOT_MANIFEST_SHA256
from msc.models.representation import ASSETS


class OMGDataset(Dataset):
    def __init__(self, dataset_root, split="train", *, limit_size=None, use_text=True, use_audio=True):
        root = Path(dataset_root)
        info = json.loads((root / "meta/info.json").read_text())
        if use_audio and tuple(info["features"]["omg.audio.feature"]["shape"]) != (35,):
            raise ValueError("Expected official OMG 35D audio features")
        self.reader = LeRobotG1MotionDataset(dataset_root=root, split=split,
            limit_size=limit_size, kinematics_path=str(ASSETS / "robots/g1/g1_kinematics.json"),
            use_text=use_text, use_audio=use_audio, use_human_motion=False,
            num_prev_states=5, canonical_frame_idx=4, rotation_representation="rot6d",
            fps=30, sequence_duration=1.0)
        # Official release metadata does not identify the extractor; never guess from dimension.
        self.audio_metadata = dict(backend="omg_precomputed", feature_type="unknown",
            dimension=35, fps=30, preprocessing_version=None, repo_id=self.reader.repo_id,
            revision=LEROBOT_REVISION, manifest_sha256=LEROBOT_MANIFEST_SHA256)

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        item = self.reader[index]
        if not item["mask"]["valid"].any():
            raise ValueError("Empty future sequence")
        return dict(motion_history=item["history_features"], motion_future=item["motion_features"],
            text=item["caption"], has_text=item["has_text"], audio_features=item["audio_features"],
            motion_mask=item["mask"]["valid"], audio_mask=item["mask"]["has_audio"] & item["mask"]["valid"])
