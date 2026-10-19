from torch.utils.data import default_collate


def collate_motion(samples):
    # Metadata is dataset-level and belongs in checkpoints, not GPU batches.
    return default_collate(samples)
