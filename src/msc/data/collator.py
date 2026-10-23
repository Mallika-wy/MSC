from torch.utils.data import default_collate
import torch


def collate_motion(samples):
    batch = {}
    variable = {f'{m}_{field}' for m in ('text','audio','video')
                for field in ('tokens','features','mask','times','intervals')}
    if any('text' in s for s in samples):
        batch['text'] = [s.get('text','') for s in samples]
    for key in set.union(*(set(s) for s in samples)) - variable - {'text'}:
        if all(key in s for s in samples):
            batch[key] = default_collate([s[key] for s in samples])
    for modality in ('text','audio','video'):
        key = next((k for k in (modality+'_tokens',modality+'_features') if any(k in s for s in samples)),None)
        if key is None:
            continue
        example = next(s[key] for s in samples if key in s)
        length = max(s[key].shape[0] for s in samples if key in s)
        shape = (len(samples),length,*example.shape[1:])
        values = example.new_zeros(shape)
        mask = torch.zeros(shape[:-1],dtype=torch.bool)
        time_key = modality+('_times' if modality=='audio' else '_intervals')
        times = torch.zeros((len(samples),length,2) if modality=='video' else (len(samples),length))
        for i,s in enumerate(samples):
            if key not in s:
                continue
            n = len(s[key])
            values[i,:n] = s[key]
            mask[i,:n] = s[modality+'_mask']
            if modality != 'text':
                if time_key not in s:
                    raise ValueError(f'{time_key} is required')
                times[i,:n] = s[time_key]
        batch[key], batch[modality+'_mask'] = values,mask
        if modality != 'text':
            batch[time_key] = times
    return batch
