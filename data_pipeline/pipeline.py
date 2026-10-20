"""One processing path: source -> GMR -> tracker/video -> MERT -> native release."""
from itertools import islice
import traceback

from data_pipeline.config import Settings
from data_pipeline.io import sha256_file, write_json
from data_pipeline.sources.audio import load_dataset


def process_sample(sample, settings, audio_cache, tracker_version, diagnostic=False):
    """Process one person; keep quality failures for inspection, never publish them."""
    from data_pipeline.motion.retarget import retarget
    from data_pipeline.motion.arm_twist import screen_reference
    from data_pipeline.motion.tracking import track

    directory = settings.output / 'samples' / sample.name
    directory.mkdir(parents=True, exist_ok=True)
    status = {
        'name': sample.name, 'sequence': sample.sequence, 'actor': sample.actor,
        'split': sample.split, 'motion': str(sample.motion), 'audio': str(sample.audio),
        'state': 'processing', 'stages': {},
    }
    report = directory / 'status.json'

    # 1. Decode the source human and retarget it to G1.
    reference, version, fingerprint = retarget(
        sample, settings.dataset, settings.retarget, directory / 'reference')
    quality = screen_reference(reference, settings.quality)
    status['stages']['reference'] = dict(path=str(reference.resolve()), version=version, quality=quality)
    status['processing'] = fingerprint
    write_json(report, status)

    # 2. Execute accepted references; diagnostics can also run rejected references.
    if quality['accepted'] or diagnostic:
        execution = track(reference, version, settings.tracker, settings.quality,
                          tracker_version, directory / 'tracking')
        execution['diagnostic_only'] = not quality['accepted']
        status['stages']['tracking'] = execution
        write_json(report, status)
    else:
        status['stages']['tracking'] = {'accepted': False, 'skipped': 'reference_rejected'}

    # 3. Encode the WAV. Multiple GDance actors share the same cached audio features.
    status['stages']['audio'] = audio_cache.extract(sample.audio, settings.output / 'audio')
    accepted = quality['accepted'] and status['stages']['tracking']['accepted']
    status['state'] = 'accepted' if accepted else 'rejected'
    write_json(report, status)
    return status


def run(config, *, limit=None, sequence=None, output=None, dry_run=False, diagnostic=False):
    """Select samples, process each independently, then publish accepted pairs."""
    settings = Settings.read(config, output)
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    samples = load_dataset(settings.dataset).samples(settings.root)
    if sequence:
        samples = (s for s in samples if s.name == sequence or s.sequence == sequence)
    selected = list(islice(samples, limit))
    if not selected:
        raise ValueError(f'No matching samples found under {settings.root}')
    if len({s.name for s in selected}) != len(selected):
        raise ValueError('Duplicate sample names')
    if dry_run:
        return dict(dataset=settings.dataset, output=str(settings.output), count=len(selected),
                    samples=[dict(name=s.name, motion=str(s.motion), audio=str(s.audio)) for s in selected])
    if (settings.output / 'release').exists():
        raise FileExistsError('Release already exists; select a new --output directory')

    from data_pipeline.audio.cache import AudioCache
    audio_cache = AudioCache(settings.audio['model'],
                             chunk_seconds=settings.audio.get('chunk_seconds', 5),
                             device=settings.audio.get('device', 'cpu'))
    tracker_version = 'holomotion:' + sha256_file(settings.tracker['onnx'])
    settings.output.mkdir(parents=True, exist_ok=True)
    write_json(settings.output / 'config.json', {
        'dataset': settings.dataset, 'root': str(settings.root),
        'retarget': settings.retarget, 'tracker': settings.tracker,
        'audio': settings.audio, 'quality': settings.quality,
    })
    results = []
    for sample in selected:
        print(f'[{settings.dataset}] {sample.name}', flush=True)
        try:
            result = process_sample(sample, settings, audio_cache, tracker_version, diagnostic)
        except Exception as error:
            # One broken recording must not discard the rest of a batch.
            directory = settings.output / 'samples' / sample.name
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / 'error.json', {
                'error': str(error), 'traceback': traceback.format_exc()})
            result = dict(name=sample.name, state='error', error=str(error))
        results.append(result)
        write_json(settings.output / 'progress.json', results)
        print(f"  {result['state']}", flush=True)

    from data_pipeline.release.audio import publish
    accepted = [r for r in results if r['state'] == 'accepted']
    release = publish(settings.dataset, accepted, settings.output) if accepted else None
    summary = dict(dataset=settings.dataset, total=len(results),
                   accepted=len(accepted),
                   rejected=sum(r['state'] == 'rejected' for r in results),
                   errors=sum(r['state'] == 'error' for r in results),
                   release=release)
    write_json(settings.output / 'summary.json', summary)
    return summary
