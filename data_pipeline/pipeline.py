"""Discover sequences, process independent modalities, and assemble one release."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import asdict
from itertools import islice
import json
from multiprocessing import get_context
from pathlib import Path
import traceback
import wave

from data_pipeline.config import Settings
from data_pipeline.io import sha256_file, write_json
from data_pipeline.sources import discover
from data_pipeline.storage.store import Store, digest

_worker_context = None


def _record(value):
    return json.loads(json.dumps(asdict(value), default=str))


def _init_worker(settings, tracker_version, diagnostic, inference=None):
    global _worker_context
    from data_pipeline.motion import sonic
    if inference is not None:
        from data_pipeline.motion.inference import connect_worker
        sonic._shared_policy = connect_worker(inference, settings.tracker['observation_config'])
    else:
        sonic._shared_policy = None
    sonic.load_policy.cache_clear()
    encoder = None
    if settings.audio['enabled']:
        from data_pipeline.audio.processor import AudioCache
        encoder = AudioCache(settings.audio['model'], chunk_seconds=settings.audio.get('chunk_seconds', 5),
                             device=settings.audio.get('device', 'cpu'))
    _worker_context = settings, tracker_version, diagnostic, encoder


def process_sequence(sequence, settings, tracker_version, diagnostic=False, audio_encoder=None):
    from data_pipeline.motion.processor import process_motion
    from data_pipeline.audio.processor import process_audio
    from data_pipeline.video.processor import process_video
    motions, assets = [], []
    sequence_record = _record(sequence)
    for source in sequence.motions:
        try:
            artifacts = process_motion(source, sequence.source_dataset, settings,
                                       diagnostic=diagnostic, tracker_version=tracker_version)
            required = settings.release['publish_labels']
            accepted_labels = {a.label_type for a in artifacts if a.quality.get('accepted')}
            accepted = all(label in accepted_labels for label in required)
            motions.append(dict(source=_record(source), artifacts=[asdict(a) for a in artifacts],
                                state='accepted' if accepted else 'rejected'))
        except Exception as error:
            motions.append(dict(source=_record(source), artifacts=[], state='error', error=str(error),
                                traceback=traceback.format_exc()))
    if any(m['state'] == 'accepted' for m in motions):
        for asset in sequence.assets:
            if not getattr(settings, asset.modality, {}).get('enabled', False):
                continue
            entry = dict(source_id=asset.asset_id, path=str(asset.path), modality=asset.modality,
                         checksum=digest(asset.path))
            if asset.modality == 'audio':
                artifact = process_audio(asset, audio_encoder, settings.output / 'cache/audio')
                entry['feature'] = asdict(artifact)
                with wave.open(str(asset.path)) as wav:
                    entry.update(duration_sec=wav.getnframes()/wav.getframerate(), sample_rate_hz=wav.getframerate())
            elif asset.modality == 'video':
                crop = process_video(asset, settings.output / 'cache/video', settings.video)
                entry.update(path=crop['video'], checksum=digest(crop['video']),
                             duration_sec=crop['frames']/crop['fps'], frame_rate=crop['fps'],
                             camera_id=asset.metadata.get('camera_id'), person_track_id=str(crop['track_id']))
                for alignment in sequence_record['alignments']:
                    if alignment['asset_id'] == asset.asset_id:
                        alignment['time_offset_sec'] -= crop['source_time_origin_sec']
            assets.append(entry)
    accepted = sum(m['state'] == 'accepted' for m in motions)
    state = ('accepted' if accepted == len(motions) else 'partial') if accepted else (
        'error' if any(m['state'] == 'error' for m in motions) else 'rejected')
    return dict(name=sequence.sequence_id, sequence=sequence_record, motions=motions, assets=assets, state=state)


def _worker_process(sequence):
    settings, tracker_version, diagnostic, encoder = _worker_context
    try:
        return process_sequence(sequence, settings, tracker_version, diagnostic, encoder)
    except Exception as error:
        return dict(name=sequence.sequence_id, sequence=_record(sequence), motions=[], assets=[], state='error',
                    error=str(error), traceback=traceback.format_exc())


def publish_work(work, output, config=None):
    from data_pipeline.release.builder import ReleaseBuilder
    work = Path(work)
    if not (work / 'results.sqlite').is_file():
        raise FileNotFoundError(work / 'results.sqlite')
    if config is None:
        config = json.loads((work.parent / 'config.json').read_text())['release']
    return ReleaseBuilder(output, config).publish(Store(work / 'results.sqlite').results())


def run(config, *, limit=None, sequence=None, root=None, output=None, dry_run=False, diagnostic=False, workers=1, finalize=False, resume=False):
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be positive')
    settings = Settings.read(config, output, root=root)
    sequences = discover(settings.dataset, settings.root, audio=settings.audio['enabled'],
                         text=settings.text['enabled'], video=settings.video['enabled'], sequence=sequence)
    if settings.asset_manifest is not None:
        from data_pipeline.sources.assets import attach_assets
        sequences = attach_assets(sequences, settings.asset_manifest)
    selected = iter(islice(sequences, limit))
    try:
        first = next(selected)
    except StopIteration:
        raise ValueError(f'No matching samples found under {settings.root}') from None
    from itertools import chain
    selected = chain([first], selected)
    if dry_run:
        samples = [dict(name=s.sequence_id, motions=[str(m.path) for m in s.motions],
                        assets=[str(a.path) for a in s.assets], texts=len(s.texts)) for s in selected]
        return dict(dataset=settings.dataset, output=str(settings.output), count=len(samples), workers=workers, samples=samples)
    if (settings.output / 'release').exists():
        raise FileExistsError('Release already exists; select a new --output directory')
    from data_pipeline.motion.sonic import version as sonic_version
    tracker_version = sonic_version(settings.tracker) if settings.tracker['enabled'] else 'not_applicable'
    settings.output.mkdir(parents=True, exist_ok=True)
    write_json(settings.output / 'config.json', json.loads(json.dumps(asdict(settings), default=str)))
    ledger = Store(settings.output / 'work/results.sqlite')
    counts = dict(total=0, accepted=0, rejected=0, errors=0, partial=0, motion_errors=0)
    completed_names = set()
    if resume:
        for result in ledger.results():
            if result['state'] != 'error' and not any(m['state'] == 'error' for m in result.get('motions', [])):
                completed_names.add(result['name'])
                counts['total'] += 1
                counts[result['state']] += 1
    else:
        with ledger.connect() as db:
            db.execute('DELETE FROM results')
    seen = set()
    result_buffer = []
    def checked():
        for ordinal, source in enumerate(selected):
            if source.sequence_id in seen:
                raise ValueError(f'Duplicate sample names: {source.sequence_id}')
            if settings.video['enabled'] and not any(a.modality == 'video' for a in source.assets):
                raise ValueError('Video enabled but source has no declared video assets/alignments')
            seen.add(source.sequence_id)
            if source.sequence_id not in completed_names:
                yield ordinal, source
    def record(ordinal, result):
        result_buffer.append((result['name'], ordinal, result))
        counts['total'] += 1
        counts['motion_errors'] += sum(m['state'] == 'error' for m in result.get('motions', []))
        counts[{'error': 'errors'}.get(result['state'], result['state'])] += 1
        if counts['total'] % 100 == 0:
            ledger.record_many(result_buffer)
            result_buffer.clear()
            write_json(settings.output / 'work/progress.json', counts)
    try:
        with ExitStack() as resources:
            inference = None
            if settings.tracker['enabled'] and settings.tracker.get('inference_batch_size', 1) > 1:
                from data_pipeline.motion.inference import SharedPolicy
                inference = resources.enter_context(SharedPolicy(settings.tracker, workers, settings.output / 'cache/policy'))
            state = inference.worker_state if inference is not None else None
            if workers == 1:
                _init_worker(settings, tracker_version, diagnostic, state)
                for ordinal, source in checked():
                    record(ordinal, _worker_process(source))
            else:
                # Bounded submission avoids retaining all episodes/futures in memory.
                with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'), initializer=_init_worker,
                                         initargs=(settings, tracker_version, diagnostic, state)) as pool:
                    iterator = iter(checked()); pending = {}
                    exhausted = False
                    while pending or not exhausted:
                        if inference is not None and inference.failed.is_set():
                            raise RuntimeError('Shared SONIC GPU inference failed')
                        while not exhausted and len(pending) < 2*workers:
                            try:
                                ordinal, source = next(iterator)
                            except StopIteration:
                                exhausted = True; break
                            pending[pool.submit(_worker_process, source)] = ordinal
                        if pending:
                            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                            for future in completed:
                                record(pending.pop(future), future.result())
    finally:
        ledger.record_many(result_buffer)
    write_json(settings.output / 'work/progress.json', counts)
    release = publish_work(settings.output / 'work', settings.output / 'release', settings.release)
    summary = dict(dataset=settings.dataset, **counts, release=release)
    write_json(settings.output / 'summary.json', summary)
    if finalize:
        from data_pipeline.release.package import finalize_output
        summary['release'] = finalize_output(settings.output)
        write_json(settings.output / 'summary.json', summary)
    return summary
