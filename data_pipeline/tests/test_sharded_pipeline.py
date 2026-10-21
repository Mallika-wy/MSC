"""Regression coverage for cleaned OMG, consolidated storage and generic release."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from data_pipeline.config import Settings
from data_pipeline.pipeline import run, publish_work
from data_pipeline.release.export import export_release, read_motion
from data_pipeline.release.validate import validate_release
from data_pipeline.schema import Inventory
from data_pipeline.sources import discover
from data_pipeline.sources.omg import read_motion as read_omg
from data_pipeline.storage.store import Store, load_arrays
from data_pipeline.tests.fixtures import fixture_records


def omg_fixture(root):
    (root / 'meta/episodes/chunk-000').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    info = dict(fps=30, robot_type='unitree_g1', features={'observation.state':{'shape':[36]}},
                splits={'train':'0:2','val':'2:3','test':'3:4'},
                data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet')
    (root / 'meta/info.json').write_text(json.dumps(info))
    # Intentionally not an untouched official snapshot.
    (root / 'meta/omg_manifest.json').write_text('{"cleaned":true}')
    frames, episodes = [], []
    qpos = np.zeros((5,36),dtype=np.float32); qpos[:,2:4] = [.8,1.]
    for episode in range(4):
        split = ['train','train','val','test'][episode]
        for i in range(5):
            frames.append({'observation.state':qpos[i].tolist(), 'action':(qpos[i]+10).tolist(),
                           'index':episode*5+i,'episode_index':episode,'frame_index':i,'timestamp':i/30})
        episodes.append({'episode_index':episode,'length':5,'dataset_from_index':episode*5,'dataset_to_index':episode*5+5,
            'data/chunk_index':0,'data/file_index':0,'tasks':['walk','walk','  '], 'omg/has_text':True,
            'omg/split':split,'omg/source_id':f'source{episode}','omg/dataset':'original',
            'omg/original_episode_index':episode+100})
    pq.write_table(pa.Table.from_pylist(frames),root / 'data/chunk-000/file-000.parquet',row_group_size=3)
    pq.write_table(pa.Table.from_pylist(episodes),root / 'meta/episodes/chunk-000/file-000.parquet')
    return qpos


def omg_config(tmp_path, **release):
    root = tmp_path / 'raw'
    qpos = omg_fixture(root)
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(dict(source={'dataset':'omg','root':'raw'},
        release={'output':'processed',**release})))
    return config,qpos


def test_omg_cleaned_release_all_splits_and_republish_without_models(tmp_path, monkeypatch):
    config,qpos = omg_config(tmp_path)
    import builtins
    original = builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.startswith(('general_motion_retargeting','mujoco','transformers','onnxruntime')):
            raise AssertionError('Disabled heavy dependency imported')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    result = run(config)
    assert result['accepted']==4
    root = tmp_path / 'processed/release'
    inv = Inventory.read(root)
    assert {s.split for s in inv.sequences}=={'train','val','test'}
    assert len(inv.annotations)==4 and not inv.assets
    assert all(m.label_type=='source_g1' and m.retarget_version=='not_applicable' for m in inv.motions)
    for motion in inv.motions:
        np.testing.assert_allclose(read_motion(root,motion)[0],qpos)
    assert len(list((root/'motion').rglob('*.parquet')))==3
    assert len(list((root/'reports').glob('quality*')))==1
    assert not list((tmp_path/'processed').rglob('*.npz'))
    assert not list((tmp_path/'processed/cache').rglob('*.sqlite'))
    assert inv.motions[1].row_start==5
    assert pq.ParquetFile(root / inv.motions[0].storage_uri).num_row_groups==1
    from data_pipeline.motion import processor
    monkeypatch.setattr(processor,'process_motion',lambda *a,**k: pytest.fail('reprocessing'))
    assert publish_work(tmp_path/'processed/work',tmp_path/'republished')['motions']==4
    assert validate_release(tmp_path/'republished',100)['valid']
    with pytest.raises(FileExistsError):
        publish_work(tmp_path/'processed/work',root)
    assert validate_release(root,100)['valid']


def test_omg_discovery_and_identity_validation(tmp_path):
    config,qpos = omg_config(tmp_path)
    settings = Settings.read(config)
    sources = list(discover('omg',settings.root,audio=False))
    assert len(sources)==4 and sources[0].texts[0].text=='walk'
    assert sources[0].provenance['episode']['omg/original_episode_index']==100
    np.testing.assert_array_equal(read_omg(sources[0].motions[0]),qpos)
    broken = replace(sources[0].motions[0],metadata={**sources[0].motions[0].metadata,'episode_index':999})
    with pytest.raises(ValueError,match='identity'):
        read_omg(broken)
    assert run(config,dry_run=True)['count']==4
    assert not (tmp_path/'processed').exists()


def test_shard_boundaries_keep_tracks_whole_and_qc_shared(tmp_path):
    inv,tracks = fixture_records(tmp_path/'raw')
    root = tmp_path/'release'
    export_release(inv,tracks,root,max_shard_frames=180)
    loaded = Inventory.read(root)
    assert len(list((root/'motion/train').glob('*.parquet')))==3
    assert [m.row_start for m in loaded.motions]==[0,90,0,90,0]
    assert len({m.quality_ref for m in loaded.motions})==1
    for original,motion in zip(inv.motions,loaded.motions):
        np.testing.assert_allclose(read_motion(root,motion)[0],tracks[original.motion_id])
    assert validate_release(root,100)['valid']


def test_transactional_store_corruption_and_recovery(tmp_path):
    store = Store(tmp_path/'cache.sqlite')
    with store.lock('one'):
        ref = store.put('one',{'stage':'test'},qpos_36=np.ones((4,36)))
    np.testing.assert_array_equal(load_arrays(ref)['qpos_36'],np.ones((4,36)))
    with store.connect() as db:
        db.execute('UPDATE artifacts SET payload=? WHERE key=?',(b'truncated','one'))
    assert store.get('one') is None
    with pytest.raises(ValueError,match='corrupt'):
        load_arrays(ref)
    store.put('one',{},qpos_36=np.zeros((4,36)))
    assert load_arrays(ref)['qpos_36'].sum()==0
    store.record_many([('later', 2, {'name':'later'}), ('first', 0, {'name':'first'})])
    assert [row['name'] for row in store.results()]==['first','later']


def _put_shared(store_path):
    import time
    store = Store(store_path)
    with store.lock('shared'):
        if store.get('shared') is None:
            time.sleep(.02)
            store.put('shared',{},values=np.arange(5))
    return store.get('shared')[1]['values'].tolist()


def test_spawn_store_deduplicates_shared_key(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    path = tmp_path/'shared.sqlite'
    with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
        assert list(pool.map(_put_shared,[path,path]))==[list(range(5))]*2
    with Store(path).connect() as db:
        assert db.execute('SELECT COUNT(*) FROM artifacts').fetchone()[0]==1


def test_song_disjoint_groups_transitive_shared_assets():
    from data_pipeline.release.splits import song_disjoint
    def result(name,songs,split):
        return dict(name=name,state='accepted',sequence={'source_dataset':'fixture','split_group_id':name,'split':split},
                    assets=[{'modality':'audio','checksum':song} for song in songs])
    rows = [result('a',['a','b'],'train'),result('b',['b','c'],'val'),result('c',['c'],'test')]
    first = list(song_disjoint(iter(rows),{'split_seed':7}))
    assert len({r['sequence']['split'] for r in first})==1
    reverse = list(song_disjoint(reversed(rows),{'split_seed':7}))
    assert {r['name']:r['sequence']['split'] for r in first}=={r['name']:r['sequence']['split'] for r in reverse}


def test_config_rejects_mixed_and_missing_parent(tmp_path):
    config,qpos = omg_config(tmp_path)
    values = yaml.safe_load(config.read_text())
    values['dataset']='omg'; config.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError,match='legacy'):
        Settings.read(config)
    del values['dataset']; values['motion']={'tracking':{'enabled':True,
        'sonic_root':'sonic', 'encoder':'encoder.onnx', 'decoder':'decoder.onnx',
        'observation_config':'observations.yaml'}}
    values['release']['publish_labels']=['tracker_executed']; values['release']['train_default']='tracker_executed'
    config.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError,match='parent'):
        Settings.read(config)


def test_generic_builder_combined_modalities_and_failed_actor(tmp_path, monkeypatch):
    from data_pipeline.sources.base import (SourceSequence, SourceMotion, SourceAsset, SourceText, SourceAlignment)
    from data_pipeline.storage.artifacts import MotionArtifact, FeatureArtifact
    from data_pipeline import pipeline
    from data_pipeline.motion import processor as motion
    from data_pipeline.audio import processor as audio
    from data_pipeline.video import processor as video
    from data_pipeline.io import sha256_file
    from data_pipeline.tests.test_pipeline import make_wav
    from data_pipeline.release.features import read_feature
    from data_pipeline.release.package import package_release
    raw = tmp_path/'raw'; raw.mkdir()
    source_motion = raw/'motion.bin'; source_motion.write_bytes(b'motion')
    wav = raw/'song.wav'; make_wav(wav)
    film = raw/'source.mp4'; film.write_bytes(b'synthetic video asset')
    cropped = tmp_path/'cropped.mp4'; cropped.write_bytes(b'cropped synthetic asset')
    sequence = SourceSequence('group','omg','group',[
        SourceMotion('good',source_motion,representation='g1_qpos36'),
        SourceMotion('bad',source_motion,'actor1',representation='g1_qpos36')],
        [SourceAsset('song',wav,'audio'),SourceAsset('front',film,'video')],
        [SourceText('walk',0.,.2)], [SourceAlignment('good','song'),
        SourceAlignment('good','front',time_offset_sec=1.02,status='verified',source='fixture')])
    qpos = np.zeros((6,36),dtype=np.float32); qpos[:,2:4]=[.8,1.]
    store = Store(tmp_path/'cache.sqlite')
    ref = store.put('motion',{},qpos_36=qpos)
    feature_ref = store.put('feature',{},values=np.ones((75,4),dtype=np.float16),timestamps=np.arange(75,dtype=np.float64)/75)
    def process(source,*args,**kwargs):
        if source.motion_id=='bad': raise ValueError('bad actor')
        return [MotionArtifact('source_g1',ref,6,{'accepted':True})]
    def sound(source,*args):
        return FeatureArtifact(source.asset_id,feature_ref,dict(encoder='synthetic',checkpoint_hash='mock',
            preprocess_hash='mock',input_checksum=sha256_file(wav),shape=[75,4]))
    monkeypatch.setattr(pipeline,'discover',lambda *a,**k:iter([sequence]))
    monkeypatch.setattr(motion,'process_motion',process)
    monkeypatch.setattr(audio,'process_audio',sound)
    monkeypatch.setattr(video,'process_video',lambda *a,**k:dict(video=str(cropped),frames=30,fps=30.,track_id=7,source_time_origin_sec=1.))
    monkeypatch.setattr(audio,'AudioCache',lambda *a,**k:None)
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(dict(source={'dataset':'omg','root':'raw'},
        audio={'enabled':True,'model':'missing'},video={'enabled':True,'weights':'missing'},
        release={'output':'processed'})))
    summary = run(config)
    assert summary['partial']==1 and summary['motion_errors']==1
    root = tmp_path/'processed/release'
    inv = Inventory.read(root)
    assert len(inv.motions)==1 and len(inv.annotations)==1 and len(inv.links)==2
    assert {link.modality for link in inv.links}=={'audio','video'}
    assert next(l for l in inv.links if l.modality=='video').time_offset_sec==pytest.approx(.02)
    assert read_feature(root,inv.features[0],inv.external_roots)[0].shape==(75,4)
    bundle = tmp_path/'bundle'
    package_release(root,bundle)
    bundle.rename(tmp_path/'moved')
    assert validate_release(tmp_path/'moved',100)['valid']
    assert len(list((tmp_path/'moved/features/audio').glob('*.parquet')))==1
    assert inv.provenance['processing_summary']['motion_error']==1


def test_supplemental_video_manifest_keeps_identity_and_alignment(tmp_path):
    from data_pipeline.sources.assets import attach_assets
    config,qpos = omg_config(tmp_path)
    video = tmp_path/'view.mp4'; video.touch()
    manifest = tmp_path/'assets.parquet'
    pq.write_table(pa.Table.from_pylist([dict(sequence_id='episode-00000000',motion_id='episode-00000000',
        asset_id='view',path='view.mp4',modality='video',time_scale=1.,time_offset_sec=.2,
        alignment_status='estimated',alignment_source='manual fixture',camera_id='front',track_id=3)]),manifest)
    sources = list(attach_assets(discover('omg',tmp_path/'raw',audio=False),manifest))
    assert sources[0].assets[0].metadata=={'camera_id':'front','track_id':3}
    assert sources[0].alignments[0].time_offset_sec==.2
    assert not sources[1].assets


def test_republish_refuses_changed_native_source(tmp_path):
    config,qpos = omg_config(tmp_path)
    run(config,limit=1)
    path = tmp_path/'raw/data/chunk-000/file-000.parquet'
    table = pq.read_table(path)
    rows = table.to_pylist(); rows[0]['observation.state'][0] = 9.
    pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),path)
    with pytest.raises(ValueError,match='changed since processing'):
        publish_work(tmp_path/'processed/work',tmp_path/'changed-release')
    assert not (tmp_path/'changed-release').exists()
    assert not list(tmp_path.glob('.changed-release-*'))
    assert validate_release(tmp_path/'processed/release',100)['valid']
