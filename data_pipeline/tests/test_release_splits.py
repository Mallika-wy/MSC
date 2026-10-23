"""Song-preserving repartitioning reuses existing release features."""
from dataclasses import replace
import numpy as np
from data_pipeline.schema import Inventory
from data_pipeline.release.export import export_release, read_motion
from data_pipeline.release.package import package_release
from data_pipeline.release.splits import assign_song_splits, resplit_release, song_disjoint
from data_pipeline.tests.fixtures import fixture_records


def test_exact_song_splits_and_feature_reuse(tmp_path):
    rows=[(str(i),f'group{i}',[f'song{i//2}']) for i in range(20)]
    assignments=assign_song_splits(rows,{'split_seed':4})
    assert {s:list(assignments.values()).count(s) for s in ('train','val','test')}=={'train':16,'val':2,'test':2}
    assert assignments==assign_song_splits(reversed(rows),{'split_seed':4})
    for i in range(0,20,2):
        assert assignments[str(i)]==assignments[str(i+1)]
    inv,tracks=fixture_records(tmp_path/'source')
    inv.provenance['train_default']='source_g1'
    audio = next(link for link in inv.links if link.modality=='audio')
    for motion in inv.motions:
        if not any(link.motion_id==motion.motion_id and link.modality=='audio' for link in inv.links):
            inv.links.append(replace(audio,link_id=motion.motion_id+':song',motion_id=motion.motion_id,
                                     actor_id=motion.actor_id))
    # This fixture is one connected song/video group, so it must stay intact.
    root=tmp_path/'reference'
    export_release(inv,tracks,root)
    bundle=tmp_path/'bundle'
    package_release(root,bundle,include_media=False)
    before=Inventory.read(bundle)
    feature_paths={f.feature_uri for f in before.features}
    inodes={p:(bundle/p).stat().st_ino for p in feature_paths}
    counts=resplit_release(bundle)
    after=Inventory.read(bundle)
    assert len(set(s.split for s in after.sequences))==1
    assert sum(counts.values())==len(before.sequences)
    for path,inode in inodes.items():
        assert (bundle/path).stat().st_ino==inode
    for a,b in zip(before.motions,after.motions):
        np.testing.assert_array_equal(read_motion(bundle,a)[0],read_motion(bundle,b)[0])


def test_aist_music_id_keeps_different_audio_excerpts_together():
    results=[]
    for song in range(10):
        for dancer in range(2):
            sid=f'gBR_d{dancer}_mBR{song}_ch01'
            results.append(dict(name=sid,state='accepted',
                sequence=dict(sequence_id=sid,source_dataset='aistpp',split_group_id=sid,split='train'),
                assets=[dict(modality='audio',checksum=f'distinct-excerpt-{song}-{dancer}')]))
    split=list(song_disjoint(results,{}))
    for song in range(10):
        assert len({r['sequence']['split'] for r in split if f'_mBR{song}_' in r['name']})==1
    assert {s:sum(r['sequence']['split']==s for r in split) for s in ('train','val','test')}=={'train':16,'val':2,'test':2}
