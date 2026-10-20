from pathlib import Path

import pytest

from data_pipeline.core.io import read_json, write_json
from data_pipeline.exporters.omg import export_release, validate_release
from data_pipeline.exporters.package import package_release
from data_pipeline.tests.test_pipeline import make_entry


def test_package_copies_shared_assets_and_survives_relocation(tmp_path, monkeypatch):
    entry = make_entry(tmp_path)
    source, output = tmp_path / 'release', tmp_path / 'packaged'
    export_release([entry], {'export': {'episode_frames': 4, 'stride_frames': 3}}, source)
    execution = tmp_path / 'execution.mp4'
    execution.write_bytes(b'test replay')
    records = read_json(source / 'meta/video_index.json')
    for record in records:
        record['execution_video_uri'] = str(execution)
        record['views'][0]['source_uri'] = '/deleted/raw/input.mp4'
    write_json(source / 'meta/video_index.json', records)
    result = package_release(source, output)
    assert result['episodes'] == 2 and result['source_sequences'] == 1
    assert result['execution_videos'] == 1
    assert len(list((output / 'videos').rglob('*.mp4'))) == 1
    records = read_json(output / 'meta/video_index.json')
    assert records[0]['views'][0]['original_video_name'] == 'input.mp4'
    assert 'source_uri' not in records[0]['views'][0]
    assert not Path(records[0]['views'][0]['uri']).is_absolute()
    # 删除原资产，再移动包并从其他cwd验证，确保不是软链接或依赖旧绝对路径。
    (tmp_path / 'source.mp4').unlink()
    execution.unlink()
    moved = tmp_path / 'moved'
    output.rename(moved)
    monkeypatch.chdir(tmp_path.parent)
    assert validate_release(moved)['valid']
    assert not (moved / records[0]['views'][0]['uri']).is_symlink()
    assert (moved / records[0]['execution_video_uri']).read_bytes() == b'test replay'


@pytest.mark.parametrize('uri', ['../outside.mp4', '/outside.mp4'])
def test_packaged_assets_cannot_escape_dataset(tmp_path, uri):
    entry = make_entry(tmp_path)
    source, output = tmp_path / 'release', tmp_path / 'packaged'
    export_release([entry], {}, source)
    package_release(source, output)
    records = read_json(output / 'meta/video_index.json')
    records[0]['views'][0]['uri'] = uri
    write_json(output / 'meta/video_index.json', records)
    with pytest.raises(ValueError, match='inside the dataset'):
        validate_release(output)


def test_package_never_overwrites_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        package_release(tmp_path / 'not-needed', tmp_path)
