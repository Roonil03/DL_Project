import hashlib
import json
from pathlib import Path
import zipfile

import pytest
from scripts.runner_support import bundle, check_identity, snapshot


def test_identity_prevents_reusing_markers_after_inputs_change(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    check_identity(run, {'options': {'epochs': 1}})
    (run / 'marker.done').write_text('complete')
    check_identity(run, {'options': {'epochs': 1}})
    with pytest.raises(ValueError, match='identity changed'):
        check_identity(run, {'options': {'epochs': 2}})


def test_legacy_run_cannot_be_trusted(tmp_path):
    (tmp_path / 'old.done').write_text('complete')
    with pytest.raises(ValueError, match='no identity'):
        check_identity(tmp_path, {})


def test_archive_manifest_matches_members_and_excludes_checkpoints(tmp_path):
    run = tmp_path / 'run'
    for name in ('logs/run.log', 'figures/loss.png', 'notebooks/run.ipynb', 'telemetry/state.json', 'checkpoints/latest.pt', 'code/src/model.py', 'code/notebooks/template.ipynb', 'code/configs/training.yaml', 'code/Dockerfile', 'code/docker-compose.yml', 'code/scripts/runner_support.py', 'code/requirements.docker.txt', 'code/run_all_models.sh'):
        p = run / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(name.encode())
    archive = tmp_path / 'bundle.zip'
    bundle(run, archive)
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('run/bundle_manifest.json'))
        assert set(z.namelist()) == {x['path'] for x in manifest['files']} | {'run/bundle_manifest.json'}
        assert not any('/checkpoints/' in name for name in z.namelist())
        assert 'run/code/src/model.py' in z.namelist()
        assert 'run/code/notebooks/template.ipynb' in z.namelist()
        assert 'run/code/configs/training.yaml' in z.namelist()
        assert 'run/code/Dockerfile' in z.namelist()
        assert 'run/code/scripts/runner_support.py' in z.namelist()
        for row in manifest['files']:
            contents = z.read(row['path'])
            assert row['bytes'] == len(contents)
            assert row['sha256'] == hashlib.sha256(contents).hexdigest()
    bundle(run, archive, include_checkpoints=True)
    with zipfile.ZipFile(archive) as z:
        assert 'run/checkpoints/latest.pt' in z.namelist()


def test_dataset_archive_uses_allowlist_and_omits_secrets(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    data = tmp_path / 'data'
    data.mkdir()
    for name in ('daily_market_data.csv', 'monthly_macro_data.csv', 'codes_and_sources.md',
                 '10_Industry_Portfolios_daily_CSV.zip', '.env', '.env.local', 'credentials.json'):
        (data / name).write_text(name)
    (data / 'nested').mkdir()
    (data / 'nested' / '.env').write_text('secret')
    archive = tmp_path / 'bundle.zip'
    bundle(run, archive, data=data)
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('run/bundle_manifest.json'))
        assert {name for name in z.namelist() if name.startswith('dataset/')} == {
            'dataset/daily_market_data.csv', 'dataset/monthly_macro_data.csv',
            'dataset/codes_and_sources.md', 'dataset/10_Industry_Portfolios_daily_CSV.zip'}
        assert {x['path'] for x in manifest['files']} == set(z.namelist()) - {'run/bundle_manifest.json'}


def test_snapshot_uses_only_explicit_files_and_removes_stale_files(tmp_path):
    root = tmp_path / 'project'
    (root / 'scripts').mkdir(parents=True)
    (root / 'src').mkdir()
    (root / 'scripts' / 'source_snapshot.txt').write_text('src/model.py\n')
    (root / 'src' / 'model.py').write_text('model')
    (root / 'src' / 'secret.py').write_text('secret')
    dest = tmp_path / 'run' / 'code'
    dest.mkdir(parents=True)
    (dest / 'old-secret').write_text('secret')
    snapshot(root, dest)
    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()) == ['src/model.py']


def test_dataset_archive_skips_symlinked_input(tmp_path):
    run = tmp_path / 'run'
    data = tmp_path / 'data'
    run.mkdir(); data.mkdir()
    secret = tmp_path / 'secret'
    secret.write_text('secret')
    try:
        (data / 'daily_market_data.csv').symlink_to(secret)
    except OSError:
        pytest.skip('Creating symlinks requires host privileges')
    archive = tmp_path / 'bundle.zip'
    bundle(run, archive, data=data)
    with zipfile.ZipFile(archive) as z:
        assert 'dataset/daily_market_data.csv' not in z.namelist()
