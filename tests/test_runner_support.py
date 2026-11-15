import hashlib
import json
from pathlib import Path
import zipfile

import pytest
from scripts.runner_support import bundle, check_identity


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
