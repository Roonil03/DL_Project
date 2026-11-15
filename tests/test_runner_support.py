import hashlib
import json
from pathlib import Path
import zipfile

import pytest
from scripts.runner_support import bundle, check_identity, snapshot, identity, resolve_fred_folds


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
    bundle(run, archive, include_checkpoints=False)
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
    bundle(run, archive)
    with zipfile.ZipFile(archive) as z:
        assert 'run/checkpoints/latest.pt' in z.namelist()


def test_dataset_archive_uses_allowlist_and_omits_secrets(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    data = tmp_path / 'data'
    data.mkdir()
    for name in ('daily_market_data.csv', 'monthly_macro_data.csv', 'codes_and_sources.md',
                 'sector_indices.csv', '.env', '.env.local', 'credentials.json'):
        (data / name).write_text(name)
    (data / 'nested').mkdir()
    (data / 'nested' / '.env').write_text('secret')
    archive = tmp_path / 'bundle.zip'
    with pytest.raises(ValueError, match='packaging is disabled'):
        bundle(run, archive, data=data)


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
    with pytest.raises(ValueError, match='packaging is disabled'):
        bundle(run, archive, data=data)


def test_identity_hashes_only_selected_dataset_and_sidecar(tmp_path):
    root = tmp_path / 'project'
    for folder in ('src', 'configs', 'notebooks', 'scripts'):
        (root / folder).mkdir(parents=True)
    for name in ('run_all_models.sh', 'Dockerfile', 'docker-compose.yml', 'requirements.docker.txt', 'requirements.mamba-compat.txt'):
        (root / name).write_text(name)
    data = tmp_path / 'data'; data.mkdir()
    for name in ('sector_indices.csv', 'sector_indices.manifest.json', 'daily_market_data.csv', '.env'):
        (data / name).write_text(name)
    options = {'dataset_mode': 'fred_sectors'}
    original = identity(root, data, options)
    assert set(k for k in original['files'] if k.startswith('data/')) == {
        'data/sector_indices.csv', 'data/sector_indices.manifest.json'}
    (data / '.env').write_text('changed secret')
    (data / 'daily_market_data.csv').write_text('unrelated')
    assert identity(root, data, options) == original
    (data / 'sector_indices.manifest.json').write_text('changed')
    assert identity(root, data, options) != original


def test_snapshot_covers_tracked_source_files():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['git', 'ls-files', 'src', 'configs', 'notebooks', 'scripts', 'tests',
                             'Dockerfile', 'docker-compose.yml', 'requirements.docker.txt',
                             'requirements.mamba-compat.txt', 'run_all_models.sh', 'README.md',
                             'RunningNotebooks.md', 'DOCKER.md', 'LICENSE', 'requirements.cpu.txt'], cwd=root, capture_output=True, text=True, check=True)
    listed = set((root / 'scripts/source_snapshot.txt').read_text(encoding='utf-8-sig').splitlines())
    assert listed == set(result.stdout.splitlines())


def test_fred_notebook_defaults_match_config(monkeypatch):
    import ast
    import os
    import yaml
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / 'configs/fred_sectors.yaml').read_text())
    monkeypatch.delenv('FIN_ASSETS', raising=False)
    for notebook, target in [('10_data_audit.ipynb', 'selected'), ('20_benchmark.ipynb', 'assets')]:
        content = json.loads((root / 'notebooks' / notebook).read_text(encoding='utf-8'))
        text = '\n'.join(''.join(cell.get('source', [])) for cell in content['cells'])
        candidates = [line.strip() for line in text.splitlines() if line.strip().startswith(target + ' = ') and "os.getenv('FIN_ASSETS'" in line and " or list(" in line and ("fred_config" in line or "configs['fred_sectors']" in line)]
        assert len(candidates) == 1
        expression = ast.parse(candidates[0]).body[0].value
        actual = eval(compile(ast.Expression(expression), '<notebook default>', 'eval'),
                      {'os': os, 'fred_config': config, 'configs': {'fred_sectors': config}})
        assert actual == config['assets'] and len(actual) == 5
    runner = (root / 'run_all_models.sh').read_text()
    fred_case = next(line for line in runner.splitlines() if line.strip().startswith('fred_sectors)'))
    assert 'FIN_ASSETS' not in fred_case
    assert 'fred-folds configs/fred_sectors.yaml' in fred_case
    assert config['folds']['test_years'] == [2008, 2020]


@pytest.mark.parametrize('mode,expected', [
    ('fred_sectors', {'sector_indices.csv', 'sector_indices.manifest.json'}),
    ('global_indian', {'daily_market_data.csv', 'monthly_macro_data.csv', 'codes_and_sources.md'}),
])
def test_audit_inventory_ignores_unrelated_files(tmp_path, mode, expected):
    from types import SimpleNamespace
    root = Path(__file__).resolve().parents[1]
    content = json.loads((root / 'notebooks/10_data_audit.ipynb').read_text(encoding='utf-8'))
    cell = next(''.join(c.get('source', [])) for c in content['cells'] if 'inventory = pd.DataFrame' in ''.join(c.get('source', [])))
    for name in expected | {'.env', 'credentials.json'}:
        (tmp_path / name).write_text(name)
    rows = []
    pd = SimpleNamespace(DataFrame=lambda values: rows.extend(values) or values)
    env = {'DATASET_MODE': mode, 'DATA_PATH': tmp_path / ('sector_indices.csv' if mode == 'fred_sectors' else 'daily_market_data.csv'),
           'MACRO_PATH': tmp_path / 'monthly_macro_data.csv', 'pd': pd, 'hashlib': hashlib,
           'display': lambda value: None, 'adapter': SimpleNamespace(manifest=lambda: {'source_page': 'test'})}
    exec(cell, env)
    assert {row['file'] for row in rows} == expected


def test_audit_manifest_keeps_identical_bytes_for_resume(tmp_path):
    import textwrap
    root = Path(__file__).resolve().parents[1]
    content = json.loads((root / 'notebooks/10_data_audit.ipynb').read_text(encoding='utf-8'))
    cell = next(''.join(c.get('source', [])) for c in content['cells'] if 'serialized = json.dumps' in ''.join(c.get('source', [])))
    fragments = []
    lines = cell.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith('serialized = json.dumps'):
            end = next(j for j in range(i, len(lines)) if lines[j].strip().startswith('temporary.write_text'))
            fragments.append(textwrap.dedent('\n'.join(lines[i:end + 1])))
    assert len(fragments) == 2
    for fragment in fragments:
        destination = tmp_path / 'data_manifest.json'
        previous = '{"created_utc": "old", "selected_assets": ["A"]}\n'
        destination.write_text(previous)
        env = {'json': json, 'destination': destination, 'manifest': {'created_utc': 'new', 'selected_assets': ['A']}}
        exec(fragment, env)
        assert destination.with_suffix('.json.tmp').read_text() == previous
        env['manifest']['selected_assets'] = ['B']
        exec(fragment, env)
        assert json.loads(destination.with_suffix('.json.tmp').read_text())['selected_assets'] == ['B']


def test_powershell_pilot_result_needs_timing_not_archive(tmp_path):
    import os
    import shutil
    import subprocess
    pwsh = shutil.which('powershell') or shutil.which('pwsh')
    if not pwsh:
        pytest.skip('PowerShell is not installed')
    root = Path(__file__).resolve().parents[1]
    run = tmp_path / 'run'
    (run / 'telemetry').mkdir(parents=True)
    (run / 'telemetry' / 'timing.json').write_text('{}')
    env = dict(os.environ, RUNNER_SCRIPT=str(root / 'scripts/run_dl_benchmark.ps1'), TEST_RUN=str(run), TEST_ARCHIVE=str(tmp_path / 'missing.zip'))
    command = """
$tokens = $null; $errors = $null
$tree = [System.Management.Automation.Language.Parser]::ParseFile($env:RUNNER_SCRIPT, [ref]$tokens, [ref]$errors)
if ($errors) { exit 10 }
$func = $tree.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-RunnerResult'}, $true)
Invoke-Expression $func.Extent.Text
$result = Get-RunnerResult -Run $env:TEST_RUN -Archive $env:TEST_ARCHIVE -Pilot $true
if (-not $result.EndsWith('timing.json')) { exit 11 }
try { Get-RunnerResult -Run $env:TEST_RUN -Archive $env:TEST_ARCHIVE -Pilot $false; exit 12 } catch { }
Remove-Item -LiteralPath (Join-Path $env:TEST_RUN 'telemetry/timing.json')
try { Get-RunnerResult -Run $env:TEST_RUN -Archive $env:TEST_ARCHIVE -Pilot $true; exit 13 } catch { }
exit 0
"""
    subprocess.run([pwsh, '-NoProfile', '-Command', command], env=env, check=True, capture_output=True, text=True)


@pytest.mark.parametrize('years,expected', [([2008], [0]), ([2008, 2020, 2022], [0, 1, 2])])
def test_french_fold_defaults_follow_validated_config(tmp_path, monkeypatch, years, expected):
    import ast
    import sys
    import types
    import yaml
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / 'configs/fred_sectors.yaml').read_text())
    config['folds']['test_years'] = years
    config_file = tmp_path / 'french.yaml'
    config_file.write_text(yaml.safe_dump(config))
    # Execute the project's actual validation function without importing its
    # unrelated training dependencies into this lightweight runner test.
    source = ast.parse((root / 'src/data/fred_benchmark.py').read_text())
    validator = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'validate_fred_config')
    namespace = {'ASSETS': config['assets']}
    exec(compile(ast.Module(body=[validator], type_ignores=[]), '<validator>', 'exec'), namespace)
    module = types.ModuleType('src.data.fred_benchmark')
    module.validate_fred_config = namespace['validate_fred_config']
    monkeypatch.setitem(sys.modules, 'src.data.fred_benchmark', module)
    assert resolve_fred_folds(config_file) == expected
    assert resolve_fred_folds(config_file, str(expected[-1])) == [expected[-1]]
    with pytest.raises(ValueError, match='range'):
        resolve_fred_folds(config_file, str(len(years)))
    with pytest.raises(ValueError, match='comma-separated'):
        resolve_fred_folds(config_file, '-1')
    config['folds']['test_years'] = []
    config_file.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='test_years'):
        resolve_fred_folds(config_file)
