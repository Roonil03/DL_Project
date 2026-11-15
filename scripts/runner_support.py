"""Integrity helpers for the benchmark runner (standard library only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(root, data, options):
    files = {}
    for base in ('src', 'configs', 'notebooks', 'scripts'):
        for path in sorted((root / base).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                files[path.relative_to(root).as_posix()] = digest(path)
    for name in ('run_all_models.sh', 'Dockerfile', 'docker-compose.yml', 'requirements.docker.txt', 'requirements.mamba-compat.txt'):
        files[name] = digest(root / name)
    for path in sorted(data.rglob('*')):
        if path.is_file():
            files['data/' + path.relative_to(data).as_posix()] = digest(path)
    return {'schema_version': '1.0', 'options': options, 'files': files}


def check_identity(run, value):
    path = run / 'metadata' / 'identity.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != value:
            raise ValueError('Run identity changed; choose a new RUN_ID or RESUME_MODE=restart')
    else:
        # Legacy runs cannot prove that completion markers belong to these inputs.
        if any(run.iterdir()):
            raise ValueError('Existing run has no identity; choose a new RUN_ID or RESUME_MODE=restart')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def bundle(run, destination, include_checkpoints=False, data=None):
    files = []
    for path in sorted(run.rglob('*')):
        rel = path.relative_to(run)
        if not path.is_file() or rel.as_posix() == 'bundle_manifest.json':
            continue
        if not include_checkpoints and rel.parts[0] == 'checkpoints':
            continue
        if '.env' in rel.parts or path.name.startswith('.env.'):
            continue
        files.append((path, (Path(run.name) / rel).as_posix()))
    if data:
        for path in sorted(data.rglob('*')):
            if path.is_file():
                files.append((path, (Path('dataset') / path.relative_to(data)).as_posix()))
    entries = [{'path': name, 'bytes': path.stat().st_size, 'sha256': digest(path)} for path, name in files]
    manifest = json.dumps({'schema_version': '1.0', 'files': entries}, indent=2, sort_keys=True) + '\n'
    (run / 'bundle_manifest.json').write_text(manifest, encoding='utf-8')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(name, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path, member in files:
                archive.write(path, member)
            archive.writestr(run.name + '/bundle_manifest.json', manifest)
        with zipfile.ZipFile(name) as archive:
            if archive.testzip() is not None:
                raise ValueError('archive integrity check failed')
            for entry in entries:
                checksum = hashlib.sha256()
                with archive.open(entry['path']) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        checksum.update(chunk)
                if checksum.hexdigest() != entry['sha256']:
                    raise ValueError('file changed during archive creation: ' + entry['path'])
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    ident = sub.add_parser('identity')
    for arg in ('root', 'data', 'run', 'options'):
        ident.add_argument(arg)
    pack = sub.add_parser('bundle')
    pack.add_argument('run'); pack.add_argument('destination')
    pack.add_argument('--checkpoints', action='store_true'); pack.add_argument('--data')
    args = parser.parse_args()
    if args.command == 'identity':
        check_identity(Path(args.run), identity(Path(args.root), Path(args.data), json.loads(args.options)))
    else:
        bundle(Path(args.run), Path(args.destination), args.checkpoints, Path(args.data) if args.data else None)


if __name__ == '__main__':
    main()
