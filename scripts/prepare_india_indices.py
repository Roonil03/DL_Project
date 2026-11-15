"""Download and hash-check the pinned research snapshot, without replacing files."""
from pathlib import Path
import sys
from urllib.request import urlopen
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.india_indices import BASE_URL, DEFAULT_PATH, HASHES, IndiaIndicesAdapter, digest

def main():
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_PATH)
    dest.mkdir(parents=True, exist_ok=True)
    for name, checksum in HASHES.items():
        path = dest / name
        if path.exists():
            if digest(path) != checksum: raise SystemExit(f"Existing snapshot hash mismatch: {path}")
            continue
        temporary = path.with_suffix(path.suffix + '.download')
        try:
            with urlopen(BASE_URL + name, timeout=60) as response: temporary.write_bytes(response.read())
            if digest(temporary) != checksum: raise ValueError(f"Downloaded snapshot hash mismatch: {name}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    print(IndiaIndicesAdapter(dest).manifest())

if __name__ == '__main__': main()
