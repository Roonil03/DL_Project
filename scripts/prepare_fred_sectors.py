"""Download and validate one immutable local snapshot of the official CSV."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.fred_sectors import DEFAULT_PATH, SOURCE_URL, FredSectorsAdapter


def main():
    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_PATH)
    if destination.exists():
        raise SystemExit(f"Refusing to replace existing snapshot: {destination}")
    try:
        with urlopen(SOURCE_URL, timeout=30) as response:
            data = response.read()
    except (URLError, TimeoutError, OSError) as exc:
        raise SystemExit(f"Download failed: {exc}. Download {SOURCE_URL} manually to {destination} and retry.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.download')
    try:
        temporary.write_bytes(data)
        retrieved = datetime.now(timezone.utc).isoformat()
        manifest = FredSectorsAdapter(temporary).manifest(retrieved_at_utc=retrieved)
        temporary.replace(destination)
        manifest['path'] = str(destination)
        destination.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Validated snapshot: {destination} (SHA-256 {manifest['sha256']})")


if __name__ == '__main__':
    main()
