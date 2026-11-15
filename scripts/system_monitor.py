"""Sample host and GPU utilization while the benchmark is running."""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def sample():
    row = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
           'cpu_percent': psutil.cpu_percent(), 'memory_percent': psutil.virtual_memory().percent,
           'gpu_utilization_percent': None, 'gpu_memory_used_mib': None}
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                          '--format=csv,noheader,nounits'], text=True, timeout=5)
        gpu, memory = result.splitlines()[0].split(',')
        row.update(gpu_utilization_percent=float(gpu), gpu_memory_used_mib=float(memory))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return row


def plot(path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    for key, label in [('cpu_percent', 'CPU'), ('memory_percent', 'RAM'), ('gpu_utilization_percent', 'GPU')]:
        values = [row.get(key) for row in rows]
        if any(value is not None for value in values):
            ax.plot(range(len(rows)), values, label=label)
    ax.set(xlabel='Sample', ylabel='Utilization (%)', ylim=(0, 100))
    ax.legend(); fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


if __name__ == '__main__':
    path = Path(sys.argv[2])
    if sys.argv[1] == 'plot':
        plot(path, Path(sys.argv[3]))
    else:
        while True:
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(sample()) + '\n')
            time.sleep(10)
