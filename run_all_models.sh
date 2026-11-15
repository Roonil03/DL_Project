#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Host entrypoint: build/start the CUDA service, then execute this same script
# inside the image.  Compose sets IN_CONTAINER=1 to prevent recursion.
if [[ "${IN_CONTAINER:-0}" != "1" ]]; then
  command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 2; }
  docker image inspect fin-dl-benchmark:cuda12.4 >/dev/null 2>&1 || docker compose build --pull jupyter
  docker compose up -d jupyter
  exec docker compose exec -T \
    -e IN_CONTAINER=1 \
    -e GIT_REVISION="${GIT_REVISION:-}" \
    -e RUN_ID="${RUN_ID:-}" -e MODELS="${MODELS:-}" -e SEEDS="${SEEDS:-}" \
    -e DEVICE="${DEVICE:-}" -e RESUME_MODE="${RESUME_MODE:-}" \
    -e EPOCHS="${EPOCHS:-}" -e FOLDS="${FOLDS:-}" \
    -e OUTPUT_ROOT="${OUTPUT_ROOT:-}" -e MAX_RETRIES="${MAX_RETRIES:-}" \
    -e INCLUDE_DATA="${INCLUDE_DATA:-}" -e PACKAGE_CHECKPOINTS="${PACKAGE_CHECKPOINTS:-}" -e ESTIMATE_ONLY="${ESTIMATE_ONLY:-}" \
    -e DATASET_MODE="${DATASET_MODE:-}" -e DATA_DIR="${DATA_DIR:-}" -e FIN_ASSETS="${FIN_ASSETS:-}" -e FIN_DATA_RETRIEVED_AT="${FIN_DATA_RETRIEVED_AT:-}" \
    jupyter ./run_all_models.sh
fi

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"

RUN_ID="${RUN_ID:-run-$(date -u +%Y%m%dT%H%M%SZ)}"
MODELS="${MODELS:-dlinear,lstm,mamba,patchtst}"
SEEDS="${SEEDS:-42,52,62,72,82}"
DEVICE="${DEVICE:-gpu}"
RESUME_MODE="${RESUME_MODE:-auto}"
EPOCHS="${EPOCHS:-100}"
FOLDS="${FOLDS:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
MAX_RETRIES="${MAX_RETRIES:-2}"
INCLUDE_DATA="${INCLUDE_DATA:-0}"
ESTIMATE_ONLY="${ESTIMATE_ONLY:-0}"
DATASET_MODE="${DATASET_MODE:-global_indian}"
case "$DATASET_MODE" in
  global_indian) DATA_DIR="${DATA_DIR:-data/kaggle_global_indian_markets/global-india-markets-macro}"; FIN_ASSETS="${FIN_ASSETS:-NIFTY50,SP500,USDINR,GOLD,BRENT}" ;;
  french_industry) DATA_DIR="${DATA_DIR:-data/french_industry}" ;;
  *) echo "Unsupported DATASET_MODE: $DATASET_MODE" >&2; exit 2 ;;
esac
PACKAGE_CHECKPOINTS="${PACKAGE_CHECKPOINTS:-0}"

# Notebook execution changes its working directory to `notebooks/`; resolve
# all host-provided paths before launching nbconvert so every notebook writes
# into the same run tree.
OUTPUT_ROOT="$(realpath -m -- "$OUTPUT_ROOT")"
DATA_DIR="$(realpath -m -- "$DATA_DIR")"
export PROJECT_ROOT="$ROOT" OUTPUT_ROOT DATA_DIR
GIT_REVISION="${GIT_REVISION:-$(git rev-parse HEAD 2>/dev/null || true)}"
export GIT_REVISION DATASET_MODE

[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ && "$RUN_ID" != *..* ]] || { echo "Invalid RUN_ID" >&2; exit 2; }
[[ "$DEVICE" == "gpu" || "$DEVICE" == "cpu" ]] || { echo "DEVICE must be gpu or cpu" >&2; exit 2; }
[[ "$RESUME_MODE" =~ ^(auto|restart|fail)$ ]] || { echo "RESUME_MODE must be auto, restart, or fail" >&2; exit 2; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "EPOCHS must be positive" >&2; exit 2; }
[[ "$MAX_RETRIES" =~ ^[0-9]+$ ]] || { echo "MAX_RETRIES must be non-negative" >&2; exit 2; }

RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
ARTIFACT="artifacts/${RUN_ID}.zip"
mkdir -p "$OUTPUT_ROOT" artifacts

if [[ -d "$RUN_DIR" && "$RESUME_MODE" == "restart" ]]; then
  archived="${RUN_DIR}.restart-$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$RUN_DIR" "$archived"
elif [[ -d "$RUN_DIR" && "$RESUME_MODE" == "fail" ]]; then
  echo "Run state already exists for $RUN_ID and RESUME_MODE=fail" >&2
  exit 3
fi

# Bind this run to all inputs before reading or writing completion markers.
mkdir -p "$RUN_DIR"
options=$(python - "$DEVICE" "$EPOCHS" "$MODELS" "$SEEDS" "$FOLDS" "$DATASET_MODE" "${FIN_ASSETS:-}" "$ESTIMATE_ONLY" "$GIT_REVISION" "${FIN_DATA_RETRIEVED_AT:-}" <<'PYIDENT'
import hashlib,json,os,subprocess,sys
options=dict(zip(('device','epochs','models','seeds','folds','dataset_mode','assets','estimate_only','git_revision','data_retrieved_at'),sys.argv[1:]))
options["dependency_lock_sha256"]=hashlib.sha256(subprocess.check_output([sys.executable,"-m","pip","freeze"])).hexdigest()
options["image_identifier"]=os.getenv("BENCHMARK_IMAGE")
print(json.dumps(options))
PYIDENT
)
python scripts/runner_support.py identity "$ROOT" "$DATA_DIR" "$RUN_DIR" "$options"
mkdir -p "$RUN_DIR"/{metadata,logs,telemetry,checkpoints,figures,tables,predictions,notebooks}
RUN_LOG="$RUN_DIR/logs/run.log"
EVENTS="$RUN_DIR/logs/events.jsonl"
FAILURES="$RUN_DIR/telemetry/failures.json"
STATE="$RUN_DIR/telemetry/run_state.json"
TIMING="$RUN_DIR/telemetry/timing.json"
METRICS="$RUN_DIR/telemetry/metrics.json"
rm -f "$ARTIFACT"
touch "$RUN_LOG" "$EVENTS"
exec 3>&1 4>&2
exec > >(tee -a "$RUN_LOG") 2>&1

event() {
  local kind="$1"; shift
  python - "$EVENTS" "$kind" "$RUN_ID" "$DEVICE" "$@" <<'PY'
import json, sys
from datetime import datetime, timezone
path, kind, run_id, device, *fields = sys.argv[1:]
record={"schema_version":"1.0","timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "event":kind,"run_id":run_id,"device":device}
for field in fields:
    key, _, value=field.partition("="); record[key]=value
with open(path,"a",encoding="utf-8") as f: f.write(json.dumps(record,sort_keys=True)+"\n")
PY
}

fail() { event failure "message=$*"; echo "ERROR: $*" >&2; exit 1; }
MONITOR_PID=""
trap 'status=$?; if [[ -n "$MONITOR_PID" ]]; then kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; fi; if (( status != 0 )); then printf "{\"schema_version\":\"1.0\",\"status\":\"failed\"}\n" > "$STATE"; fi; event runner_exit "status=$status"; exit $status' EXIT
printf '{"schema_version":"1.0","status":"running"}\n' > "$STATE"

csv_words() { printf '%s' "$1" | tr ',' ' '; }
IFS=' ' read -r -a MODEL_LIST <<< "$(csv_words "$MODELS")"
IFS=' ' read -r -a SEED_LIST <<< "$(csv_words "$SEEDS")"
IFS=' ' read -r -a FOLD_LIST <<< "$(csv_words "$FOLDS")"

if [[ "$ESTIMATE_ONLY" == "1" ]]; then
  SEED_LIST=("${SEED_LIST[0]}")
  FOLD_LIST=("${FOLD_LIST[0]}")
  EPOCHS=1
fi

for required in notebooks/{00_core,10_data_audit,20_benchmark,30_validation,40_report}.ipynb; do
  [[ -f "$required" ]] || fail "missing notebook $required"
done
if [[ "$DATASET_MODE" == "global_indian" ]]; then
  for file in daily_market_data.csv monthly_macro_data.csv codes_and_sources.md; do
    [[ -s "$DATA_DIR/$file" ]] || fail "missing or empty dataset file $DATA_DIR/$file"
  done
else
  [[ -s "$DATA_DIR/10_Industry_Portfolios_daily_CSV.zip" ]] || fail "missing or empty French industry ZIP"
fi

event preflight_started
nvidia-smi || { [[ "$DEVICE" == "cpu" ]] || fail "nvidia-smi failed"; }
python - "$DEVICE" <<'PY'
import sys, torch
device_name=sys.argv[1]
if device_name == "gpu":
    if not torch.cuda.is_available(): raise SystemExit("CUDA requested but unavailable")
    device=torch.device("cuda")
else: device=torch.device("cpu")
x=torch.randn(16,8,device=device,requires_grad=True)
model=torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.GELU(),torch.nn.Linear(16,1)).to(device)
model(x).square().mean().backward()
print({"torch":torch.__version__,"cuda":torch.version.cuda,
       "device":torch.cuda.get_device_name(0) if device.type=="cuda" else "cpu"})
PY
if [[ ",$MODELS," == *,mamba,* ]]; then
python - <<'PY'
import torch
from src.models.mamba_model import MambaSignalModel
m=MambaSignalModel(20,15,5,d_model=16,n_layers=1)
assert getattr(m,"implementation_name",None)=="Mamba", "real Mamba implementation not active"
if torch.cuda.is_available():
    m=m.cuda(); y=m(torch.randn(2,20,15,device="cuda")); y.sum().backward()
    assert y.shape==(2,5) and torch.isfinite(y).all()
print("Mamba identity and GPU forward/backward verified")
PY
fi
event preflight_complete
touch "$RUN_DIR/telemetry/system_metrics.jsonl"
python scripts/system_monitor.py sample "$RUN_DIR/telemetry/system_metrics.jsonl" &
MONITOR_PID=$!

python - "$RUN_DIR" "$RUN_ID" "$DEVICE" "$RESUME_MODE" "$EPOCHS" "$MODELS" "$SEEDS" "$FOLDS" "$DATA_DIR" <<'PY'
import hashlib, json, os, platform, subprocess, sys, torch
from datetime import datetime, timezone
from pathlib import Path
run_dir=Path(sys.argv[1]); run_id,device,resume,epochs,models,seeds,folds,data_dir=sys.argv[2:]
def digest(path):
    h=hashlib.sha256();
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
files=[]
for p in sorted(Path(data_dir).iterdir()):
    if p.is_file(): files.append({"name":p.name,"bytes":p.stat().st_size,"sha256":digest(p)})
git=os.getenv("GIT_REVISION") or None
if git is None:
    try: git=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: git=None
manifest={"schema_version":"1.0","timestamp_utc":datetime.now(timezone.utc).isoformat(),
 "run_id":run_id,"git_revision":git,"device":device,"resume_mode":resume,"epochs":int(epochs),
 "models":models.split(','),"seeds":[int(x) for x in seeds.split(',')],"folds":folds.split(','),
 "data_files":files,"dataset_mode":os.getenv("DATASET_MODE", "global_indian")}
(run_dir/"metadata"/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
system={"platform":platform.platform(),"python":platform.python_version(),"torch":torch.__version__,
 "cuda_available":torch.cuda.is_available(),"cuda_runtime":torch.version.cuda,
 "cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
 "image_identifier":os.getenv("BENCHMARK_IMAGE")}
(run_dir/"metadata"/"system.json").write_text(json.dumps(system,indent=2)+"\n")
PY
python - "$RUN_DIR/metadata/config_snapshot.json" configs/*.yaml <<'PY'
import json,sys,yaml
payload={}
for name in sys.argv[2:]:
    with open(name,encoding="utf-8") as f: payload[name]=yaml.safe_load(f)
open(sys.argv[1],"w",encoding="utf-8").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
python -m pip freeze > "$RUN_DIR/metadata/dependency_lock.txt"

execute_notebook() {
  local source="$1" output="$2" timeout="${3:--1}"
  jupyter nbconvert --to notebook --execute "$source" \
    --output "$output" --output-dir "$RUN_DIR/notebooks" \
    --ExecutePreprocessor.timeout="$timeout" \
    --ExecutePreprocessor.kernel_name=fin-dl-benchmark
}

export RUN_ID OUTPUT_ROOT DEVICE RESUME_MODE EPOCHS DATASET_MODE
export FIN_ASSETS="${FIN_ASSETS:-}" FIN_DATA_RETRIEVED_AT="${FIN_DATA_RETRIEVED_AT:-}"
if [[ "$DATASET_MODE" == "french_industry" ]]; then
  export FIN_DATA_PATH="$DATA_DIR/10_Industry_Portfolios_daily_CSV.zip"
  unset FIN_MACRO_PATH
else
  export FIN_DATA_PATH="$DATA_DIR/daily_market_data.csv"
  export FIN_MACRO_PATH="$DATA_DIR/monthly_macro_data.csv"
fi

execute_notebook notebooks/00_core.ipynb 00_core.executed.ipynb 300
execute_notebook notebooks/10_data_audit.ipynb 10_data_audit.executed.ipynb 600

declare -a FAIL_LIST=()
declare -a DURATIONS=()
for model in "${MODEL_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    for fold in "${FOLD_LIST[@]}"; do
      export MODEL="$model" SEED="$seed" FOLD="$fold" RUN_TRAINING=1
      unit="${model}__seed_${seed}__fold_${fold}"
      done_marker="$RUN_DIR/telemetry/${unit}.done"
      if [[ "$RESUME_MODE" == "auto" && -s "$done_marker" && -s "$RUN_DIR/tables/$model/seed_$seed/fold_$fold/metrics.csv" && -s "$RUN_DIR/tables/$model/seed_$seed/fold_$fold/returns.csv" && -s "$RUN_DIR/tables/$model/seed_$seed/fold_$fold/predictions.csv" && -s "$RUN_DIR/notebooks/${unit}.ipynb" ]]; then
        event unit_skipped "model=$model" "seed=$seed" "fold=$fold"
        continue
      fi
      start=$(date +%s)
      attempt=0
      success=0
      while (( attempt <= MAX_RETRIES )); do
        event unit_started "model=$model" "seed=$seed" "fold=$fold" "attempt=$attempt"
        if execute_notebook notebooks/20_benchmark.ipynb "${unit}.ipynb" -1; then
          success=1; break
        fi
        event unit_failed "model=$model" "seed=$seed" "fold=$fold" "attempt=$attempt"
        (( attempt += 1 ))
        (( attempt <= MAX_RETRIES )) || break
        checkpoint_dir="$RUN_DIR/checkpoints/$model/seed_${seed}/fold_${fold}"
        [[ -s "$checkpoint_dir/latest.pt" || -s "$checkpoint_dir/previous_latest.pt" ]] || break
        python - "$checkpoint_dir" "$model" "$seed" "$fold" <<'PY' || break
import sys, torch
from pathlib import Path
valid=False
for path in (Path(sys.argv[1])/"latest.pt",Path(sys.argv[1])/"previous_latest.pt"):
    if not path.is_file(): continue
    try: p=torch.load(path,map_location="cpu",weights_only=False)
    except Exception: continue
    meta=p.get("unit_identity",{})
    valid = (str(meta.get("model"))==sys.argv[2] and str(meta.get("seed"))==sys.argv[3]
             and str(meta.get("fold"))==sys.argv[4])
    if valid: break
assert valid, "no compatible valid checkpoint"
PY
      done
      elapsed=$(( $(date +%s) - start )); DURATIONS+=("$model:$elapsed")
      if (( success )); then
        date -u +%Y-%m-%dT%H:%M:%SZ > "$done_marker"
        event unit_complete "model=$model" "seed=$seed" "fold=$fold" "seconds=$elapsed"
      else
        FAIL_LIST+=("$unit")
      fi
    done
  done
done

python - "$TIMING" "$EVENTS" "$ESTIMATE_ONLY" "${DURATIONS[@]}" <<'PY'
import json, statistics, sys
from datetime import datetime, timezone
from pathlib import Path
path,events,estimate,*items=sys.argv[1:]
if not items and Path(events).is_file():
    for line in Path(events).read_text(encoding="utf-8").splitlines():
        try:
            row=json.loads(line)
            if row.get("event")=="unit_complete":
                items.append(f"{row.get('model','unknown')}:{row.get('seconds',0)}")
        except json.JSONDecodeError:
            continue
rows=[{"model":x.rsplit(':',1)[0],"seconds":float(x.rsplit(':',1)[1])} for x in items]
vals=[r["seconds"] for r in rows]
payload={"schema_version":"1.0","timestamp_utc":datetime.now(timezone.utc).isoformat(),"units":rows,
 "median_seconds":statistics.median(vals) if vals else None,
 "p95_seconds":sorted(vals)[min(len(vals)-1,int(.95*len(vals)))] if vals else None,
 "measured_pilot":estimate=="1"}
open(path,"w",encoding="utf-8").write(json.dumps(payload,indent=2)+"\n")
PY

if (( ${#FAIL_LIST[@]} )); then
  python - "$FAILURES" "${FAIL_LIST[@]}" <<'PY'
import json,sys
open(sys.argv[1],"w",encoding="utf-8").write(json.dumps({"schema_version":"1.0","failed_units":sys.argv[2:]},indent=2)+"\n")
PY
  python - "$STATE" <<'PY'
import json,sys
open(sys.argv[1],"w").write(json.dumps({"schema_version":"1.0","status":"failed"},indent=2)+"\n")
PY
  fail "required units failed: ${FAIL_LIST[*]}"
fi

export RUN_TRAINING=0 MODEL=all
execute_notebook notebooks/30_validation.ipynb 30_validation.executed.ipynb 900
execute_notebook notebooks/40_report.ipynb 40_report.executed.ipynb 900

python - "$RUN_DIR" "$METRICS" "$STATE" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
root,metrics,state=map(Path,sys.argv[1:])
tables=sorted(str(x.relative_to(root)) for x in (root/"tables").rglob("*.csv"))
payload={"schema_version":"1.0","timestamp_utc":datetime.now(timezone.utc).isoformat(),"status":"complete","tables":tables}
metrics.write_text(json.dumps(payload,indent=2)+"\n")
state.write_text(json.dumps({"schema_version":"1.0","status":"complete"},indent=2)+"\n")
PY

cp "$RUN_DIR/Final_Metrics.md" Final_Metrics.md
mkdir -p "$RUN_DIR/code"
cp -R src configs notebooks tests scripts "$RUN_DIR/code/"
cp Dockerfile docker-compose.yml requirements.docker.txt requirements.mamba-compat.txt run_all_models.sh README.md RunningNotebooks.md DOCKER.md "$RUN_DIR/code/"
# Stop the sampler and render its history before taking the archive snapshot.
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
MONITOR_PID=""
python scripts/system_monitor.py plot "$RUN_DIR/telemetry/system_metrics.jsonl" "$RUN_DIR/figures/system_utilization.png"
# End logging before taking the archive snapshot, so checksums cover final bytes.
event packaging_started "archive=$ARTIFACT"
trap - EXIT
exec 1>&3 2>&4
# Wait for the tee process to flush before reading run.log.
wait || true
bundle_args=()
[[ "$PACKAGE_CHECKPOINTS" == "1" ]] && bundle_args+=(--checkpoints)
[[ "$INCLUDE_DATA" == "1" ]] && bundle_args+=(--data "$DATA_DIR")
if ! python scripts/runner_support.py bundle "$RUN_DIR" "$ARTIFACT" "${bundle_args[@]}"; then
  python - "$STATE" <<'PY'
import json,sys
open(sys.argv[1],"w").write(json.dumps({"schema_version":"1.0","status":"failed","reason":"archive creation failed"})+"\n")
PY
  exit 1
fi
echo "Completed successfully: $ARTIFACT"
