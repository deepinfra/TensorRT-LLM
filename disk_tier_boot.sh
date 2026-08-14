#!/bin/bash
# Disk-tier boot wrapper: per-pod KV disk-cache dir with a flock+signature
# lifecycle. BASE is derived from the config's disk_cache_path, so it works with
# a dedicated mount (/kvdisk) OR a subdir of an existing mount (/data/kvdisk).
# On boot: reap dead pods' dirs (free lock); create <base>/<sig>_<id> and hold
# its lock for the container's life; on graceful exit remove it. Crash/OOM skips
# cleanup -> next boot reaps it. No-ops to a plain exec when the disk tier is off.
set -u
SIG=dikv

# Per-pod id: hash of the GPU UUIDs (unique per pod, visible without downward API;
# NVIDIA_VISIBLE_DEVICES can be "all", nvidia-smi is ground truth). POD_NAME overrides.
ID="${POD_NAME:-}"
GPUS=""
if [ -z "$ID" ]; then
  GPUS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | sort)
  [ -n "$GPUS" ] && ID=$(printf '%s' "$GPUS" | tr -d '[:space:]' | md5sum | cut -c1-16)
fi
[ -z "$ID" ] && ID="${HOSTNAME:-}"

DIR=""
ARGS=()
for a in "$@"; do
  case "$a" in
    --extra_llm_api_options=*)
      SRC="${a#*=}"; OUT=/tmp/di_llm_opts.json
      DIR=$(SRC="$SRC" OUT="$OUT" SIG="$SIG" ID="$ID" python3 - <<'PY'
import json, os
try:
    src = os.environ["SRC"]
    d = json.load(open(src)) if os.path.exists(src) else json.loads(src)
    kv = d.get("kv_cache_config") or {}
    base = (kv.get("disk_cache_path") or "").rstrip("/")   # config path is the BASE
    if base and (kv.get("disk_cache_size") or 0) > 0 and os.environ["ID"]:
        p = "{}/{}_{}".format(base, os.environ["SIG"], os.environ["ID"])
        kv["disk_cache_path"] = p
        d["kv_cache_config"] = kv
        json.dump(d, open(os.environ["OUT"], "w"))
        print(p, end="")                 # non-empty -> disk tier on; this is the per-pod DIR
except Exception:
    pass
PY
)
      if [ -n "$DIR" ]; then ARGS+=("--extra_llm_api_options=$OUT"); else ARGS+=("$a"); fi
      ;;
    *) ARGS+=("$a") ;;
  esac
done

# disk tier off / no options / unparseable / no id -> original behavior, untouched
[ -z "$DIR" ] && exec /usr/local/bin/trtllm-serve "${ARGS[@]}"

BASE="$(dirname "$DIR")"                  # e.g. /data/kvdisk or /kvdisk
echo "[disk-tier] per-pod dir: $DIR (base=$BASE id=$ID gpus:$(printf ' %s' $GPUS))"
mkdir -p "$BASE"
{
  flock -x 8
  mkdir -p "$DIR"
  exec 9>"$DIR/.lock"; flock -x 9        # held for the container's whole life
  shopt -s nullglob
  for d in "$BASE/${SIG}_"*/; do
    [ "$d" = "$DIR/" ] && continue
    flock -xn "${d}.lock" -c "rm -rf '$d'" 2>/dev/null && echo "[disk-tier] reaped orphan $d"
  done
} 8>"$BASE/.sweep.lock"

trap '[ -n "${DIR:-}" ] && rm -rf "$DIR"' EXIT          # graceful self-cleanup
trap 'kill -TERM "$engine" 2>/dev/null' TERM INT        # forward drain signal

/usr/local/bin/trtllm-serve "${ARGS[@]}" &
engine=$!
while kill -0 "$engine" 2>/dev/null; do wait "$engine"; done
