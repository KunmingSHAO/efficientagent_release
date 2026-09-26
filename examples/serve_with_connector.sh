#!/usr/bin/env bash
# Serve a model with LMCache and the EfficientAgent connector (capacity-conditioned admission), then replay a trace
# against it with the standalone client. The launcher (examples/run_replay.sh) does the same with bookkeeping.
#
#   MODEL=/models/Qwen3-Coder-30B-A3B-Instruct TP=8 HOST_GIB=5 TRACE=traces/my_trace bash examples/serve_with_connector.sh
set -euo pipefail

MODEL=${MODEL:?set MODEL}
TP=${TP:-1}
HOST_GIB=${HOST_GIB:-5}
PORT=${PORT:-8000}
TRACE=${TRACE:?set TRACE to a trace directory}
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN=${RUN:-runs/manual_conditioned}
mkdir -p "$RUN/kvtier_stats"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONHASHSEED=0

# KV bytes per token per TP rank (beta) of the served model; computed from MODEL/config.json unless given.
BYTES_PER_TOKEN=${BYTES_PER_TOKEN:-$(python -c "from efficientagent.kvtier.policies import kv_bytes_per_token_from_config as f; print(f('$MODEL', $TP))")}
export VLLM_PLUGINS=lmcache.vllm_plugin LMCACHE_LOCAL_CPU=True LMCACHE_CHUNK_SIZE=1024
export LMCACHE_MAX_LOCAL_CPU_SIZE="$HOST_GIB" LMCACHE_PRE_CACHING_HASH_ALGORITHM=sha256_cbor
export EA_KVTIER_ADMISSION=conditioned EA_KVTIER_BYTES_PER_TOKEN="$BYTES_PER_TOKEN"
export EA_KVTIER_STATS_DIR="$RUN/kvtier_stats" EA_KVTIER_TELEMETRY_FILE="$RUN/kvtier_stats/telemetry_rank0.json"

python -m vllm.entrypoints.openai.api_server --model "$MODEL" --port "$PORT" --tensor-parallel-size "$TP" \
  --enable-prefix-caching --disable-hybrid-kv-cache-manager \
  --logits-processors efficientagent.replay.forced_output:ForcedOutputLogitsProcessor \
  --kv-transfer-config "{\"kv_connector\": \"EfficientAgentConnector\", \"kv_connector_module_path\": \"efficientagent.kvtier.connector\", \"kv_role\": \"kv_both\", \"kv_connector_extra_config\": {\"lmcache.local_cpu\": true, \"lmcache.max_local_cpu_size\": $HOST_GIB}}" \
  > "$RUN/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do sleep 5; done

python -m efficientagent.replay.client --trace "$TRACE" --out "$RUN/replay" --base "http://127.0.0.1:$PORT"
