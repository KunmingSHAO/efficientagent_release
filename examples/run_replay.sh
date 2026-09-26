#!/usr/bin/env bash
# Replay one trace under the three write-admission options with the EfficientAgent connector.
#
#   MODEL=/models/Qwen3-Coder-30B-A3B-Instruct TP=8 HOST_GIB=5 TRACE=traces/my_trace bash examples/run_replay.sh
#
# Without TRACE, a small synthetic trace is generated first. Each run starts a fresh server with a cold cache.
set -euo pipefail

MODEL=${MODEL:?set MODEL to a model directory or Hugging Face ID}
TP=${TP:-1}
HOST_GIB=${HOST_GIB:-5}                     # host (CPU) KV capacity per TP rank
WORKERS=${WORKERS:-16}                      # active task slots
PYTHON=${PYTHON:-python}                    # interpreter of the vLLM 0.13.0 + LMCache 0.3.12 environment
OUT=${OUT:-runs}
EXTRA=${EXTRA:-}                            # e.g. "--gpu-memory-utilization 0.9 --max-model-len 65536 --max-num-seqs 16"

if [[ -z "${TRACE:-}" ]]; then
  TRACE=traces/synthetic
  "$PYTHON" -m efficientagent.replay.synthetic_trace --out "$TRACE" --tasks 8 --steps 6 12
fi

for ADMISSION in none fixed conditioned; do
  "$PYTHON" -m efficientagent.replay.launch \
    --admission "$ADMISSION" --host-gib "$HOST_GIB" \
    --model "$MODEL" --tp "$TP" --python "$PYTHON" $EXTRA \
    --trace "$TRACE" --workers "$WORKERS" \
    --out "$OUT/${ADMISSION}_${HOST_GIB}gib" --execute
done

"$PYTHON" -m efficientagent.analysis.run_summary "$OUT" --out "$OUT/summary.json" --markdown "$OUT/summary.md"
