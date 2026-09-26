# EfficientAgent

Code for **EfficientAgent: What Makes Host KV Offloading Work for Concurrent Agents?**
(ICLR 2027 submission).

EfficientAgent sizes and admits host-memory KV state for serving many concurrent language-model agents with
vLLM and LMCache. This repository contains the write-admission connector, the dependency-preserving replay
framework used to measure it, and the CPU-only analysis tools (stack-distance capacity model, reuse-distance and
admission analyses, run summaries).

## Overview

Agents resubmit a growing context at every turn, so almost every prompt starts with tokens the server has already
processed. Keeping that KV state and reusing it (prefix caching, and KV tiering to host memory when GPU memory is
short) is the main lever for serving agents. For many concurrent agents, however, whether a stored prefix is still
there at the agent's next turn is not a per-request property: between two turns of one agent, the server processes
the contexts of all other agents in the active pool. An agent's prefix survives only if the tier holds this
**reuse working set**, which grows with the pool size and the context length. A host tier smaller than the working
set thrashes: it keeps writing refills of evicted context that are evicted again before they are reused.

EfficientAgent sizes and admits KV state by this working set:

* **Sizing.** A stack-distance model over the recorded agent histories counts the distinct KV referenced between two
  uses of every chunk, predicts the host prefix coverage, restored tokens and computed prefill as a function of host
  capacity, and locates the capacity at which offloading starts to pay. The working-set scale of a backlogged pool
  is `C_reuse ≈ (A − 1) · N̄ · β_rank` (A active agents, mean prompt length N̄, KV bytes per token per rank β_rank).
* **Admission.** At run time, a connector inside LMCache's vLLM integration schedules writes to the host tier by
  working-set load control. It decides once per request, at the scheduler's first prefix lookup:

  ```
  p_t     = [C_reuse > C_H]  ∧  [tier full and evicting]        (estimate alone when no fresh telemetry)
  save(r) = ¬(p_t ∧ u > κ),    u = ⌊n/b⌋ − ⌊h/b⌋ new full chunks of a request with n prompt tokens, h held by the tier
  ```

  Under pressure the runtime keeps saving small extensions of host-resident prefixes and declines large refills;
  when pressure subsides, refills are admitted again. Copy-time deduplication drops chunks the tier already holds.
  Offload without write admission (`p_t ≡ 0`) and fixed write admission (`p_t ≡ 1`) are the endpoints of this rule.

## Repository layout

```
efficientagent/
  kvtier/
    connector.py         EfficientAgentConnector: write admission + deduplication for vLLM's LMCacheConnectorV1
    policies.py          admission options (none / fixed / conditioned) -> vLLM arguments and environment
  replay/
    build_trace.py       replay trace from agent logs with exact token IDs
    synthetic_trace.py   small synthetic trace for trying the pipeline
    client.py            dependency-preserving closed-loop replay client
    forced_output.py     vLLM logits processor that forces recorded outputs
    launch.py            one replay run: fresh server, replay, counters, per-run metrics
    lmcache_observer/    read-only sampler of the LMCache CPU tier (loaded via PYTHONPATH)
  analysis/
    common.py            trace/run readers, LMCache log parsing, stack distances
    runsets.py           run arguments (RUN_DIR or RUN_DIR@HOST_GIB) resolved against run.json
    run_metrics.py       per-run metrics (makespan, computed prefill, host traffic, counters)
    run_summary.py       summary table over runs
    compare_runs.py      paired comparison of two runs of the same trace
    trace_profile.py     workload profile and reuse structure of a trace
    capacity_model.py    stack-distance capacity model and active-pool predictions
    pressure_estimate.py offline evaluation of the pressure signal on a run
    admission_sensitivity.py  write-threshold and occupancy-threshold sensitivity from decision streams
    declined_reuse.py    reuse distance of the writes an admission run declined
examples/                run_replay.sh (launcher), serve_with_connector.sh (manual server + client)
tests/                   CPU tests (pytest); GPU end-to-end test behind --run-gpu
```

## Requirements

* Serving: Linux with NVIDIA GPUs supported by vLLM, **vLLM 0.13.0** and **LMCache 0.3.12**, Python 3.10 or newer
  (developed with Python 3.12). The paper serves Qwen3-Coder-30B-A3B-Instruct (BF16) with tensor parallelism 8.
  The host tier needs `host capacity per rank × TP size` of CPU memory in addition to the server's own memory.
* Analyses and tests: CPU only; Python 3.10+, NumPy, pytest.

## Install

```bash
git clone <this repository> efficientagent && cd efficientagent
python -m venv .venv && . .venv/bin/activate
pip install -e ".[serve,test]"      # serving stack (vllm==0.13.0, lmcache==0.3.12) + tests
# or, for the analyses and CPU tests only:
pip install -e ".[test]"
```

## Quick CPU tests

```bash
pytest
```

The tests use small synthetic inputs generated in the test code and need no GPU. With vLLM and LMCache installed,
the connector and logits-processor tests run against the real base classes; without them, the connector tests use a
minimal stand-in for `LMCacheConnectorV1`. The end-to-end GPU test starts real servers on a small local model:

```bash
EA_TEST_MODEL=/path/to/small-model pytest --run-gpu tests/test_gpu_replay.py
```

## Write admission

| `--admission` | behavior |
|---|---|
| `none` | offload without write admission (unmodified LMCache; `--use-connector` serves it through the EfficientAgent connector with admission disabled); `--host-gib 0` runs without a host tier |
| `fixed` | fixed write admission: the request filter `u > κ` on every request, plus copy-time deduplication |
| `conditioned` | capacity-conditioned admission: the request filter under pressure `p_t`, plus copy-time deduplication |

The connector decides at a request's first resolved prefix lookup and keeps the store/skip choice for later lookups
of the same request, including lookups after preemption. A skip sets LMCache's request-level `lmcache.skip_save`
switch; lookup, restore, eviction and serialization are LMCache's own. For the pressure signal, the LMCache worker of
TP rank 0 publishes the tier's telemetry (registered chunks, chunk capacity `K_H`, chunks evicted in the last window
to make room for new ones); the tier is *full and evicting* when occupancy `o_t ≥ θ` and evictions `e_t > 0`.
`A_t` counts the tasks with a first prefix lookup in the last window (tasks are identified by
`efficientagent.task_id` in the request's `kv_transfer_params`), and `N̄_t` is the mean prompt length of the last
256 first lookups.

Parameters (environment variables of the serving process; the launcher sets them from its options):

| Parameter | Variable | Default |
|---|---|---|
| admission option | `EA_KVTIER_ADMISSION` | `none` |
| write threshold κ (full chunks) | `EA_KVTIER_WRITE_THRESHOLD` | 8 |
| occupancy threshold θ | `EA_KVTIER_OCCUPANCY_THRESHOLD` | 0.95 |
| task-activity and eviction window (s) | `EA_KVTIER_WINDOW_S` | 60 |
| prompt-length window (first lookups) | `EA_KVTIER_PROMPT_WINDOW` | 256 |
| telemetry report interval (s) | `EA_KVTIER_TELEMETRY_INTERVAL_S` | 1 |
| scheduler report-read interval (s) | `EA_KVTIER_TELEMETRY_READ_S` | 0.5 |
| maximum age of a fresh report (s) | `EA_KVTIER_TELEMETRY_MAX_AGE_S` | 5 |
| KV bytes per token per rank β_rank | `EA_KVTIER_BYTES_PER_TOKEN` | 24576 (Qwen3-Coder-30B-A3B-Instruct, BF16, TP8) |
| host chunk size b (tokens) | `LMCACHE_CHUNK_SIZE` | 1024 |
| telemetry file | `EA_KVTIER_TELEMETRY_FILE` | unset (estimate alone) |
| counter directory | `EA_KVTIER_STATS_DIR` | unset |

`β_rank = 2 · layers · head_dim · bytes · max(1, ⌈KV heads / TP⌉)`;
`efficientagent.kvtier.policies.kv_bytes_per_token_from_config(model_dir, tp)` computes it from a model's
`config.json`, and the launcher does so automatically for a local model directory.

To use the connector with your own vLLM server, put this repository on `PYTHONPATH` (or `pip install -e .`), set the
LMCache and `EA_KVTIER_*` variables, and select the connector (see `examples/serve_with_connector.sh`):

```bash
--kv-transfer-config '{"kv_connector": "EfficientAgentConnector",
                       "kv_connector_module_path": "efficientagent.kvtier.connector",
                       "kv_role": "kv_both",
                       "kv_connector_extra_config": {"lmcache.local_cpu": true, "lmcache.max_local_cpu_size": 5.0}}'
```

Counters are written per process to `EA_KVTIER_STATS_DIR/<role>-<pid>.json`: decisions, skipped and saved requests
and new chunks; for `conditioned` also requests under pressure, estimate above capacity, telemetry full and
evicting, and stale telemetry; on the workers, deduplicated chunks, capacity evictions and store calls. The first
occurrence of each event is logged as `EA_KVTIER first-hit <EVENT>`.

## Replay

Dependency-preserving replay fixes each call's tokens while serving decisions shift the timing of later calls:
task slots take tasks in trace order, and a task's next call is sent after its previous response has finished and the
recorded inter-call interval (tool execution, agent processing) has elapsed. Prompts are the exact recorded token
IDs, and a logits processor forces the recorded output tokens, so every run submits the same token work while the
model executes normally.

### 1. Build a trace from agent logs

The trace builder reads an agent run directory with one record per task and one log line per LLM call, including
the exact prompt and completion token IDs and wall-clock timestamps:

```
RUN/tasks/<task_id>.json
    {"attempt_index": 0, "trial_id": "...", "task_timing": {"assigned_ns": ..., "finished_ns": ...}}
RUN/attempts/<task_id>/attempt_00/telemetry/llm_requests.jsonl
    {"trial_id": "...", "request_sequence": 1, "error": null, "token_identity_verified": true,
     "prompt_token_ids": [...], "generated_token_ids": [[...]],
     "start_wall_ns": ..., "end_wall_ns": ..., "client_elapsed_ns": ...}
```

Any agent harness that records token IDs and call timestamps can be exported to this layout. Then:

```bash
python -m efficientagent.replay.build_trace --run-dir RUN --out traces/my_trace [--tasks task_order.json] [--limit N]
```

The trace is a directory of `NNN_<task_id>.jsonl.gz` files (a header line with the task ID, order and trailing
gap, then one line per call with the delta-encoded prompt, the forced output and the gap before the call) plus
`trace_manifest.json`. Failed calls are not replayed; their time stays inside the recorded gaps. To try the pipeline
without agent logs, generate a synthetic trace:

```bash
python -m efficientagent.replay.synthetic_trace --out traces/synthetic --tasks 8 --steps 6 12
```

### 2. Run a replay

```bash
python -m efficientagent.replay.launch \
    --admission conditioned --host-gib 5 \
    --model /models/Qwen3-Coder-30B-A3B-Instruct --tp 8 \
    --max-model-len 262144 --max-num-seqs 16 \
    --trace traces/my_trace --workers 16 \
    --out runs/conditioned_5gib --execute
```

* `--admission {none,fixed,conditioned}` and `--host-gib` (host tier per TP rank; `0` = no host tier) select the
  policy; `--use-connector` routes `none` through the connector with admission disabled.
* `--workers` is the active pool (task slots); `--max-num-seqs` is the engine's running-request cap;
  `--gpu-memory-utilization` sets vLLM's GPU memory fraction and thereby the GPU KV capacity.
* `--write-threshold`, `--occupancy-threshold`, `--window-s`, `--prompt-window`, `--telemetry-max-age-s`,
  `--chunk-tokens` and `--kv-bytes-per-token` set the admission parameters (defaults as in the table above).
* Without `--execute` the launcher prints the server command and environment. `--python` selects the interpreter
  of the serving environment; `--server-arg=...` passes extra vLLM arguments; `--greedy-check N` records unforced
  greedy outputs for N tasks after the replay (compare them across runs to check restored KV state);
  `--wait-idle-gpus S` waits for GPUs without compute processes before starting.

Each run starts a fresh server with a cold cache and writes to `--out`:

```
run.json                  spec, status, client summary, connector counters
manifest.json             server command, environment, source checksums
server.log                vLLM + LMCache log
metrics_start.prom        Prometheus snapshot at replay start
metrics_end.prom          Prometheus snapshot at replay end
metrics.jsonl.gz, load_samples.jsonl   raw metrics and scheduler gauges every 2 s
replay/requests.jsonl, replay/tasks.jsonl, replay/replay_summary.json
kvtier_stats/             connector counters and the rank-0 telemetry file
telemetry_history.jsonl   5-s copies of the telemetry (conditioned admission)
lmcache_observer/         10-s samples of the CPU tier per rank (resident keys, evictions)
run_metrics.json          per-run metrics
```

`examples/run_replay.sh` runs the three options on one trace and writes a summary table.

### 3. Summaries and comparisons

```bash
python -m efficientagent.analysis.run_metrics runs/conditioned_5gib            # recompute run_metrics.json
python -m efficientagent.analysis.run_summary runs/ --out runs/summary.json --markdown runs/summary.md
python -m efficientagent.analysis.compare_runs runs/none_5gib runs/conditioned_5gib --out runs/compare.json
```

Makespan is the replay makespan with all recorded intervals included. Computed prefill is the interval difference of
vLLM's `request_prefill_kv_computed_tokens_sum`, preemptions use `num_preemptions_total`, and host read/write
volumes sum LMCache's per-worker transfer log lines and divide by the TP size.

## Analyses

All analyses run on CPU from a trace and run directories and write JSON (and a short Markdown summary where noted).

A run argument is a run directory; policy, host capacity, active pool, TP size and admission parameters are read
from its `run.json`. `RUN_DIR@GIB` overrides the host capacity. `--bytes-per-token` (β_rank, default 24576) and
`--chunk` (default 1024) set the KV footprint and chunk size where a command needs them.

**Trace profile** — calls per task, prompt and output tokens, and the shared-prefix sums on token IDs: the longest
common prefix of each prompt with the task's previous prompt, and with the previous prompt followed by its recorded
output (the cache-stable prompt length). Also reports how much newly written KV the next call reuses.

```bash
python -m efficientagent.analysis.trace_profile --trace traces/my_trace --out profile.json
```

**Capacity model** — single-pass LRU stack distances over the reference stream in which every request references
all full chunks of its prompt at its scheduling lookup (prefix-dependent keys). A chunk survives in a tier of
`K_H = ⌊C_H / (b·β_rank)⌋` chunks iff its stack distance is below `K_H`, and a request's host coverage is the run of
surviving chunks from the first chunk; useful restoration is the coverage above the GPU-resident prefix. `curve`
predicts host coverage, restored tokens and computed prefill over a grid of capacities on each run's recorded order,
compares the prediction with the run's own measurements, and reports reuse-distance distributions and the
working-set scale `(A − 1)·N̄·β_rank`. `pool` predicts the curve for a different active pool from a closed-loop queue
model and the GPU prefix capacities implied by reference runs; service rates are parameters.

```bash
python -m efficientagent.analysis.capacity_model curve --trace traces/my_trace \
    --run runs/none_5gib --run runs/none_40gib --out capacity_curve.json --md capacity_curve.md
python -m efficientagent.analysis.capacity_model pool --trace traces/my_trace \
    --run runs/none_5gib --run runs/none_40gib --active-pool 8 --engine-slots 3 5 \
    --prefill-tok-s PREFILL_TOKENS_PER_S --decode-s-per-token DECODE_S_PER_TOKEN --host-gib 2 3 4 5 10 --out pool8.json
```

**Pressure estimate** — evaluates the pressure signal offline on any run: the working-set estimate at each request's
dispatch for given capacities and, from the host-tier observer samples, the full-and-evicting condition and its
conjunction with the estimate.

```bash
python -m efficientagent.analysis.pressure_estimate --run runs/none_5gib --run runs/none_40gib --host-gib 5 10 40 --out pressure.json
```

**Admission sensitivity** — rebuilds each request's decision inputs `n` and `h` from LMCache's first-lookup log line
and reports, for every write threshold κ, the selected requests and the share of new-chunk writes they carry (for
fixed admission the stream must reproduce the connector counters); for the occupancy threshold θ it lists the
occupancy of every telemetry report with eviction activity.

```bash
python -m efficientagent.analysis.admission_sensitivity --run runs/fixed_5gib --run runs/conditioned_5gib --out sensitivity.json
```

**Declined reuse distance** — for the chunks an admission run declined, the next reuse distance `D` relative to the
tier's `K_H` chunks: in an LRU tier, declining insertions only for chunks with `D ≥ K_H` (or no later reference) keeps
every hit of full admission (Proposition 1 of the paper). Fixed-admission decisions follow exactly from the logged
lookups; conditioned decisions are rebuilt from the estimate and the telemetry history. An LRU replay of each run's
stream with its declines is compared with the logged host hits.

```bash
python -m efficientagent.analysis.declined_reuse --trace traces/my_trace \
    --run runs/fixed_5gib --run runs/conditioned_5gib --run runs/none_5gib --out declined.json --md declined.md
```

## License

Apache License 2.0; see [LICENSE](LICENSE).
