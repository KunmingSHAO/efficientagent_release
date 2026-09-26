"""EfficientAgent host-tier write admission for vLLM + LMCache.

``EfficientAgentConnector`` subclasses vLLM's native ``LMCacheConnectorV1``. It changes only the storage decision
for the LMCache host (CPU) tier; LMCache's lookup, restore, eviction and token serialization are unchanged, and a
skipped write leaves computation and decoding to the serving engine.

Admission mode (environment variable ``EA_KVTIER_ADMISSION``):

  none         Offload without write admission: behaves exactly like ``LMCacheConnectorV1``.
  fixed        Fixed write admission: the request filter below applies to every request (p_t = 1), together with
               copy-time deduplication.
  conditioned  Capacity-conditioned admission: the request filter applies only under pressure p_t, together with
               copy-time deduplication.

Request filter (scheduler side). The decision is made once per request, at the request's first resolved prefix
lookup, which the vLLM scheduler issues when a scheduling step considers the waiting request. For a request with n
prompt tokens of which the host tier already holds the first h, u = floor(n / b) - floor(h / b) is the number of new
full chunks (b = LMCache chunk size). The request is saved unless p_t and u > kappa; otherwise LMCache's
request-level ``lmcache.skip_save`` switch is set, and the request stores none of its new KV. The choice is kept for
later lookups of the same request, including lookups after preemption.

Pressure (conditioned mode)::

    p_t = [C_reuse > C_H] and [o_t >= theta and e_t > 0],     C_reuse = (A_t - 1) * N_t * beta

  A_t      tasks (``efficientagent.task_id`` in the request's kv_transfer_params) with a first prefix lookup in the
           last ``EA_KVTIER_WINDOW_S`` seconds;
  N_t      mean prompt length of the last ``EA_KVTIER_PROMPT_WINDOW`` first lookups;
  beta     KV bytes per token per rank (``EA_KVTIER_BYTES_PER_TOKEN``);
  C_H      host capacity per rank (``lmcache.max_local_cpu_size`` GiB);
  o_t, e_t tier telemetry: occupancy (registered chunks / chunk capacity K_H) and the number of chunks evicted in
           the last ``EA_KVTIER_WINDOW_S`` seconds to make room for new ones.

The LMCache worker of the first GPU (TP rank 0) publishes the telemetry every ``EA_KVTIER_TELEMETRY_INTERVAL_S``
seconds to the JSON file ``EA_KVTIER_TELEMETRY_FILE``; the scheduler re-reads it at most every
``EA_KVTIER_TELEMETRY_READ_S`` seconds. When no fresh report exists (older than ``EA_KVTIER_TELEMETRY_MAX_AGE_S``, not
yet published, or no telemetry file configured), the first factor alone decides. LMCache 0.3.12 evicts from its CPU
tier only inside allocation, and e_t counts exactly those evictions (``batched_remove`` with ``force=False``).

Copy-time deduplication (worker side). Before each GPU->CPU copy in ``LMCacheEngine.store``, the chunk keys LMCache
selects for storage are checked against the storage backend; the absent keys are grouped into contiguous runs and
each run is passed to LMCache's native ``store`` with its original GPU KV locations (slot mapping).

Parameters and defaults (environment variables):

  EA_KVTIER_WRITE_THRESHOLD        kappa, full chunks                              8
  EA_KVTIER_OCCUPANCY_THRESHOLD    theta                                           0.95
  EA_KVTIER_WINDOW_S               task-activity and eviction window, seconds      60
  EA_KVTIER_PROMPT_WINDOW          first lookups averaged for N_t                  256
  EA_KVTIER_TELEMETRY_INTERVAL_S   telemetry publication interval, seconds         1
  EA_KVTIER_TELEMETRY_READ_S       scheduler telemetry re-read interval, seconds   0.5
  EA_KVTIER_TELEMETRY_MAX_AGE_S    maximum age of a fresh report, seconds          5
  EA_KVTIER_BYTES_PER_TOKEN        beta, bytes per token per rank                  24576
  LMCACHE_CHUNK_SIZE               b, tokens per chunk                             1024

The ``EA_KVTIER_BYTES_PER_TOKEN`` default is the BF16 footprint of Qwen3-Coder-30B-A3B-Instruct at TP8
(2 x 48 layers x 1 KV head per rank x 128 x 2 bytes); ``efficientagent.kvtier.policies.kv_bytes_per_token`` computes
it for other models. Counters are written to ``EA_KVTIER_STATS_DIR/<role>-<pid>.json`` (every 30 s and at exit), and
the first occurrence of each event logs one ``EA_KVTIER first-hit <EVENT>`` line.

Select the connector with::

    --kv-transfer-config '{"kv_connector": "EfficientAgentConnector",
                           "kv_connector_module_path": "efficientagent.kvtier.connector",
                           "kv_role": "kv_both",
                           "kv_connector_extra_config": {"lmcache.local_cpu": true,
                                                         "lmcache.max_local_cpu_size": 5.0}}'
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from collections import Counter, deque
from pathlib import Path

from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import LMCacheConnectorV1

log = logging.getLogger('vllm')

MODES = ('none', 'fixed', 'conditioned')
ADMISSION = os.environ.get('EA_KVTIER_ADMISSION', 'none').strip().lower()
if ADMISSION not in MODES:
    raise ValueError(f'EA_KVTIER_ADMISSION must be one of {MODES}, got {ADMISSION!r}')
KAPPA = int(os.environ.get('EA_KVTIER_WRITE_THRESHOLD', '8'))
THETA = float(os.environ.get('EA_KVTIER_OCCUPANCY_THRESHOLD', '0.95'))
WINDOW_S = float(os.environ.get('EA_KVTIER_WINDOW_S', '60'))
PROMPT_WINDOW = int(os.environ.get('EA_KVTIER_PROMPT_WINDOW', '256'))
TELEMETRY_INTERVAL_S = float(os.environ.get('EA_KVTIER_TELEMETRY_INTERVAL_S', '1'))
TELEMETRY_READ_S = float(os.environ.get('EA_KVTIER_TELEMETRY_READ_S', '0.5'))
TELEMETRY_MAX_AGE_S = float(os.environ.get('EA_KVTIER_TELEMETRY_MAX_AGE_S', '5'))
BYTES_PER_TOKEN = int(os.environ.get('EA_KVTIER_BYTES_PER_TOKEN', '24576'))
CHUNK = int(os.environ.get('LMCACHE_CHUNK_SIZE', '1024'))
TELEMETRY_FILE = os.environ.get('EA_KVTIER_TELEMETRY_FILE')
DECISION_CACHE = 65536   # bound on remembered per-request decisions (oldest dropped first)


def capacity_chunks(gib: float) -> int:
    """Chunk capacity K_H of a host tier of ``gib`` GiB per rank."""
    return int(gib * 2**30 // (CHUNK * BYTES_PER_TOKEN))


class Stats:
    """Per-process counters and first-hit markers, flushed atomically to a JSON file."""

    def __init__(self, role: str):
        self.c = Counter(); self.first = set(); self.role = role; self.lock = threading.Lock(); self.last = 0.0
        d = os.environ.get('EA_KVTIER_STATS_DIR'); self.path = Path(d) / f'{role}-{os.getpid()}.json' if d else None
        atexit.register(self.flush)

    def hit(self, event, **counts):
        with self.lock:
            for k, v in counts.items(): self.c[k] += v
            if event and event not in self.first:
                self.first.add(event); log.warning('EA_KVTIER first-hit %s role=%s pid=%d counts=%s', event, self.role, os.getpid(), dict(counts))
        if time.monotonic() - self.last > 30: self.flush()

    def flush(self):
        if not self.path: return
        with self.lock:
            self.last = time.monotonic()
            data = dict(role=self.role, pid=os.getpid(), admission=ADMISSION, write_threshold=KAPPA, counters=dict(self.c),
                        first_hits=sorted(self.first), time=time.time())
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True); tmp = self.path.with_suffix('.tmp'); tmp.write_text(json.dumps(data)); tmp.replace(self.path)
        except OSError: pass


def set_skip_save(request) -> None:
    """Set LMCache's request-level ``lmcache.skip_save`` switch in the request's kv_transfer_params."""
    sp = request.sampling_params; extra = dict(sp.extra_args or {}); params = dict(extra.get('kv_transfer_params') or {})
    params['lmcache.skip_save'] = True; extra['kv_transfer_params'] = params; sp.extra_args = extra


def install_dedup(engine, stats: Stats) -> None:
    """Wrap ``engine.store`` so that chunks already present in the storage backend are not copied again."""
    orig = engine.store; td = engine.token_database; sm = engine.storage_manager

    def store(tokens=None, hashes=None, offsets=None, mask=None, **kw):
        if tokens is None or mask is None or 'slot_mapping' not in kw:
            return orig(tokens, hashes, offsets, mask, **kw)
        chunks = list(td.process_tokens(tokens=tokens, mask=mask, request_configs=kw.get('request_configs')))
        present = [sm.contains(k) is not None for _, _, k in chunks]
        n_skip = sum(present)
        stats.hit('DEDUP' if n_skip else None, store_calls=1, store_chunks_considered=len(chunks), dedup_skipped_chunks=n_skip)
        if not n_skip: return orig(tokens, hashes, offsets, mask, **kw)
        runs = []; cur = None
        for (s, e, _), p in zip(chunks, present):
            if p: cur = None; continue
            if cur is None: cur = [s, e]; runs.append(cur)
            else: cur[1] = e
        import torch
        slot = kw['slot_mapping']
        for s, e in runs:
            m = torch.zeros(e, dtype=torch.bool); m[s:e] = True
            stats.hit(None, store_runs_issued=1, store_run_tokens=e - s)
            orig(tokens[:e], None, None, m, **dict(kw, slot_mapping=slot[:e], offset=s))
        return None
    engine.store = store


def install_telemetry_publisher(engine, stats: Stats) -> None:
    """Count capacity evictions of the CPU backend and publish {t, evicted_in_window, registered, capacity}."""
    sm = engine.storage_manager; be = getattr(sm, 'local_cpu_backend', None) or sm.storage_backends.get('LocalCPUBackend')
    if be is None or not TELEMETRY_FILE: return
    events = deque(); lock = threading.Lock(); orig = be.batched_remove

    def batched_remove(keys, force=True):
        n = orig(keys, force)
        if not force:
            with lock: events.append((time.monotonic(), len(keys)))
            stats.hit(None, cpu_evicted_chunks=len(keys))
        return n
    be.batched_remove = batched_remove
    cap = capacity_chunks(float(getattr(be.config, 'max_local_cpu_size', 0) or 0))

    def loop():
        while True:
            time.sleep(TELEMETRY_INTERVAL_S); now = time.monotonic()
            with lock:
                while events and events[0][0] < now - WINDOW_S: events.popleft()
                ev = sum(n for _, n in events)
            row = dict(t=time.time(), mono=now, evicted_in_window=ev, registered=len(be.hot_cache), capacity=cap)
            try:
                tmp = TELEMETRY_FILE + '.tmp'
                with open(tmp, 'w') as f: json.dump(row, f)
                os.replace(tmp, TELEMETRY_FILE)
            except OSError: pass
    threading.Thread(target=loop, daemon=True, name='ea-kvtier-telemetry').start()


class EfficientAgentConnector(LMCacheConnectorV1):
    """``LMCacheConnectorV1`` with host-tier write admission selected by ``EA_KVTIER_ADMISSION``."""

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        name = getattr(role, 'name', str(role)).lower()
        self._ea_stats = Stats('scheduler' if 'scheduler' in name else 'worker')
        self._ea_decided = {}
        self._ea_tasks = {}; self._ea_lens = deque(maxlen=PROMPT_WINDOW); self._ea_telemetry = (0.0, None)
        cfg = (vllm_config.kv_transfer_config.kv_connector_extra_config or {})
        self._ea_host_bytes = float(cfg.get('lmcache.max_local_cpu_size', os.environ.get('LMCACHE_MAX_LOCAL_CPU_SIZE', 5.0))) * 2**30
        impl = self._lmcache_engine; eng = getattr(impl, 'lmcache_engine', None)
        if self._ea_stats.role == 'worker' and eng is not None:
            if ADMISSION != 'none': install_dedup(eng, self._ea_stats)
            orig_fail = impl.record_failed_blocks
            def record_failed_blocks(*a, **k):
                blocks = orig_fail(*a, **k); self._ea_stats.hit('LOAD_FAILURE_RECOMPUTE', load_failures=1, load_failed_blocks=len(blocks or ())); return blocks
            impl.record_failed_blocks = record_failed_blocks
            if ADMISSION == 'conditioned':
                try:
                    from vllm.distributed import get_tensor_model_parallel_rank
                    rank = get_tensor_model_parallel_rank()
                except Exception:
                    rank = int(getattr(getattr(eng, 'metadata', None), 'worker_id', 0) or 0)
                if rank == 0: install_telemetry_publisher(eng, self._ea_stats)
        self._ea_stats.hit(None, connector_init=1); self._ea_stats.flush()
        log.warning('EA_KVTIER connector role=%s admission=%s write_threshold=%d', self._ea_stats.role, ADMISSION, KAPPA)

    def _pressure(self):
        """-> (p_t, full_and_evicting, estimate_above_capacity, fresh)."""
        now = time.time()
        if now - self._ea_telemetry[0] > TELEMETRY_READ_S:
            row = None
            try:
                with open(TELEMETRY_FILE) as f: row = json.load(f)
            except (OSError, ValueError, TypeError): row = None
            self._ea_telemetry = (now, row)
        row = self._ea_telemetry[1]
        cutoff = now - WINDOW_S
        for t in [t for t, ts in self._ea_tasks.items() if ts < cutoff]: self._ea_tasks.pop(t, None)
        n = len(self._ea_tasks); nbar = (sum(self._ea_lens) / len(self._ea_lens)) if self._ea_lens else 0.0
        above = (n - 1) * nbar * BYTES_PER_TOKEN > self._ea_host_bytes
        fresh = row is not None and now - row.get('t', 0) <= TELEMETRY_MAX_AGE_S
        full_evicting = bool(fresh and row['evicted_in_window'] > 0 and row['registered'] >= THETA * row['capacity'])
        return bool(above and (full_evicting if fresh else True)), full_evicting, above, fresh

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        rid = request.request_id
        if ADMISSION == 'none' or result[0] is None or rid in self._ea_decided: return result
        extra = {}
        if ADMISSION == 'conditioned':
            params = ((request.sampling_params.extra_args or {}).get('kv_transfer_params') or {}) if request.sampling_params else {}
            task = params.get('efficientagent.task_id')
            if task: self._ea_tasks[task] = time.time()
            self._ea_lens.append(request.num_tokens)
            pressure, full_evicting, above, fresh = self._pressure()
            extra = dict(pressure_requests=int(pressure), no_pressure_requests=int(not pressure), estimate_above_capacity=int(above),
                         telemetry_full_evicting=int(full_evicting), telemetry_stale=int(not fresh))
        else:
            pressure = True
        spec = self._lmcache_engine.load_specs.get(rid); hit = spec.lmcache_cached_tokens if spec is not None else 0
        new_chunks = request.num_tokens // CHUNK - hit // CHUNK
        skip = bool(pressure and new_chunks > KAPPA)
        self._ea_decided[rid] = skip
        if len(self._ea_decided) > DECISION_CACHE: self._ea_decided.pop(next(iter(self._ea_decided)))
        if skip: set_skip_save(request)
        event = ('PRESSURE_ON' if pressure else 'PRESSURE_OFF') if ADMISSION == 'conditioned' else None
        self._ea_stats.hit(event, decisions=1, skipped_requests=int(skip), saved_requests=int(not skip),
                           skipped_new_chunks=new_chunks if skip else 0, saved_new_chunks=0 if skip else max(0, new_chunks), **extra)
        if skip: self._ea_stats.hit('SKIP_WRITE')
        return result
