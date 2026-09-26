"""Tiny deterministic synthetic inputs for the CPU tests: replay traces and complete run directories.

``make_trace(trace_dir, n_tasks=8, n_steps=5, seed=0, edit_task=None) -> Path``
    Writes a replay trace (one ``NNN_<task>.jsonl.gz`` per task: a header line, then one line per step) in the format
    of ``efficientagent.replay.build_trace``. Every task starts with the same system prompt; each step appends the
    previous output and a new observation, so consecutive prompts share their prefix. With ``edit_task=i`` task i
    rewrites part of its history once (the next prompt diverges inside the previous prompt).

``make_run(run_dir, trace_dir, admission='none', host_gib=0.25, workers=3, tp=2, write_threshold=2,
           bytes_per_token=24576, gpu_chunks=4, theta=0.95, window_s=60.0, prompt_window=256, chunk=1024) -> dict``
    Simulates a replay of the trace (tasks in order on ``workers`` slots, recorded gaps, fixed service times) with an
    LRU host tier of ``chunks_of(host_gib)`` chunks and an LRU GPU prefix cache of ``gpu_chunks`` chunks, following the
    reference model of the analyses (at each lookup every full prompt chunk is referenced tail first; a miss is
    inserted unless the request's write is declined). Admission ``none`` / ``fixed`` / ``conditioned`` applies the
    connector's rule (conditioned: working-set estimate above capacity and the latest 5-s telemetry report full and
    evicting). ``host_gib=0`` gives a run without host tier. Writes run.json, server.log (LMCache INFO lines for
    lookups, stores and restores, one line per rank), metrics_start.prom / metrics_end.prom, load_samples.jsonl
    (rows {time, phase: 'replay', vllm: {...}}), replay/requests.jsonl, replay/tasks.jsonl, replay/replay_summary.json,
    kvtier_stats/*.json (admission runs), telemetry_history.jsonl (conditioned) and lmcache_observer/cpu-<pid>.jsonl.
    Returns dict(spec, truth) where truth holds the simulated host hits, GPU hits, declined calls, counters and
    per-rank volumes.
"""
from __future__ import annotations

import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

T0_MS = int(datetime.datetime(2025, 1, 1, 8, 0, 0).timestamp()) * 1000


def _tokens(rng, n):
    return [int(x) for x in rng.integers(100, 50000, n)]


def make_trace(trace_dir, n_tasks: int = 8, n_steps: int = 5, seed: int = 0, edit_task: int | None = None,
               system_tokens: int = 1100, first_tokens=(1500, 2500), step_tokens=(600, 2200), output_tokens=(20, 160),
               gap_s=(1.0, 6.0)) -> Path:
    rng = np.random.default_rng(seed)
    d = Path(trace_dir); d.mkdir(parents=True, exist_ok=True)
    system = _tokens(rng, system_tokens)
    for k in range(n_tasks):
        inst = f'task-{k:03d}'
        prompt = system + _tokens(rng, int(rng.integers(*first_tokens)))
        steps = []; prev = []
        for i in range(n_steps):
            out = _tokens(rng, int(rng.integers(*output_tokens)))
            shared = 0; n = min(len(prompt), len(prev))
            while shared < n and prompt[shared] == prev[shared]: shared += 1
            steps.append(dict(seq=i + 1, gap_before_s=round(float(rng.uniform(*gap_s)), 3), prompt_shared=shared, prompt_suffix=prompt[shared:],
                              prompt_len=len(prompt), output=out, output_len=len(out), recorded_client_elapsed_s=1.0,
                              prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest()))
            prev = prompt
            nxt = prompt + out + _tokens(rng, int(rng.integers(*step_tokens)))
            if edit_task == k and i == n_steps // 2:
                cut = system_tokens + 10
                nxt = nxt[:cut] + _tokens(rng, 300) + nxt[cut + 300:]
            prompt = nxt
        tail = 2.0
        with gzip.open(d / f'{k:03d}_{inst}.jsonl.gz', 'wt') as z:
            z.write(json.dumps(dict(instance_id=inst, order=k, tail_gap_s=tail, recorded_jct_s=60.0, calls_total=n_steps, calls_replayed=n_steps)) + '\n')
            for s in steps: z.write(json.dumps(s) + '\n')
    return d


def _load(trace_dir):
    tasks = []
    for f in sorted(Path(trace_dir).glob('*.jsonl.gz')):
        with gzip.open(f, 'rt') as z:
            head = json.loads(z.readline()); steps = [json.loads(line) for line in z]
        prompt = []; calls = []
        for s in steps:
            prompt = prompt[:s['prompt_shared']] + s['prompt_suffix']
            calls.append(dict(seq=s['seq'], gap=s['gap_before_s'], prompt=list(prompt), plen=s['prompt_len'], olen=s['output_len']))
        tasks.append(dict(inst=head['instance_id'], order=head['order'], tail=head['tail_gap_s'], calls=calls))
    return tasks


def _keys(prompt, chunk):
    keys = []; h = b''
    for c in range(len(prompt) // chunk):
        h = hashlib.blake2b(h + np.asarray(prompt[c * chunk:(c + 1) * chunk], dtype=np.int32).tobytes(), digest_size=12).digest()
        keys.append(h)
    return keys


def _stamp(t_ms: int) -> str:
    return datetime.datetime.fromtimestamp(t_ms // 1000).strftime('%Y-%m-%d %H:%M:%S') + ',%03d' % (t_ms % 1000)


def make_run(run_dir, trace_dir, admission: str = 'none', host_gib: float = 0.25, workers: int = 3, tp: int = 2,
             write_threshold: int = 2, bytes_per_token: int = 24576, gpu_chunks: int = 4, theta: float = 0.95,
             window_s: float = 60.0, prompt_window: int = 256, chunk: int = 1024) -> dict:
    d = Path(run_dir); (d / 'replay').mkdir(parents=True, exist_ok=True)
    tasks = _load(trace_dir)
    K = int(host_gib * 2 ** 30 // (chunk * bytes_per_token)); host = host_gib > 0
    # ---- timeline (independent of the caches): tasks in order on `workers` slots, fixed service times, integer ms
    free = [T0_MS] * workers; calls = []; task_rows = []
    for t in tasks:
        w = int(np.argmin(free)); now = free[w]; assigned = now
        for c in t['calls']:
            start = now + int(round(c['gap'] * 1000)) + t['order']
            start += start % 2          # lookups on even ms, telemetry ticks on odd ms: no ties
            dur = 500 + c['plen'] // 20 + c['olen'] * 10
            calls.append(dict(inst=t['inst'], seq=c['seq'], order=t['order'], worker=w, start=start, end=start + dur, plen=c['plen'],
                              olen=c['olen'], keys=_keys(c['prompt'], chunk)))
            now = start + dur
        now += int(t['tail'] * 1000); free[w] = now
        task_rows.append(dict(instance_id=t['inst'], order=t['order'], worker=w, assigned=assigned / 1000, finished=now / 1000,
                              jct_s=(now - assigned) / 1000, recorded_jct_s=60.0, steps=len(t['calls']), ok=True))
    end_ms = max(free)
    # ---- event loop: telemetry ticks (5 s), observer samples (10 s) and lookups, in time order (ticks first at equal times)
    events = [(c['start'] + 10, 1, i) for i, c in enumerate(calls)]
    events += [(T0_MS - 1 + 5000 * k, 0, 'tick') for k in range((end_ms - T0_MS) // 5000 + 2)]
    events += [(T0_MS + 10000 * k, 0, 'obs') for k in range((end_ms - T0_MS) // 10000 + 2)]
    events.sort(key=lambda e: (e[0], e[1]))
    lru = collections.OrderedDict(); gpu = collections.OrderedDict(); evictions = []
    hist, hist_ms, obs = [], [], []; log = []; recent_tasks = {}; lens = collections.deque()
    cnt = collections.Counter(); truth = dict(host_hit={}, gpu_hit={}, declined=set(), retrieved_per_rank=0, stored_per_rank=0, prefill_computed=0)
    first_hits = set()
    for t_ms, kind, what in events:
        if kind == 0:
            if what == 'tick' and admission == 'conditioned':
                ev = sum(1 for e in evictions if t_ms - window_s * 1000 <= e <= t_ms)
                hist.append(dict(t=t_ms / 1000, mono=(t_ms - T0_MS) / 1000, evicted_in_window=ev, registered=len(lru), capacity=K, read_t=t_ms / 1000))
                hist_ms.append(t_ms)
            elif what == 'obs' and host:
                obs.append(dict(t=t_ms / 1000, keys=len(lru), pinned=0, evicted=len([e for e in evictions if e <= t_ms]),
                                evict_calls=len([e for e in evictions if e <= t_ms]), capacity_gb=host_gib, rss_bytes=1 << 30))
            continue
        c = calls[what]; call = (c['inst'], c['seq']); keys = c['keys']; m = len(keys); n = c['plen']
        hc = 0
        while host and hc < m and keys[hc] in lru: hc += 1
        hg = 0
        while hg < m and keys[hg] in gpu: hg += 1
        g = hg * chunk
        truth['host_hit'][call] = hc * chunk; truth['gpu_hit'][call] = g
        u = n // chunk - hc
        skip = False
        if host and admission in ('fixed', 'conditioned'):
            extra = {}
            if admission == 'conditioned':
                recent_tasks[c['inst']] = t_ms
                lens.append(n)
                if len(lens) > prompt_window: lens.popleft()
                A = sum(1 for v in recent_tasks.values() if v >= t_ms - window_s * 1000); nbar = sum(lens) / len(lens)
                above = (A - 1) * nbar * bytes_per_token > host_gib * 2 ** 30
                j = next((i for i in range(len(hist_ms) - 1, -1, -1) if hist_ms[i] <= t_ms), None)
                row = hist[j] if j is not None else None
                fresh = row is not None and t_ms - hist_ms[j] <= 5000
                full_ev = bool(fresh and row['evicted_in_window'] > 0 and row['registered'] >= theta * row['capacity'])
                pressure = bool(above and (full_ev if fresh else True))
                extra = dict(pressure_requests=int(pressure), no_pressure_requests=int(not pressure), estimate_above_capacity=int(above),
                             telemetry_full_evicting=int(full_ev), telemetry_stale=int(not fresh))
                first_hits.add('PRESSURE_ON' if pressure else 'PRESSURE_OFF')
            else:
                pressure = True
            skip = bool(pressure and u > write_threshold)
            cnt.update(dict(decisions=1, skipped_requests=int(skip), saved_requests=int(not skip), skipped_new_chunks=u if skip else 0,
                            saved_new_chunks=0 if skip else max(0, u), **extra))
            if skip: truth['declined'].add(call); first_hits.add('SKIP_WRITE')
        if host:
            for r in range(tp):
                log.append((t_ms, f'Reqid: req-{what}, Total tokens {n}, LMCache hit tokens: {hc * chunk}, need to load: {hc * chunk - g}'))
            gal = (g // chunk) * chunk
            if hc * chunk > gal:
                R = hc * chunk - gal; truth['retrieved_per_rank'] += R
                for r in range(tp):
                    log.append((t_ms + 5, f'Retrieved {R} out of {R} required tokens (from {hc * chunk} total tokens). size: 0.01 gb, cost 0.50 ms'))
            Y = (n // chunk) * chunk; S = hc * chunk
            if not skip and Y > S:
                truth['stored_per_rank'] += Y - S; cnt['store_calls_worker'] += 1
                for r in range(tp):
                    log.append((t_ms + 10, f'Storing KV cache for {Y - S} out of {Y} tokens (skip_leading_tokens={S}) for request req-{what}'))
                for r in range(tp):
                    log.append((t_ms + 20, f'Stored {Y - S} out of total {Y} tokens. size: 0.01 gb, cost 1.00 ms'))
            for i in reversed(range(m)):
                k = keys[i]
                if k in lru: lru.move_to_end(k)
                elif not (skip and hc <= i < n // chunk):
                    lru[k] = True
                    if len(lru) > K: lru.popitem(last=False); evictions.append(t_ms)
        for i in reversed(range(m)):
            k = keys[i]
            if k in gpu: gpu.move_to_end(k)
            else:
                gpu[k] = True
                if len(gpu) > gpu_chunks: gpu.popitem(last=False)
        truth['prefill_computed'] += n - max(g, hc * chunk)
    # ---- files
    spec = dict(admission=admission, host_gib=host_gib, use_connector=False, workers=workers, tp=tp, model='synthetic',
                served_model_name='synthetic', write_threshold=write_threshold, bytes_per_token=bytes_per_token, chunk_tokens=chunk,
                trace_dir=str(trace_dir))
    reqs = [dict(instance_id=c['inst'], seq=c['seq'], worker=c['worker'], start=c['start'] / 1000, end=c['end'] / 1000, prompt_len=c['plen'],
                 output_len=c['olen'], usage=dict(prompt_tokens=c['plen'], completion_tokens=c['olen']), finish_reason='length',
                 forced_match=True, prompt_ok=True, latency_s=(c['end'] - c['start']) / 1000) for c in sorted(calls, key=lambda c: c['start'])]
    summary = dict(requests=len(reqs), forced_match=len(reqs), forced_mismatch=0, errors=0, prompt_len_mismatch=0, tasks=len(tasks), workers=workers,
                   gap_scale=1.0, start=T0_MS / 1000, end=end_ms / 1000, makespan_s=(end_ms - T0_MS) / 1000, all_outputs_forced_exactly=True,
                   interrupted=False)
    with open(d / 'replay/requests.jsonl', 'w') as f:
        for r in reqs: f.write(json.dumps(r) + '\n')
    with open(d / 'replay/tasks.jsonl', 'w') as f:
        for r in task_rows: f.write(json.dumps(r) + '\n')
    (d / 'replay/replay_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (d / 'run.json').write_text(json.dumps(dict(spec=spec, status='passed', errors=[], replay=summary), indent=1) + '\n')
    lines = [f'[{_stamp(t)}] LMCache INFO: {msg}' for t, msg in sorted(log, key=lambda x: x[0])]
    lines += [f'EA_KVTIER first-hit {h} role=scheduler pid=1 counts={{}}' for h in sorted(first_hits)]
    (d / 'server.log').write_text('\n'.join(lines) + '\n')
    n_req = len(reqs); ptok = sum(c['plen'] for c in calls); otok = sum(c['olen'] for c in calls)
    lab = '{engine="0",model_name="synthetic"}'
    start_prom = [f'vllm:request_prefill_kv_computed_tokens_sum{lab} 0.0', f'vllm:prompt_tokens_total{lab} 0.0']
    end_prom = [f'vllm:request_prefill_kv_computed_tokens_sum{lab} {float(truth["prefill_computed"])}',
                f'vllm:request_prefill_kv_computed_tokens_count{lab} {float(n_req)}',
                f'vllm:prompt_tokens_total{lab} {float(ptok)}', f'vllm:generation_tokens_total{lab} {float(otok)}',
                f'vllm:request_success_total{lab} {float(n_req)}', f'vllm:num_preemptions_total{lab} 0.0',
                f'vllm:prefix_cache_queries_total{lab} {float(ptok)}', f'vllm:prefix_cache_hits_total{lab} {float(sum(truth["gpu_hit"].values()))}',
                f'vllm:request_queue_time_seconds_sum{lab} {0.1 * n_req}', f'vllm:request_queue_time_seconds_count{lab} {float(n_req)}',
                'vllm:request_queue_time_seconds_bucket{engine="0",le="0.1"} %s' % float(n_req),
                'vllm:request_queue_time_seconds_bucket{engine="0",le="+Inf"} %s' % float(n_req),
                f'vllm:e2e_request_latency_seconds_sum{lab} {sum(r["latency_s"] for r in reqs)}', f'vllm:e2e_request_latency_seconds_count{lab} {float(n_req)}',
                f'vllm:request_prefill_time_seconds_sum{lab} {0.2 * n_req}', f'vllm:request_prefill_time_seconds_count{lab} {float(n_req)}',
                f'vllm:request_decode_time_seconds_sum{lab} {0.01 * otok}', f'vllm:request_decode_time_seconds_count{lab} {float(n_req)}',
                f'vllm:time_to_first_token_seconds_sum{lab} {0.3 * n_req}', f'vllm:time_to_first_token_seconds_count{lab} {float(n_req)}']
    (d / 'metrics_start.prom').write_text('\n'.join(['# HELP synthetic'] + start_prom) + '\n')
    (d / 'metrics_end.prom').write_text('\n'.join(['# HELP synthetic'] + end_prom) + '\n')
    with open(d / 'load_samples.jsonl', 'w') as f:
        for t_ms in range(T0_MS, end_ms + 1, 2000):
            running = sum(1 for c in calls if c['start'] <= t_ms < c['end'])
            f.write(json.dumps(dict(time=t_ms / 1000, phase='replay', vllm={'vllm:num_requests_running': running, 'vllm:num_requests_waiting': 0,
                                                                              'vllm:kv_cache_usage_perc': 0.5})) + '\n')
    if host and admission in ('fixed', 'conditioned'):
        (d / 'kvtier_stats').mkdir(exist_ok=True)
        sched = {k: v for k, v in cnt.items() if k != 'store_calls_worker'}
        sched['connector_init'] = 1
        (d / 'kvtier_stats/scheduler-1.json').write_text(json.dumps(dict(role='scheduler', pid=1, admission=admission, write_threshold=write_threshold,
                                                                          counters=sched, first_hits=sorted(first_hits))))
        for r in range(tp):
            wc = dict(connector_init=1, store_calls=cnt['store_calls_worker'], store_chunks_considered=truth['stored_per_rank'] // chunk,
                      dedup_skipped_chunks=0)
            if r == 0 and admission == 'conditioned': wc['cpu_evicted_chunks'] = len(evictions)
            (d / f'kvtier_stats/worker-{2 + r}.json').write_text(json.dumps(dict(role='worker', pid=2 + r, admission=admission, counters=wc)))
        if admission == 'conditioned':
            (d / 'kvtier_stats/telemetry_rank0.json').write_text(json.dumps(hist[-1]))
    if admission == 'conditioned' and host:
        with open(d / 'telemetry_history.jsonl', 'w') as f:
            for h in hist: f.write(json.dumps(h) + '\n')
    if host:
        (d / 'lmcache_observer').mkdir(exist_ok=True)
        for r in range(tp):
            with open(d / f'lmcache_observer/cpu-{100 + r}.jsonl', 'w') as f:
                for o in obs: f.write(json.dumps(o) + '\n')
    truth['counters'] = dict(cnt); truth['evictions'] = len(evictions); truth['capacity_chunks'] = K
    return dict(spec=spec, truth=truth)
