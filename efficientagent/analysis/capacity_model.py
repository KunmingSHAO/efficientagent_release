"""Stack-distance capacity model of the LMCache host tier (CPU only).

Reference stream: every request references all full chunks of its prompt at its scheduling lookup, tail first and
head last, with prefix-dependent chunk keys. The key set of a lookup does not depend on the host capacity, so one
single-pass LRU stack-distance computation (Mattson et al.) serves every capacity: a chunk is resident in a tier of C
chunks iff its stack distance is < C, and a request's host coverage is the leading run of resident chunks. Useful
restoration is the host coverage above the request's GPU-resident prefix g (from the run's logged lookup,
g = hit - need), and computed prefill is the prompt length minus max(g, host coverage).

Subcommands:

``curve``  For each run with a host tier (its recorded scheduling order): predicted host coverage, restored tokens
           and computed prefill over a grid of host capacities; the prediction at the run's own capacity next to
           the logged values; reuse-distance distributions and the capacity that reaches a given share of the
           maximum prefix coverage; the working-set scale (A - 1) * mean prompt * beta. With ``--noqueue-scales``
           the same statistics are computed on synthetic timelines without queueing (tasks in trace order on A
           slots, recorded gaps, service = new prompt tokens / prefill rate + output tokens * decode time).

``pool``   Prediction for a different active pool A. The per-request GPU prefix capacity is taken from reference
           runs (the interval of LRU capacities that reproduces each logged GPU hit), and request orders come from a
           closed-loop queue model (A agents, R engine slots) and from a timeline without queueing. The result is an
           interval over timelines and reference runs.

Examples::

    python -m efficientagent.analysis.capacity_model curve --trace TRACE --run runs/offload_5 --run runs/offload_40 \\
        --out capacity_curve.json
    python -m efficientagent.analysis.capacity_model pool --trace TRACE --run runs/offload_5 --run runs/offload_40 \\
        --active-pool 8 --engine-slots 3 5 --prefill-tok-s PF --decode-s-per-token D --host-gib 2 3 4 5 10 \\
        --out pool8.json
"""
from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import numpy as np

from efficientagent.analysis import common as C
from efficientagent.analysis.runsets import RunInfo, fmt_gib, makespan, prefill_computed, resolve_run, write_json

DEFAULT_GRID = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 120, 160]
COVERAGE_Q = (0.5, 0.8, 0.9, 0.95)
OPEN_SPREAD, OPEN_OFFSET = 1.5, 50   # spread of an open-ended implied GPU capacity interval: [lo, 1.5 * lo + 50] chunks


# ---------------------------------------------------------------------------------------------------- shared pieces
def run_lookups(run: RunInfo, tmap: dict) -> dict:
    """Scheduling lookups of a run plus logged host volumes per rank (LMCache log lines are per rank)."""
    look, storing, _, retr = C.parse_lmcache_log(run.dir / 'server.log')
    M = C.map_lookups(run.dir, tmap, look, storing)
    st = C.group_events(storing, key=lambda x: (x[1], x[2], x[3]), size=run.ranks)
    rt = C.group_events(retr, key=lambda x: (x[1], x[2]), size=run.ranks)
    M['stored_per_rank'] = sum(ev['lines'][0][2] - ev['lines'][0][3] for ev in st)
    M['retrieved_per_rank'] = sum(ev['lines'][0][1] for ev in rt)
    return M


def gpu_hit_fn(final: dict):
    """g = logged GPU-resident prefix (hit - need) for requests with a host hit, else unknown (None)."""
    return lambda call: (final[call][1] - final[call][2]) if final[call][1] > 0 else None


def predict(sd: dict, calls: list, tmap: dict, g_of, cap_chunks: int, chunk: int) -> dict:
    hit_tok = retr = comp = 0
    for call in calls:
        h = C.prefix_hit_chunks(sd[call], cap_chunks); plen = tmap[call]['plen']
        g = g_of(call)
        hit_tok += h * chunk
        if g is not None:
            gal = (g // chunk) * chunk
            retr += max(0, h * chunk - gal); comp += plen - max(g, h * chunk)
    return dict(cpu_prefix_hit_tokens=hit_tok, retrieved=retr, prefill_computed_pred=comp)


def no_queue_order(tasks: list, workers: int, prefill_tok_s: float | None, decode_s_per_token: float | None, scale: float = 1.0):
    """Timeline without queueing: tasks in trace order on ``workers`` slots, recorded gaps, unloaded service."""
    free = [0.0] * workers; events = []
    for t in tasks:
        w = int(np.argmin(free)); now = free[w]
        for c in t['calls']:
            start = now + c['gap']
            svc = scale * ((c['plen'] - c['shared']) / prefill_tok_s + c['olen'] * decode_s_per_token) if scale else 0.0
            events.append((start, (t['inst'], c['seq']))); now = start + svc
        free[w] = now
    events.sort(); return [c for _, c in events], max(free)


def timeline_stats(order: list, sd: dict, tmap: dict, grid: list, chunk: int, bytes_per_token: int, pool: int,
                   plen_mean: float, mk=None) -> dict:
    chunk_gib = chunk * bytes_per_token / C.GIB
    a = np.concatenate([sd[c] for c in order]); warm = a[np.isfinite(a)]; gib = warm * chunk_gib
    tot_tokens = sum(tmap[c]['plen'] // chunk * chunk for c in order); cold = int(np.sum(~np.isfinite(a)))
    cov = {}
    for g in grid:
        K = C.chunks_of(g, bytes_per_token, chunk); h = sum(C.prefix_hit_chunks(sd[c], K) for c in order)
        cov[fmt_gib(g)] = h * chunk / tot_tokens
    need = {}
    for q in COVERAGE_Q:
        lo, hi = 1, int(np.nanmax(warm)) + 2
        while lo < hi:
            m = (lo + hi) // 2
            h = sum(C.prefix_hit_chunks(sd[c], m) for c in order)
            if h / (len(a) - cold) >= q: hi = m
            else: lo = m + 1
        need[str(q)] = lo * chunk_gib
    return dict(makespan_model_s=mk, refs=len(a), cold_refs=cold, reuse_distance_gib=C.dist(gib),
                median_reuse_distance_over_working_set=float(np.median(warm * chunk) / ((pool - 1) * plen_mean)) if pool > 1 else None,
                prefix_coverage_by_gib=cov, max_coverage=(len(a) - cold) / len(a), capacity_gib_for_frac_of_max_coverage=need)


# ---------------------------------------------------------------------------------------------------- curve
def curve(args) -> dict:
    runs = [resolve_run(r, args.bytes_per_token, args.chunk) for r in args.run]
    beta, ch = runs[0].bytes_per_token, runs[0].chunk
    tasks = C.load_trace(args.trace); keys = C.prompt_chunk_keys(tasks, ch); tmap = C.call_map(tasks)
    pool = args.active_pool or runs[0].workers
    grid = sorted(set(float(g) for g in args.grid) | {r.host_gib for r in runs if r.host_gib > 0})
    plen_mean = float(np.mean([c['plen'] for t in tasks for c in t['calls']]))
    res = dict(schema='efficientagent.capacity_curve', version=1, chunk_tokens=ch, bytes_per_token=beta,
               chunk_gib_per_rank=ch * beta / C.GIB, grid_gib=[fmt_gib(g) for g in grid], runs={}, recompute_runs={}, timelines={})
    for run in runs:
        if run.host_gib == 0:
            res['recompute_runs'][run.name] = dict(makespan_s=makespan(run), logged_prefill_computed=prefill_computed(run), active_pool=run.workers)
            continue
        R = run_lookups(run, tmap)
        order = [c for c, _ in sorted(R['final'].items(), key=lambda kv: kv[1][0])]
        sd = C.stack_distances(order, keys, tmap, ch); g_of = gpu_hit_fn(R['final'])
        cv = {fmt_gib(g): predict(sd, order, tmap, g_of, C.chunks_of(g, beta, ch), ch) for g in grid}
        p = cv[fmt_gib(run.host_gib)]; logged_pf = prefill_computed(run); K = C.chunks_of(run.host_gib, beta, ch)
        res['runs'][run.name] = dict(
            admission=run.admission, own_capacity_gib=run.host_gib, active_pool=run.workers, mapped=R['mapped'], n=R['n'],
            makespan_s=makespan(run), logged_stored_per_rank=R['stored_per_rank'], logged_retrieved_per_rank=R['retrieved_per_rank'],
            logged_prefill_computed=logged_pf, predicted_at_own=p,
            retrieved_rel_err=(p['retrieved'] - R['retrieved_per_rank']) / R['retrieved_per_rank'] if R['retrieved_per_rank'] else None,
            prefill_computed_rel_err=(p['prefill_computed_pred'] - logged_pf) / logged_pf if logged_pf else None,
            exact_hit_match_frac=float(np.mean([C.prefix_hit_chunks(sd[c], K) * ch == R['final'][c][1] for c in order if len(sd[c])])),
            predicted_without_host_tier=predict(sd, order, tmap, g_of, 0, ch)['prefill_computed_pred'],
            curve=cv)
        res['timelines'][run.name] = timeline_stats(order, sd, tmap, grid, ch, beta, run.workers, plen_mean)
    for scale in args.noqueue_scales or []:
        if scale and (args.prefill_tok_s is None or args.decode_s_per_token is None):
            raise SystemExit('--noqueue-scales with a nonzero scale needs --prefill-tok-s and --decode-s-per-token')
        order, mk = no_queue_order(tasks, pool, args.prefill_tok_s, args.decode_s_per_token, scale)
        sd = C.stack_distances(order, keys, tmap, ch)
        res['timelines'][f'noqueue_x{fmt_gib(scale)}'] = timeline_stats(order, sd, tmap, grid, ch, beta, pool, plen_mean, mk)
    q = [v for k, v in res['timelines'].items() if not k.startswith('noqueue_')]
    n = [v for k, v in res['timelines'].items() if k.startswith('noqueue_')]
    if q and n:
        med = lambda xs: [x['reuse_distance_gib']['p50'] for x in xs]
        c90 = lambda xs: [x['capacity_gib_for_frac_of_max_coverage']['0.9'] for x in xs]
        res['queue_amplification'] = dict(median_reuse_distance=[min(med(q)) / max(med(n)), max(med(q)) / min(med(n))],
                                          capacity_for_90pct=[min(c90(q)) / max(c90(n)), max(c90(q)) / min(c90(n))])
    res['workload'] = dict(active_pool=pool, mean_prompt_tokens=plen_mean, kv_bytes_per_token_rank=beta,
                           working_set_gib=(pool - 1) * plen_mean * beta / C.GIB)
    return res


def curve_md(res: dict) -> str:
    M = lambda x: f'{x / 1e6:.2f}M'; P = lambda x: '-' if x is None else f'{100 * x:.1f}%'
    L = ['# Host-tier capacity model', '',
         f"Working-set scale (A - 1) x mean prompt x beta = {res['workload']['working_set_gib']:.2f} GiB per rank "
         f"(A = {res['workload']['active_pool']}, mean prompt {res['workload']['mean_prompt_tokens']:,.0f} tokens).", '',
         '## Prediction at each run\'s own capacity', '',
         '| run | GiB/rank | restored/rank logged | predicted | computed prefill logged | predicted | exact per-request host hit |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in res['runs'].items():
        p = r['predicted_at_own']
        L.append(f"| {name} | {fmt_gib(r['own_capacity_gib'])} | {M(r['logged_retrieved_per_rank'])} | {M(p['retrieved'])} | "
                 f"{M(r['logged_prefill_computed'] or 0)} | {M(p['prefill_computed_pred'])} | {P(r['exact_hit_match_frac'])} |")
    names = list(res['runs'])
    if names:
        L += ['', '## Capacity curve (predicted restored / computed prefill per run timeline)', '',
              '| GiB/rank | ' + ' | '.join(names) + ' |', '|---:|' + '---|' * len(names)]
        for g in res['grid_gib']:
            L.append(f'| {g} | ' + ' | '.join(f"{M(res['runs'][n]['curve'][g]['retrieved'])} / {M(res['runs'][n]['curve'][g]['prefill_computed_pred'])}"
                                             for n in names) + ' |')
    L += ['', '## Timelines', '', '| timeline | reuse distance p50 / p90 / p99 GiB | GiB for 50% / 90% of max coverage |', '|---|---|---|']
    for k, v in res['timelines'].items():
        d = v['reuse_distance_gib']; c = v['capacity_gib_for_frac_of_max_coverage']
        L.append(f"| {k} | {d['p50']:.2f} / {d['p90']:.2f} / {d['p99']:.2f} | {c['0.5']:.2f} / {c['0.9']:.2f} |")
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------------------------------------------- pool
def qsim(tasks: list, pool: int, slots: int, prefill_tok_s: float, decode_s_per_token: float):
    """Closed-loop queue model: ``pool`` agents take tasks in trace order, ``slots`` FIFO engine slots serve requests
    (service = new prompt tokens / prefill rate + output tokens * decode time). Returns request order and statistics."""
    it = iter(tasks); ev = []; seq = [0]; agents = {}

    def push(t, kind, a):
        heapq.heappush(ev, (t, seq[0], kind, a)); seq[0] += 1

    def start_task(a, now):
        t = next(it, None)
        if t is not None: agents[a] = [t, 0]; push(now + t['calls'][0]['gap'], 'arrive', a)
    for a in range(pool): start_task(a, 0.0)
    free = slots; fifo = []; order = []; lat = []; qs = []; arr = {}; now = 0.0
    while ev:
        now, _, kind, a = heapq.heappop(ev)
        if kind == 'arrive': fifo.append(a); arr[a] = now
        else:
            free += 1; t, i = agents[a]; lat.append(now - arr[a]); agents[a][1] = i + 1
            if i + 1 < len(t['calls']): push(now + t['calls'][i + 1]['gap'], 'arrive', a)
            else: start_task(a, now)
        while free and fifo:
            b = fifo.pop(0); free -= 1; t, i = agents[b]; c = t['calls'][i]
            qs.append(now - arr[b]); order.append((t['inst'], c['seq']))
            push(now + (c['plen'] - c['shared']) / prefill_tok_s + c['olen'] * decode_s_per_token, 'done', b)
    return order, dict(makespan=now, lat_mean=float(np.mean(lat)), queue_mean=float(np.mean(qs)))


def implied_gpu_capacity(run: RunInfo, keys: dict, tmap: dict) -> list:
    """Per logged request with a host hit: the interval of GPU LRU capacities (chunks) reproducing its GPU hit."""
    R = run_lookups(run, tmap); ch = run.chunk
    order = [c for c, _ in sorted(R['final'].items(), key=lambda kv: kv[1][0])]
    sd = C.stack_distances(order, keys, tmap, ch); out = []
    for c in order:
        hit, need = R['final'][c][1], R['final'][c][2]
        if hit <= 0: continue
        g = min(hit - need, tmap[c]['plen']); gc = g // ch
        pm = np.maximum.accumulate(sd[c]) if len(sd[c]) else sd[c]; n = len(pm)
        if n == 0: continue
        lo = pm[gc - 1] if gc >= 1 else 0.0
        hi = pm[gc] if gc < n else np.inf
        if not np.isfinite(lo): lo = 1e6
        out.append((lo, hi))
    return out


def sample_capacity(rng, intervals: list, n: int):
    idx = rng.integers(0, len(intervals), n); lo = np.array([intervals[i][0] for i in idx]); hi = np.array([intervals[i][1] for i in idx])
    hi2 = np.where(np.isfinite(hi), hi, np.maximum(lo, 1) * OPEN_SPREAD + OPEN_OFFSET)
    return lo + (hi2 - lo) * rng.random(n)


def predict_pool(order: list, sd: dict, tmap: dict, prevfull: dict, gpu_cap, caps_gib: list, beta: int, ch: int) -> dict:
    cg = gpu_cap[:len(order)]
    res = {fmt_gib(g): dict(retrieved=0, prefill_computed=0) for g in caps_gib}
    res['none'] = dict(retrieved=0, prefill_computed=0)
    for j, c in enumerate(order):
        pm = np.maximum.accumulate(sd[c]) if len(sd[c]) else sd[c]; plen = tmap[c]['plen']; n = len(pm)
        hg = int(np.sum(pm < cg[j])); g = prevfull[c] if (hg == n and prevfull[c] > n * ch) else hg * ch
        g = min(g, plen - 1) if g >= plen else g
        res['none']['prefill_computed'] += plen - g
        for gib in caps_gib:
            hc = int(np.sum(pm < C.chunks_of(gib, beta, ch))) * ch
            r = res[fmt_gib(gib)]; r['retrieved'] += max(0, hc - (g // ch) * ch); r['prefill_computed'] += plen - max(g, hc)
    return res


def pool(args) -> dict:
    tasks = C.load_trace(args.trace); tmap = C.call_map(tasks)
    runs = [resolve_run(r, args.bytes_per_token, args.chunk) for r in args.run]
    refs = [r for r in runs if r.host_gib > 0]
    if not refs:
        raise SystemExit('pool needs at least one reference run with a host tier')
    beta, ch = refs[0].bytes_per_token, refs[0].chunk
    keys = C.prompt_chunk_keys(tasks, ch)
    rng = np.random.default_rng(args.seed)
    prevfull = {}
    for t in tasks:
        for i, c in enumerate(t['calls']):
            prevfull[(t['inst'], c['seq'])] = (t['calls'][i - 1]['plen'] + t['calls'][i - 1]['olen']) if i else 0
    cgi = {r.name: implied_gpu_capacity(r, keys, tmap) for r in refs}
    caps = sorted(set(float(g) for g in args.host_gib))
    A = args.active_pool; A_ref = refs[0].workers
    res = dict(schema='efficientagent.capacity_pool', version=1, active_pool=A, reference_pool=A_ref, engine_slots=args.engine_slots,
               prefill_tok_s=args.prefill_tok_s, decode_s_per_token=args.decode_s_per_token, seed=args.seed,
               host_gib=[fmt_gib(g) for g in caps], reference_runs=[r.name for r in refs])
    if args.validate:
        own = sorted({r.host_gib for r in refs})
        measured = {r.name: dict(host_gib=r.host_gib, prefill_computed=prefill_computed(r)) for r in runs}
        for r in refs:
            measured[r.name]['retrieved_per_rank'] = run_lookups(r, tmap)['retrieved_per_rank']
        sim = {}
        for R_ in args.engine_slots:
            o, st = qsim(tasks, A_ref, R_, args.prefill_tok_s, args.decode_s_per_token); sd = C.stack_distances(o, keys, tmap, ch)
            for v in cgi:
                p = predict_pool(o, sd, tmap, prevfull, sample_capacity(rng, cgi[v], len(o)), own, beta, ch)
                sim[f'pool{A_ref}_slots{R_}_gpu:{v}'] = dict(timeline=st, no_host_prefill=p['none']['prefill_computed'],
                                                            per_capacity={k: p[k] for k in p if k != 'none'})
        res['validation_reference_pool'] = dict(measured=measured, simulated=sim)
    tls = {}
    nq_order, nq_mk = no_queue_order(tasks, A, args.prefill_tok_s, args.noqueue_decode_s_per_token or args.decode_s_per_token)
    tls['noqueue'] = (nq_order, dict(makespan=nq_mk))
    for R_ in args.engine_slots:
        tls[f'queued_slots{R_}'] = qsim(tasks, A, R_, args.prefill_tok_s, args.decode_s_per_token)
    preds = {}; sdist = {}
    chunk_gib = ch * beta / C.GIB
    for name, (o, st) in tls.items():
        sd = C.stack_distances(o, keys, tmap, ch)
        a = np.concatenate([sd[c] for c in o]); w = a[np.isfinite(a)] * chunk_gib
        tot = sum(tmap[c]['plen'] // ch for c in o); cold = int(np.sum(~np.isfinite(a)))
        top = int(np.nanmax(a[np.isfinite(a)])) + 2 if np.isfinite(a).any() else 2

        def cap_for(q):
            lo, hi = 1, top
            while lo < hi:
                m = (lo + hi) // 2
                if sum(C.prefix_hit_chunks(sd[c], m) for c in o) / (tot - cold) >= q: hi = m
                else: lo = m + 1
            return lo * chunk_gib
        sdist[name] = dict(timeline=st, reuse_distance_gib=dict(p50=float(np.percentile(w, 50)), p90=float(np.percentile(w, 90)),
                                                                p99=float(np.percentile(w, 99))),
                           cap_for_90pct_cov_gib=cap_for(0.9), cap_for_50pct_cov_gib=cap_for(0.5))
        for v in cgi:
            preds[f'{name}|gpu:{v}'] = predict_pool(o, sd, tmap, prevfull, sample_capacity(rng, cgi[v], len(o)), caps, beta, ch)

    def interval(key, field):
        vals = [p[key][field] for p in preds.values()]; return [float(min(vals)), float(max(vals))]
    table = {fmt_gib(g): dict(retrieved_per_rank=interval(fmt_gib(g), 'retrieved'),
                              prefill_computed_per_rank=interval(fmt_gib(g), 'prefill_computed')) for g in caps}
    table['no_host_tier'] = dict(retrieved_per_rank=[0.0, 0.0], prefill_computed_per_rank=interval('none', 'prefill_computed'))
    L = float(np.mean([c['plen'] for t in tasks for c in t['calls']]))
    res.update(working_set=dict(mean_prompt_tokens=L, bytes_per_token=beta,
                                gib={str(A): (A - 1) * L * beta / C.GIB, str(A_ref): (A_ref - 1) * L * beta / C.GIB}),
               gpu_effective_capacity_chunks={v: dict(median=float(np.median(sample_capacity(rng, cgi[v], args.capacity_samples))),
                                                      p10=float(np.percentile(sample_capacity(rng, cgi[v], args.capacity_samples), 10)),
                                                      p90=float(np.percentile(sample_capacity(rng, cgi[v], args.capacity_samples), 90)))
                                              for v in cgi},
               timelines=sdist, prediction=table, prediction_detail=preds)
    return res


def pool_md(res: dict) -> str:
    M = lambda x: f'{x / 1e6:.1f}M'
    L = [f"# Host-tier prediction for an active pool of {res['active_pool']}", '',
         f"Intervals span the timelines ({', '.join(res['timelines'])}) and the GPU-capacity distributions of the reference runs "
         f"({', '.join(res['reference_runs'])}). Working-set scale: " +
         ', '.join(f'A = {k}: {v:.2f} GiB' for k, v in res['working_set']['gib'].items()) + '.', '',
         '| host GiB/rank | restored per rank | computed prefill |', '|---:|---|---|']
    for g in ['no_host_tier'] + res['host_gib']:
        r = res['prediction'][g]
        L.append(f"| {g} | {M(r['retrieved_per_rank'][0])} - {M(r['retrieved_per_rank'][1])} | "
                 f"{M(r['prefill_computed_per_rank'][0])} - {M(r['prefill_computed_per_rank'][1])} |")
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('curve', 'pool'):
        p = sub.add_parser(name)
        p.add_argument('--trace', type=Path, required=True, help='replay trace directory')
        p.add_argument('--run', action='append', required=True, help='run directory, optionally DIR@HOST_GIB (repeatable)')
        p.add_argument('--bytes-per-token', type=int, help='KV bytes per token per rank (default: from run.json)')
        p.add_argument('--chunk', type=int, help='LMCache chunk size in tokens (default: from run.json, else 1024)')
        p.add_argument('--prefill-tok-s', type=float, help='unloaded prefill rate, tokens/s')
        p.add_argument('--decode-s-per-token', type=float, help='unloaded decode time per output token, s')
        p.add_argument('--out', type=Path, required=True, help='output JSON')
        p.add_argument('--md', type=Path, help='optional Markdown summary')
        if name == 'curve':
            p.add_argument('--grid', type=float, nargs='+', default=DEFAULT_GRID, help='host capacities, GiB per rank')
            p.add_argument('--active-pool', type=int, help='A for the working-set scale and no-queue timelines (default: first run)')
            p.add_argument('--noqueue-scales', type=float, nargs='*', help='service-time scales of no-queue timelines (0 = gaps only)')
        else:
            p.add_argument('--active-pool', type=int, required=True, help='active pool A to predict')
            p.add_argument('--engine-slots', type=int, nargs='+', required=True, help='engine slots R of the queue model')
            p.add_argument('--host-gib', type=float, nargs='+', required=True, help='host capacities, GiB per rank')
            p.add_argument('--noqueue-decode-s-per-token', type=float, help='decode time of the no-queue timeline (default: --decode-s-per-token)')
            p.add_argument('--seed', type=int, default=0)
            p.add_argument('--capacity-samples', type=int, default=20000)
            p.add_argument('--validate', action='store_true', help='also simulate the reference pool and report measured values')
    a = ap.parse_args(argv)
    if a.cmd == 'pool' and (a.prefill_tok_s is None or a.decode_s_per_token is None):
        ap.error('pool needs --prefill-tok-s and --decode-s-per-token')
    res = curve(a) if a.cmd == 'curve' else pool(a)
    write_json(a.out, res)
    if a.md:
        Path(a.md).write_text(curve_md(res) if a.cmd == 'curve' else pool_md(res))
    return res


if __name__ == '__main__':
    main()
