"""Reuse distance of the chunks that write admission declined (CPU only).

Proposition 1 of the paper: in an LRU tier of K_H chunks, declining the insertion of chunks whose next reuse distance
D satisfies D >= K_H (or that are never referenced again) keeps every hit of full admission. This command measures the
condition on the writes the admission runs declined.

Reference stream (as in ``capacity_model``): each request references all full chunks of its prompt at its scheduling
lookup, tail first, with prefix-dependent keys. For a chunk referenced at position s, D is the number of distinct other
chunks referenced strictly before its next reference (the stack distance of that next reference); 'never' if there is
none. K_H = floor(C_H / (chunk tokens x bytes per token)).

Declines. The connector decides once per request at its first resolved lookup from n (request tokens) and h (host
hit): u = floor(n / b) - floor(h / b), declined iff p_t and u > kappa. The declined chunks of a request are its new full
chunks h//b .. n//b - 1.
  fixed admission (p_t = 1): the decisions follow exactly from the logged first lookups and are checked against the
    connector counters.
  conditioned admission: p_t is recomputed per request: the working-set estimate (A_t - 1) * N_t * beta > C_H with A_t
    the tasks with a first lookup in the last ``window`` seconds and N_t the mean n of the last ``prompt_window`` first
    lookups, and the full-and-evicting condition from the telemetry snapshot preceding the lookup
    (telemetry_history.jsonl). Bounds use the snapshots before and after the lookup; the counters are reported beside.
Offload runs (no admission) report the D distribution of every newly written chunk for comparison.

Checks: the stack-distance host hit of each request against the logged hit, and an LRU replay of each run's stream with
its declines against the logged hits; the LRU replay also gives the host hits under full admission and under only the
declines that meet the condition.

Example::

    python -m efficientagent.analysis.declined_reuse --trace TRACE --run runs/fixed_5 --run runs/conditioned_5 \\
        --run runs/offload_5 --out declined.json --md declined.md
"""
from __future__ import annotations

import argparse
import bisect
import collections
import itertools
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from efficientagent.analysis import common as C
from efficientagent.analysis.runsets import RunInfo, fmt_gib, resolve_run, write_json

ORDERS = ('sched_lookup', 'first_lookup', 'client_start')
MAX_AMBIGUOUS = 16   # enumerate counter-consistent choices only up to this many ambiguous requests


# ------------------------------------------------------------------ log parsing and mapping
def parse(path: str):
    """-> (path, lookups (t, rid, n, h, need), storing (t, rid))."""
    look, storing = [], []
    with open(Path(path) / 'server.log', errors='replace') as f:
        for line in f:
            if 'LMCache INFO' not in line: continue
            if 'LMCache hit tokens' in line:
                for m in C.R_LOOK.finditer(line): look.append((C.ts(m[1]), m[2], int(m[3]), int(m[4]), int(m[5])))
            elif 'Storing KV cache' in line:
                for m in C.R_STORING.finditer(line): storing.append((C.ts(m[1]), m[5]))
    return path, look, storing


def map_decisions(run: RunInfo, look: list, storing: list):
    """-> ({call: dict(rid, t_first, n, h, u, t_sched, hit_sched, start)}, unmapped request IDs, decision stream)."""
    req = C.read_requests(run.dir); req.sort(key=lambda r: r['start'])
    first = {}
    for t_, rid, tot, hit, need in look: first.setdefault(rid, (t_, tot, hit))
    by_len = collections.defaultdict(list)
    for rid, (t_, tot, _) in first.items(): by_len[tot].append((t_, rid))
    used, rid2req = set(), {}
    for r in req:
        c = [(t_ - r['start'], rid) for t_, rid in by_len.get(r['prompt_len'], []) if rid not in used and r['start'] - 0.5 <= t_ <= r['end']]
        if c:
            rid = min(c)[1]; used.add(rid); rid2req[rid] = r
    fs = {}
    for t_, rid in storing: fs.setdefault(rid, t_)
    lb = collections.defaultdict(list)
    for x in look: lb[x[1]].append(x)
    ch = run.chunk; out = {}
    for rid, r in rid2req.items():
        t1, n, h = first[rid]
        L = [x for x in lb[rid] if x[2] == n and (rid not in fs or x[0] <= fs[rid] + 0.05)]
        out[(r['instance_id'], r['seq'])] = dict(rid=rid, t_first=t1, n=n, h=h, u=n // ch - h // ch, t_sched=(L or lb[rid])[-1][0],
                                                   hit_sched=(L or lb[rid])[-1][3], start=r['start'])
    stream = sorted(first.values())
    return out, len(first) - len(rid2req), stream


def scheduler_counters(run: RunInfo) -> dict:
    return {k.split('.', 1)[1]: v for k, v in C.run_counters(run.dir).items() if k.startswith('scheduler.')}


# ------------------------------------------------------------------ pressure reconstruction (conditioned runs)
def reconstruct_pressure(run: RunInfo, M: dict, window: float, prompt_window: int, theta: float) -> dict:
    """Per call: the estimate condition (computable for every run) and, where telemetry snapshots exist, the
    full-and-evicting condition from the snapshot before (full_ev) and after (full_ev_next) the first lookup.
    skip = estimate and full_ev and u > kappa; skip_lo / skip_hi use both / either snapshot."""
    hist = C.read_jsonl(run.dir / 'telemetry_history.jsonl')
    ht = [h['t'] for h in hist]
    full_ev_of = lambda h: bool(h['evicted_in_window'] > 0 and h['registered'] >= theta * h['capacity'])
    host = run.host_gib * C.GIB; beta = run.bytes_per_token; kappa = run.write_threshold
    tasks, lens, lsum = {}, collections.deque(), 0
    res = {}
    for call, m in sorted(M.items(), key=lambda kv: kv[1]['t_first']):
        t = m['t_first']; tasks[call[0]] = t
        lens.append(m['n']); lsum += m['n']
        if len(lens) > prompt_window: lsum -= lens.popleft()
        A = sum(1 for v in tasks.values() if v >= t - window); nbar = lsum / len(lens)
        above = (A - 1) * nbar * beta > host
        r = dict(above=above, A=A, estimate_gib=(A - 1) * nbar * beta / C.GIB)
        if hist:
            i = bisect.bisect_right(ht, t) - 1
            fe = full_ev_of(hist[i]) if i >= 0 else False
            fen = full_ev_of(hist[i + 1]) if i + 1 < len(hist) else fe
            big = m['u'] > kappa
            r.update(full_ev=fe, full_ev_next=fen, full_ev_or=fe or fen, full_ev_and=fe and fen, p=above and fe, skip=above and fe and big,
                     skip_lo=above and fe and fen and big, skip_hi=above and (fe or fen) and big)
        res[call] = r
    return res


def active_tasks(run: RunInfo):
    """-> f(t) = tasks between their first request start and last request end, last task start, run start, run end."""
    span = {}
    for r in C.read_requests(run.dir):
        a = span.setdefault(r['instance_id'], [float('inf'), 0.0])
        a[0] = min(a[0], r['start']); a[1] = max(a[1], r['end'])
    ev = sorted([(a, 1) for a, _ in span.values()] + [(b, -1) for _, b in span.values()])
    ts = [e[0] for e in ev]; cum = list(np.cumsum([e[1] for e in ev]))
    f = lambda t: int(cum[bisect.bisect_right(ts, t) - 1]) if bisect.bisect_right(ts, t) else 0
    return f, max(a for a, _ in span.values()), min(a for a, _ in span.values()), max(b for _, b in span.values())


# ------------------------------------------------------------------ reference stream and next-reference distance
def next_distance(order: list, keys: dict, tmap: dict, ch: int):
    """References in ``order``, each call's full prompt chunks tail first. Returns D[call] (per chunk 0..m-1, distinct
    other chunks referenced before the chunk's next reference, inf = never), nxt[call] (call holding that next
    reference), sd[call] (stack distance of the reference itself, inf = first reference), total references."""
    total = sum(tmap[c]['plen'] // ch for c in order)
    bit = C.BIT(total + 1); last = {}; pos = 0
    ref_call = [None] * total
    Dnext = np.full(total, np.inf); nxt_pos = np.full(total, -1, dtype=np.int64)
    sd = {}
    for call in order:
        k = keys[call][:tmap[call]['plen'] // ch]; s = np.full(len(k), np.inf)
        for c in reversed(range(len(k))):
            key = k[c]
            if key in last:
                p = last[key]; d = bit.pre(pos) - bit.pre(p + 1); bit.add(p, -1)
                s[c] = d; Dnext[p] = d; nxt_pos[p] = pos
            bit.add(pos, 1); last[key] = pos; ref_call[pos] = call; pos += 1
        sd[call] = s
    D, nxt = {}, {}
    pos = 0
    for call in order:
        m = tmap[call]['plen'] // ch
        idx = np.arange(m)[::-1]          # positions pos .. pos+m-1 hold chunks m-1 .. 0
        arr = np.full(m, np.inf); arr[idx] = Dnext[pos:pos + m]; D[call] = arr
        nc = [None] * m
        for j in range(m):
            q = nxt_pos[pos + j]
            if q >= 0: nc[idx[j]] = ref_call[q]
        nxt[call] = nc; pos += m
    return D, nxt, sd, total


def classify(chunks: list, D: dict, nxt: dict, K: int) -> dict:
    """chunks: [(call, idx)] -> counts, shares, per-request shares and next-reference location."""
    c = collections.Counter(); per_req = collections.defaultdict(lambda: [0, 0])
    dvals = []; loc = collections.Counter()
    for call, i in chunks:
        d = D[call][i]; per_req[call][1] += 1
        if not np.isfinite(d): c['never'] += 1; per_req[call][0] += 1; continue
        dvals.append(d)
        same_task = nxt[call][i][0] == call[0]
        if d >= K: c['ge_K'] += 1; per_req[call][0] += 1
        else:
            c['lt_K'] += 1
            loc['same_task' if same_task else 'other_task'] += 1
    n = len(chunks)
    shares = [a / b for a, b in per_req.values() if b]
    dv = np.asarray(dvals)
    return dict(chunks=n, never=c['never'], ge_K=c['ge_K'], lt_K=c['lt_K'],
                share_safe=(c['never'] + c['ge_K']) / n if n else None,
                share_never=c['never'] / n if n else None, share_ge_K=c['ge_K'] / n if n else None,
                share_lt_K=c['lt_K'] / n if n else None,
                requests=len(per_req),
                req_mean_share_safe=float(np.mean(shares)) if shares else None,
                req_all_safe=sum(1 for a, b in per_req.values() if a == b) / len(per_req) if per_req else None,
                req_majority_safe=sum(1 for a, b in per_req.values() if 2 * a >= b) / len(per_req) if per_req else None,
                req_all_lt_K=sum(1 for a, b in per_req.values() if a == 0) / len(per_req) if per_req else None,
                lt_K_next_ref_same_task=loc['same_task'], lt_K_next_ref_other_task=loc['other_task'],
                D_over_K_quantiles_reused=({str(q): float(np.percentile(dv, q) / K) for q in (5, 10, 25, 50, 75, 90)} if len(dv) else None))


def lru_replay(order: list, keys: dict, tmap: dict, K: int, declined: dict, ch: int):
    """LRU tier of K chunks on the stream. At each call the host hit is the leading resident run (before the call's
    references); then chunks are referenced tail first, resident ones promoted, missing ones inserted unless declined.
    Returns {call: predicted hit tokens} and the number of declined chunks that were already resident."""
    od = collections.OrderedDict(); pred = {}; resident_at_decline = 0
    for call in order:
        k = keys[call][:tmap[call]['plen'] // ch]
        h = 0
        while h < len(k) and k[h] in od: h += 1
        pred[call] = h * ch
        dec = declined.get(call)
        for c in reversed(range(len(k))):
            key = k[c]
            if key in od:
                od.move_to_end(key)
                if dec is not None and dec[0] <= c < dec[1]: resident_at_decline += 1
            elif dec is None or not (dec[0] <= c < dec[1]):
                od[key] = True
                if len(od) > K: od.popitem(last=False)
    return pred, resident_at_decline


def orders_of(M: dict) -> dict:
    return {'sched_lookup': [c for c, _ in sorted(M.items(), key=lambda kv: (kv[1]['t_sched'], kv[1]['t_first']))],
            'first_lookup': [c for c, _ in sorted(M.items(), key=lambda kv: kv[1]['t_first'])],
            'client_start': [c for c, _ in sorted(M.items(), key=lambda kv: kv[1]['start'])]}


# ------------------------------------------------------------------ per run
def analyze_run(run: RunInfo, look: list, storing: list, tasks_keys, compare_gib: list, window: float, prompt_window: int, theta: float) -> dict:
    keys, tmap = tasks_keys
    ch = run.chunk; kappa = run.write_threshold; K = run.capacity_chunks
    Kof = lambda g: C.chunks_of(g, run.bytes_per_token, ch)
    M, unmapped, stream = map_decisions(run, look, storing)
    cnt = scheduler_counters(run)
    u_all = [n // ch - h // ch for _, n, h in stream]
    rec = dict(admission=run.admission, host_gib=run.host_gib, K_H=K, write_threshold=kappa, decisions_logged=len(stream), mapped=len(M),
               unmapped=unmapped, counters=cnt or None, new_chunks_all=sum(max(0, x) for x in u_all),
               new_chunks_mapped=sum(max(0, m['u']) for m in M.values()),
               u_gt_kappa_requests=sum(x > kappa for x in u_all), u_gt_kappa_chunks=sum(x for x in u_all if x > kappa))
    P = reconstruct_pressure(run, M, window, prompt_window, theta)
    act, t_last_task, t_run0, t_run1 = active_tasks(run)
    rec['drain_phase_start_min'] = (t_last_task - t_run0) / 60; rec['run_span_min'] = (t_run1 - t_run0) / 60
    skip_bounds = None
    if run.admission == 'fixed':
        skip = {c for c, m in M.items() if m['u'] > kappa}
        rec['decision_source'] = 'exact: p = 1, skip iff u > kappa at the first logged lookup'
        if cnt:
            rec['counter_check'] = dict(
                skip_requests=(len(skip), cnt.get('skipped_requests')),
                skipped_chunks=(sum(M[c]['u'] for c in skip), cnt.get('skipped_new_chunks')),
                saved_new_chunks=(sum(max(0, M[c]['u']) for c in M if c not in skip), cnt.get('saved_new_chunks')),
                decisions=(len(stream), cnt.get('decisions')))
    elif run.admission == 'conditioned':
        skip = {c for c, v in P.items() if v.get('skip')}
        skip_bounds = ({c for c, v in P.items() if v.get('skip_lo')}, {c for c, v in P.items() if v.get('skip_hi')})
        rec['decision_source'] = ('reconstructed: estimate from the logged first lookups, full-and-evicting from the telemetry '
                                  'snapshot preceding each lookup')
        if cnt:
            rec['counter_check'] = dict(
                full_evicting=(sum(v.get('full_ev', False) for v in P.values()), cnt.get('telemetry_full_evicting')),
                estimate_above_capacity=(sum(v['above'] for v in P.values()), cnt.get('estimate_above_capacity')),
                pressure=(sum(v.get('p', False) for v in P.values()), cnt.get('pressure_requests')),
                skip_requests=(len(skip), cnt.get('skipped_requests')),
                decisions=(len(stream), cnt.get('decisions')))
            rec['full_evicting_variants_vs_counter'] = dict(
                counter=cnt.get('telemetry_full_evicting'), before=sum(v.get('full_ev', False) for v in P.values()),
                after=sum(v.get('full_ev_next', False) for v in P.values()),
                either=sum(v.get('full_ev_or', False) for v in P.values()), both=sum(v.get('full_ev_and', False) for v in P.values()))
            rec['skip_bounds'] = dict(lo=len(skip_bounds[0]), primary=len(skip), hi=len(skip_bounds[1]), counter=cnt.get('skipped_requests'),
                                      counter_within_bounds=len(skip_bounds[0]) <= (cnt.get('skipped_requests') or 0) <= len(skip_bounds[1]))
    else:
        skip = set()
    if run.admission in ('fixed', 'conditioned'):
        if 'counter_check' in rec:
            rec['counter_check_exact'] = all(a == b for a, b in rec['counter_check'].values())
        rec['declined_requests'] = len(skip); rec['declined_chunks'] = sum(M[c]['u'] for c in skip)
    dch = lambda calls: [(c, i) for c in calls for i in range(M[c]['h'] // ch, M[c]['n'] // ch)]
    A = run.workers; upper = math.ceil(0.75 * A)
    rec['by_order'] = {}
    for oname, order in orders_of(M).items():
        D, nxt, sd, tot = next_distance(order, keys, tmap, ch)
        r = dict(stream_chunk_refs=tot)
        if skip:
            dc = dch(skip)
            r['declined'] = classify(dc, D, nxt, K)
            r['declined_vs_other_K'] = {fmt_gib(g): classify(dc, D, nxt, Kof(g))['share_safe'] for g in compare_gib}
            if skip_bounds is not None and cnt:
                lo, hi = skip_bounds; extra = sorted(hi - lo); k = (cnt.get('skipped_requests') or 0) - len(lo)
                cl, chh = classify(dch(lo), D, nxt, K), classify(dch(hi), D, nxt, K)
                r['declined_bounds'] = dict(lo_set=cl['share_safe'], hi_set=chh['share_safe'],
                                            lo_set_req=cl['req_mean_share_safe'], hi_set_req=chh['req_mean_share_safe'])
                if 0 <= k <= len(extra) <= MAX_AMBIGUOUS:   # every choice of ambiguous requests that matches the skip counter
                    vals = [classify(dch(lo | set(sub)), D, nxt, K) for sub in itertools.combinations(extra, k)]
                    r['declined_bounds'].update(counter_consistent_min=min(v['share_safe'] for v in vals),
                                                counter_consistent_max=max(v['share_safe'] for v in vals),
                                                counter_consistent_req_min=min(v['req_mean_share_safe'] for v in vals),
                                                counter_consistent_req_max=max(v['req_mean_share_safe'] for v in vals),
                                                counter_consistent_chunks=sorted({v['chunks'] for v in vals}),
                                                ambiguous_requests=len(extra), pick=k)
            if oname == 'sched_lookup':
                groups = {'active_full': lambda c: act(M[c]['t_first']) >= A,
                          'active_upper_quarter': lambda c: upper <= act(M[c]['t_first']) < A,
                          'active_below': lambda c: act(M[c]['t_first']) < upper,
                          'estimate_above': lambda c: P[c]['above'], 'estimate_below': lambda c: not P[c]['above'],
                          'backlogged': lambda c: M[c]['t_first'] < t_last_task, 'drain': lambda c: M[c]['t_first'] >= t_last_task}
                r['declined_by_group'] = {}
                for gname, f in groups.items():
                    sub = [c for c in skip if f(c)]
                    r['declined_by_group'][gname] = classify(dch(sub), D, nxt, K) if sub else dict(chunks=0, requests=0)
        newch = dch(M)
        r['new_chunks_all'] = classify(newch, D, nxt, K)
        r['new_chunks_u_gt_kappa'] = classify([(c, i) for c, i in newch if M[c]['u'] > kappa], D, nxt, K)
        r['new_chunks_u_le_kappa'] = classify([(c, i) for c, i in newch if M[c]['u'] <= kappa], D, nxt, K)
        if oname == 'sched_lookup':
            pm = {c: C.prefix_hit_chunks(sd[c], K) * ch for c in order}
            r['sd_prefix_hit_exact_frac'] = float(np.mean([pm[c] == M[c]['hit_sched'] for c in order]))
            dec = {c: (M[c]['h'] // ch, M[c]['n'] // ch) for c in skip}
            pred, moot = lru_replay(order, keys, tmap, K, dec, ch)
            r['lru_replay'] = dict(exact_hit_frac=float(np.mean([pred[c] == M[c]['hit_sched'] for c in order])),
                                   pred_hit_tokens=int(sum(pred.values())), logged_hit_tokens=int(sum(M[c]['hit_sched'] for c in order)),
                                   declined_chunks_resident_at_decline=moot)
            if skip:
                full, _ = lru_replay(order, keys, tmap, K, {}, ch)
                safe_calls = {c for c in skip if all((not np.isfinite(D[c][i])) or D[c][i] >= K for i in range(dec[c][0], dec[c][1]))}
                comp, _ = lru_replay(order, keys, tmap, K, {c: dec[c] for c in safe_calls}, ch)
                r['lru_counterfactual_host_hit_tokens'] = dict(full_admission=int(sum(full.values())), realized_declines=int(sum(pred.values())),
                                                               condition_meeting_declines_only=int(sum(comp.values())),
                                                               condition_meeting_requests=len(safe_calls))
        rec['by_order'][oname] = r
    return rec


def run_all(trace: Path, runs: list[RunInfo], compare_gib: list, window: float, prompt_window: int, theta: float, workers: int | None = None) -> dict:
    tasks = C.load_trace(trace); ch = runs[0].chunk
    keys = C.prompt_chunk_keys(tasks, ch); tmap = C.call_map(tasks)
    rows = []
    for t in tasks:
        cs = t['calls']
        for i, c in enumerate(cs):
            new = c['plen'] - c['shared'] + c['olen']
            nxt = C.lcp(np.concatenate([c['prompt'], c['out']]), cs[i + 1]['prompt']) if i + 1 < len(cs) else 0
            rows.append((new, max(0, min(nxt, c['plen']) - c['shared']) + (max(0, nxt - c['plen']) if nxt > c['plen'] else 0)))
    new_tok = sum(a for a, _ in rows); next_tok = sum(b for _, b in rows)
    with ProcessPoolExecutor(workers or len(runs)) as ex:
        parsed = {Path(p): (l, s) for p, l, s in ex.map(parse, [str(r.dir) for r in runs])}
    out = dict(schema='efficientagent.declined_reuse', version=1, chunk_tokens=ch, bytes_per_token=runs[0].bytes_per_token,
               chunk_bytes_per_rank=ch * runs[0].bytes_per_token, theta=theta, window_s=window, prompt_window=prompt_window,
               K_H={fmt_gib(g): C.chunks_of(g, runs[0].bytes_per_token, ch) for g in sorted(set(compare_gib) | {r.host_gib for r in runs})},
               trace=dict(calls=len(tmap), full_chunk_refs=sum(c['plen'] // ch for c in tmap.values()), new_tokens=new_tok,
                          reused_next_tokens=next_tok, reused_next_share=next_tok / new_tok if new_tok else None),
               admission={}, offload={})
    for run in runs:
        look, storing = parsed[Path(str(run.dir))]
        rec = analyze_run(run, look, storing, (keys, tmap), compare_gib, window, prompt_window, theta)
        (out['admission'] if run.admission in ('fixed', 'conditioned') else out['offload'])[run.name] = rec
    return out


# ------------------------------------------------------------------ Markdown summary
def pct(x): return '-' if x is None else f'{100 * x:.1f}%'


def write_md(o: dict) -> str:
    A, O = o['admission'], o['offload']
    Mt = lambda x: f'{x / 1e6:.1f}M'
    L = ['# Reuse distance of declined chunks', '',
         'D = distinct other chunks referenced between a declined chunk\'s reference and its next reference; the condition of '
         'Proposition 1 is D >= K_H or no later reference ("meets"). K_H per capacity: ' +
         ', '.join(f'{g} GiB: {k}' for g, k in o['K_H'].items()) + '.', '',
         '| run | admission | GiB | declined requests | declined chunks | D >= K_H | never | meets | D < K_H | meets, by request |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in A.items():
        d = r['by_order']['sched_lookup'].get('declined')
        if not d:
            L.append(f"| {name} | {r['admission']} | {fmt_gib(r['host_gib'])} | 0 | 0 | - | - | - | - | - |"); continue
        L.append(f"| {name} | {r['admission']} | {fmt_gib(r['host_gib'])} | {r['declined_requests']:,} | {r['declined_chunks']:,} | "
                 f"{pct(d['share_ge_K'])} | {pct(d['share_never'])} | **{pct(d['share_safe'])}** | {pct(d['share_lt_K'])} | {pct(d['req_mean_share_safe'])} |")
    L += ['', 'D/K_H of re-referenced declined chunks (p10 / p25 / p50 / p90):', '']
    for name, r in A.items():
        d = r['by_order']['sched_lookup'].get('declined')
        if d and d['D_over_K_quantiles_reused']:
            q = d['D_over_K_quantiles_reused']
            L.append(f"- {name}: {q['10']:.2f} / {q['25']:.2f} / {q['50']:.2f} / {q['90']:.2f}")
    L += ['', '## Decisions', '']
    for name, r in A.items():
        cc = r.get('counter_check', {})
        L.append(f"- {name}: {r['decision_source']}; mapped {r['mapped']:,}/{r['decisions_logged']:,} requests"
                 + ('; ' + '; '.join(f'{k} {a} vs counter {b}' for k, (a, b) in cc.items()) if cc else '') + '.')
    L += ['', '## Newly written chunks, all runs', '',
          '| run | GiB | K_H | new chunks | meets | never | D < K_H |', '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in list(O.items()) + list(A.items()):
        n = r['by_order']['sched_lookup']['new_chunks_all']
        L.append(f"| {name} | {fmt_gib(r['host_gib'])} | {r['K_H']} | {n['chunks']:,} | {pct(n['share_safe'])} | {pct(n['share_never'])} | {pct(n['share_lt_K'])} |")
    L += ['', '## Stream checks', '']
    for name, r in list(O.items()) + list(A.items()):
        s = r['by_order']['sched_lookup']; lr = s['lru_replay']
        L.append(f"- {name}: per-request host hit reproduced by the stack-distance model for {pct(s['sd_prefix_hit_exact_frac'])} and by the LRU "
                 f"replay with the run's declines for {pct(lr['exact_hit_frac'])} of requests (hit tokens {Mt(lr['pred_hit_tokens'])} vs "
                 f"{Mt(lr['logged_hit_tokens'])} logged).")
    return '\n'.join(L) + '\n'


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trace', type=Path, required=True)
    ap.add_argument('--run', action='append', required=True, help='run directory, optionally DIR@GIB (repeatable)')
    ap.add_argument('--compare-gib', type=float, nargs='*', help='also classify declines against these capacities (default: the runs\' own)')
    ap.add_argument('--theta', type=float, default=0.95, help='occupancy threshold')
    ap.add_argument('--window-s', type=float, default=60.0, help='task-activity window')
    ap.add_argument('--prompt-window', type=int, default=256, help='first lookups averaged for the mean prompt length')
    ap.add_argument('--bytes-per-token', type=int, help='KV bytes per token per rank (default: from run.json)')
    ap.add_argument('--processes', type=int, help='log-parsing processes (default: one per run)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--md', type=Path)
    a = ap.parse_args(argv)
    runs = [resolve_run(r, a.bytes_per_token) for r in a.run]
    runs = [r for r in runs if r.host_gib > 0]
    if not runs:
        raise SystemExit('no run with a host tier')
    compare = sorted(set(a.compare_gib)) if a.compare_gib else sorted({r.host_gib for r in runs})
    out = run_all(a.trace, runs, compare, a.window_s, a.prompt_window, a.theta, a.processes)
    write_json(a.out, out)
    if a.md:
        Path(a.md).write_text(write_md(out))
    return out


if __name__ == '__main__':
    main()
