"""Offline evaluation of the capacity-conditioned pressure signal on a recorded run (CPU only).

The pressure signal of capacity-conditioned admission is::

    p_t = [C_reuse > C_H] and [o_t >= theta and e_t > 0],     C_reuse = (A_t - 1) * N_t * beta

For any run, the working-set estimate is evaluated at each request's client dispatch time from replay/requests.jsonl:
A_t counts the tasks with a request in the last ``window`` seconds (the current request included) and N_t is the mean
prompt length of the last ``prompt_window`` requests. The command reports the share of requests (and of a 10-s time
grid) with C_reuse above each given host capacity, C_reuse quantiles, and the first minute the estimate exceeds each
capacity. When the run has host-tier observer samples (lmcache_observer/cpu-<pid>.jsonl, cumulative capacity evictions
and resident chunks every 10 s), it also evaluates the full-and-evicting condition at the run's own capacity (the 60-s
eviction window on sample boundaries) and its conjunction with the estimate, weighted by requests and by time.

Example::

    python -m efficientagent.analysis.pressure_estimate --run runs/offload_5 --run runs/offload_40 --host-gib 5 10 40 --out pressure.json
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

from efficientagent.analysis import common as C
from efficientagent.analysis.runsets import RunInfo, fmt_gib, resolve_run, write_json


def estimate_series(reqs: list[dict], beta: int, window: float, prompt_window: int) -> list[tuple]:
    """-> [(t, C_reuse bytes, A_t, N_t)] at each request dispatch, sorted by dispatch time."""
    rs = sorted((r for r in reqs if 'start' in r), key=lambda r: r['start'])
    out = []; last = {}; lens = []; lsum = 0
    for r in rs:
        t = r['start']; last[r['instance_id']] = t
        lens.append(r['prompt_len']); lsum += r['prompt_len']
        if len(lens) > prompt_window: lsum -= lens[-prompt_window - 1]
        n = sum(1 for v in last.values() if v >= t - window); nbar = lsum / min(len(lens), prompt_window)
        out.append((t, (n - 1) * nbar * beta, n, nbar))
    return out


def observer_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if '"keys"' in line]


def full_evicting_series(rows: list[dict], cap: int, theta: float, window: float) -> list[tuple]:
    """-> [(t, occupancy >= theta and evictions in the last window, resident keys)] per observer sample."""
    ts = [r['t'] for r in rows]; out = []
    for r in rows:
        j = bisect.bisect_right(ts, r['t'] - window) - 1
        base = rows[j]['evicted'] if j >= 0 else 0
        out.append((r['t'], r['keys'] >= theta * cap and (r['evicted'] - base) > 0, r['keys']))
    return out


def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def rank_summaries(run: RunInfo, files: list[Path], reqs: list[dict], t0: float | None, t1: float | None, theta: float,
                   window: float) -> list[dict]:
    """Per-rank observer summary over the replay window (the observer's own span when the client summary has none)."""
    per = []
    for f in files:
        rows = observer_rows(f)
        if not rows: continue
        if t0 is None: t0 = rows[0]['t']
        if t1 is None: t1 = rows[-1]['t']
        gib = rows[-1]['capacity_gb']; cap = C.chunks_of(gib, run.bytes_per_token, run.chunk)
        s = [x for x in full_evicting_series(rows, cap, theta, window) if t0 <= x[0] <= t1]
        if not s: continue
        ts = [t for t, _, _ in s]

        def at(t):
            i = bisect.bisect_right(ts, t) - 1
            return s[i][1] if i >= 0 else False
        rq = [at(r['start']) for r in reqs if 'start' in r]
        m = lambda t: None if t is None else round((t - t0) / 60.0, 1)
        k0 = next((i for i, x in enumerate(s) if x[1]), None)
        per.append(dict(file=f.name, gib=gib, capacity_chunks=cap, samples=len(s), max_keys=max(k for _, _, k in s),
                        first_ge_theta_min=m(next((t for t, _, k in s if k >= theta * cap), None)),
                        first_full_min=m(next((t for t, _, k in s if k >= cap), None)),
                        first_full_evicting_min=m(next((t for t, v, _ in s if v), None)),
                        full_evicting_share_of_time=sum(v for _, v, _ in s) / len(s),
                        full_evicting_share_of_requests=(sum(rq) / len(rq)) if rq else None,
                        full_evicting_share_of_time_after_first=(sum(v for _, v, _ in s[k0:]) / len(s[k0:])) if k0 is not None else None,
                        evicted_total=rows[-1]['evicted']))
    return per


def evaluate(run: RunInfo, capacities: list[float], theta: float, window: float, prompt_window: int, grid_s: float) -> dict:
    reqs = C.read_requests(run.dir); summ = C.read_summary(run.dir)
    ser = estimate_series(reqs, run.bytes_per_token, window, prompt_window); ts = [x[0] for x in ser]
    t0 = summ.get('start') or ts[0]; t1 = summ.get('end') or max(r.get('end', r['start']) for r in reqs)
    ids = [r['instance_id'] for r in sorted((r for r in reqs if 'start' in r), key=lambda r: r['start'])]
    grid = [t0 + grid_s * k for k in range(int((t1 - t0) // grid_s) + 1)]

    def estimate_at(t):
        i = bisect.bisect_right(ts, t) - 1
        if i < 0: return 0.0
        n = len(set(ids[bisect.bisect_right(ts, t - window):i + 1])); return (n - 1) * ser[i][3] * run.bytes_per_token if n else 0.0
    cg = [estimate_at(t) for t in grid]
    m = lambda t: None if t is None else round((t - t0) / 60.0, 1)
    G = C.GIB
    res = dict(own_gib=run.host_gib, admission=run.admission, requests=len(ser), makespan_min=round((t1 - t0) / 60.0, 1),
               estimate_gib=dict(p10=q([x[1] / G for x in ser], .1), median=q([x[1] / G for x in ser], .5), p90=q([x[1] / G for x in ser], .9),
                                 max=max(x[1] for x in ser) / G, mean=sum(x[1] for x in ser) / len(ser) / G),
               active_tasks_median=q([x[2] for x in ser], .5), mean_prompt_final=ser[-1][3], estimate_above={})
    for g in capacities:
        cap = g * G; rq = [x[1] > cap for x in ser]
        res['estimate_above'][fmt_gib(g)] = dict(share_of_requests=sum(rq) / len(rq), share_of_time=sum(c > cap for c in cg) / len(cg),
                                                 first_true_min=m(next((x[0] for x in ser if x[1] > cap), None)))
    od = run.dir / 'lmcache_observer'
    files = sorted(od.glob('cpu-*.jsonl'), key=lambda p: int(p.stem.split('-')[1])) if od.exists() else []
    rows = observer_rows(files[0]) if files else []
    if rows:
        cap_chunks = C.chunks_of(rows[-1]['capacity_gb'], run.bytes_per_token, run.chunk)
        s = [x for x in full_evicting_series(rows, cap_chunks, theta, window) if t0 <= x[0] <= t1]
        st = [x[0] for x in s]; cap = rows[-1]['capacity_gb'] * G

        def fe_at(t):
            i = bisect.bisect_right(st, t) - 1; return s[i][1] if i >= 0 else False
        rq1 = [fe_at(x[0]) for x in ser]; rq2 = [x[1] > cap for x in ser]
        tm1 = [v for _, v, _ in s]; tm2 = [estimate_at(t) > cap for t, _, _ in s]
        res['telemetry'] = dict(capacity_gb=rows[-1]['capacity_gb'], capacity_chunks=cap_chunks, samples=len(s),
                                full_evicting_share_of_requests=sum(rq1) / len(rq1), full_evicting_share_of_time=sum(tm1) / len(tm1),
                                pressure_share_of_requests=sum(a and b for a, b in zip(rq1, rq2)) / len(rq1),
                                pressure_share_of_time=sum(a and b for a, b in zip(tm1, tm2)) / len(tm1),
                                estimate_above_own_share_of_requests=sum(rq2) / len(rq2),
                                first_pressure_min=m(next((x[0] for x, a, b in zip(ser, rq1, rq2) if a and b), None)))
        res['telemetry_ranks'] = rank_summaries(run, files, reqs, summ.get('start'), summ.get('end'), theta, window)
    return res


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True, help='run directory, optionally DIR@GIB (repeatable)')
    ap.add_argument('--host-gib', type=float, nargs='*', help='capacities to test the estimate against (default: each run\'s own)')
    ap.add_argument('--theta', type=float, default=0.95, help='occupancy threshold')
    ap.add_argument('--window-s', type=float, default=60.0, help='task-activity and eviction window')
    ap.add_argument('--prompt-window', type=int, default=256, help='requests averaged for the mean prompt length')
    ap.add_argument('--bytes-per-token', type=int, help='KV bytes per token per rank (default: from run.json)')
    ap.add_argument('--grid-s', type=float, default=10.0, help='time grid for time-weighted shares')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    out = dict(schema='efficientagent.pressure_estimate', version=1, theta=a.theta, window_s=a.window_s, prompt_window=a.prompt_window, runs={})
    for arg in a.run:
        run = resolve_run(arg, a.bytes_per_token)
        caps = a.host_gib or ([run.host_gib] if run.host_gib > 0 else [])
        out['runs'][run.name] = evaluate(run, caps, a.theta, a.window_s, a.prompt_window, a.grid_s)
    write_json(a.out, out)
    return out


if __name__ == '__main__':
    main()
