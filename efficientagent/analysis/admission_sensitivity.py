"""Sensitivity of the write-admission rule to its parameters kappa and theta, from recorded runs (CPU only).

kappa: the connector decides at a request's first resolved LMCache lookup from n (request tokens) and h (host hit
tokens); LMCache logs both in the same call ('Reqid: <rid>, Total tokens <n>, LMCache hit tokens: <h>, ...'), so the
first such line of each request gives u = floor(n / b) - floor(h / b) exactly. For every kappa the command reports the
requests with u > kappa and the share of new-chunk writes they carry. Under fixed write admission every request is under
pressure, so the stream at the run's own kappa must reproduce the connector counters (decisions, skipped requests,
skipped and saved new chunks); a mismatch raises ValueError. For conditioned runs, the requests under pressure are a
subset of those with u > kappa.

theta: conditioned runs record the rank-0 telemetry every 5 s (telemetry_history.jsonl). For the reports with
eviction activity (e_t > 0) the command lists the occupancy o_t = registered / capacity and the share with o_t >= theta
on a grid, which shows the range of theta that yields the same pressure signal.

Example::

    python -m efficientagent.analysis.admission_sensitivity --run runs/fixed_5 --run runs/conditioned_5 --out sensitivity.json
"""
from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path

from efficientagent.analysis import common as C
from efficientagent.analysis.runsets import RunInfo, resolve_run, write_json

PAT = re.compile(r'Reqid: (\S+), Total tokens (\d+), LMCache hit tokens: (\d+), need to load: (-?\d+)')


def decision_stream(run: RunInfo) -> list[int]:
    """u at each request's first logged lookup, in log order."""
    first = {}
    with open(run.dir / 'server.log', errors='replace') as f:
        for line in f:
            if 'LMCache hit tokens:' not in line: continue
            for m in PAT.finditer(line):
                first.setdefault(m.group(1), (int(m.group(2)), int(m.group(3))))
    return [n // run.chunk - h // run.chunk for n, h in first.values()]


def scheduler_counters(run: RunInfo) -> dict:
    return {k.split('.', 1)[1]: v for k, v in C.run_counters(run.dir).items() if k.startswith('scheduler.')}


def kappa_curve(run: RunInfo, max_kappa: int) -> dict:
    u = decision_stream(run)
    c = scheduler_counters(run)
    new_chunks = sum(max(0, x) for x in u)
    rows = [{'kappa': k, 'selected_requests': sum(x > k for x in u), 'declined_chunks': sum(x for x in u if x > k)} for k in range(max_kappa + 1)]
    for r in rows:
        r['declined_share'] = r['declined_chunks'] / new_chunks if new_chunks else None
    rec = {'admission': run.admission, 'host_gib_per_rank': run.host_gib, 'write_threshold': run.write_threshold, 'requests': len(u),
           'new_chunks': new_chunks, 'u_histogram': sorted(collections.Counter(u).items()), 'curve': rows, 'counters': c}
    own = rows[run.write_threshold] if run.write_threshold <= max_kappa else None
    if run.admission == 'fixed' and c:
        checks = {'decisions': (len(u), c.get('decisions')),
                  'skipped_requests': (own['selected_requests'], c.get('skipped_requests')),
                  'skipped_new_chunks': (own['declined_chunks'], c.get('skipped_new_chunks')),
                  'saved_new_chunks': (new_chunks - own['declined_chunks'], c.get('saved_new_chunks'))}
        bad = {k: v for k, v in checks.items() if v[0] != v[1]}
        if bad:
            raise ValueError(f'{run.name}: decision stream does not reproduce the connector counters: {bad}')
        rec['validated_against_counters'] = True
    elif run.admission == 'conditioned' and c:
        if len(u) != c.get('decisions'):
            raise ValueError(f'{run.name}: {len(u)} logged decisions, counter {c.get("decisions")}')
        if own is not None and c.get('skipped_requests', 0) > own['selected_requests']:
            raise ValueError(f'{run.name}: more skipped requests than requests with u > kappa')
        rec['pressure_requests'] = c.get('pressure_requests'); rec['telemetry_stale'] = c.get('telemetry_stale')
    return rec


def theta_scan(run: RunInfo, grid_lo: float, grid_step: float) -> dict | None:
    snaps = C.read_jsonl(run.dir / 'telemetry_history.jsonl')
    if not snaps: return None
    ev = [s['registered'] / s['capacity'] for s in snaps if s['evicted_in_window'] > 0 and s['capacity']]
    n_grid = int(round((1.0 - grid_lo) / grid_step)) + 1
    grid = [round(grid_lo + grid_step * i, 3) for i in range(n_grid)]
    return {
        'host_gib_per_rank': run.host_gib, 'snapshots': len(snaps), 'evicting_snapshots': len(ev),
        'min_occupancy_evicting': min(ev) if ev else None,
        'evicting_in_[0.80,0.95)': sum(0.80 <= o < 0.95 for o in ev),
        'evicting_below_0.80': sum(o < 0.80 for o in ev),
        'evicting_at_or_above_0.95': sum(o >= 0.95 for o in ev),
        'evicting_full': sum(o >= 1.0 for o in ev),
        'evicting_occupancies': ev,
        'share_vs_theta': [{'theta': t, 'share_of_evicting_with_occ_ge_theta': sum(o >= t for o in ev) / len(ev) if ev else None} for t in grid]}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', required=True, help='admission run directory (fixed or conditioned), optionally DIR@GIB')
    ap.add_argument('--max-kappa', type=int, default=64)
    ap.add_argument('--theta-grid-start', type=float, default=0.5)
    ap.add_argument('--theta-grid-step', type=float, default=0.005)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    out = {'schema': 'efficientagent.admission_sensitivity', 'version': 1, 'kappa': {}, 'theta': {}}
    for arg in a.run:
        run = resolve_run(arg)
        if run.admission not in ('fixed', 'conditioned'):
            raise SystemExit(f'{run.name}: admission is {run.admission!r}; pass runs with fixed or conditioned admission')
        out['kappa'][run.name] = kappa_curve(run, a.max_kappa)
        if run.admission == 'conditioned':
            th = theta_scan(run, a.theta_grid_start, a.theta_grid_step)
            if th is not None: out['theta'][run.name] = th
    write_json(a.out, out)
    return out


if __name__ == '__main__':
    main()
