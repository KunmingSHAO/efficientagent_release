"""Summary table over replay runs: policy, host capacity, makespan, computed prefill, host traffic, admission counters.

Runs are given as run directories or as parent directories that contain run directories (any directory with a
``run.json``). Writes ``--out`` (JSON) and, with ``--markdown``, a Markdown table.

Usage::

    python -m efficientagent.analysis.run_summary runs/ --out summary.json --markdown summary.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from efficientagent.analysis import run_metrics
from efficientagent.analysis.common import read_run

POLICY_ORDER = {'recompute': 0, 'offload': 1, 'offload (connector)': 2, 'fixed': 3, 'conditioned': 4}
COLUMNS = ('run', 'policy', 'host_gib', 'workers', 'makespan_min', 'prefill_computed_tokens', 'retrieved_per_rank',
           'stored_per_rank', 'preemptions', 'decisions', 'pressure_requests', 'skipped_requests', 'dedup_skipped_chunks')


def find_runs(paths) -> list[Path]:
    runs = []
    for p in map(Path, paths):
        if (p / 'run.json').exists(): runs.append(p)
        elif p.is_dir(): runs += sorted(d for d in p.iterdir() if (d / 'run.json').exists())
    return runs


def policy_name(spec: dict) -> str:
    if not spec: return 'unknown'
    if not spec.get('host_gib'): return 'recompute'
    a = spec.get('admission', 'none')
    if a == 'none': return 'offload (connector)' if spec.get('use_connector') else 'offload'
    return a


def admission_counters(counters: dict) -> dict:
    g = lambda k: counters.get('scheduler.' + k)
    return dict(decisions=g('decisions'), pressure_requests=g('pressure_requests'), skipped_requests=g('skipped_requests'),
                skipped_new_chunks=g('skipped_new_chunks'), telemetry_full_evicting=g('telemetry_full_evicting'),
                estimate_above_capacity=g('estimate_above_capacity'), telemetry_stale=g('telemetry_stale'),
                dedup_skipped_chunks=counters.get('worker.dedup_skipped_chunks'), cpu_evicted_chunks=counters.get('worker.cpu_evicted_chunks'))


def summarize(paths, ranks: int | None = None) -> list[dict]:
    rows = []
    for d in find_runs(paths):
        m = run_metrics.compute(d, ranks); spec = m['spec'] or read_run(d).get('spec') or {}
        rows.append(dict(run=d.name, policy=policy_name(spec), host_gib=spec.get('host_gib'), workers=spec.get('workers'),
                         status=m['status'], tasks=m['tasks'], requests=m['requests'], forced_mismatch=m['forced_mismatch'],
                         makespan_s=m['makespan_s'], makespan_min=m['makespan_min'], prompt_tokens=m['prompt_tokens'],
                         prefill_computed_tokens=m['prefill_computed_tokens'], retrieved_per_rank=m['retrieved_per_rank'],
                         stored_per_rank=m['stored_per_rank'], gpu_prefix_hit_tokens=m['gpu_prefix_hit_tokens'], preemptions=m['preemptions'],
                         mean_queue_s=m['mean_queue_s'], **admission_counters(m['kvtier_counters'])))
    rows.sort(key=lambda r: (r['workers'] or 0, POLICY_ORDER.get(r['policy'], 9), r['host_gib'] or 0, r['run']))
    return rows


def fmt(v) -> str:
    if v is None: return '-'
    if isinstance(v, float) and abs(v) >= 1e6: return f'{v / 1e6:,.2f}M'
    if isinstance(v, float): return f'{int(v):,}' if v.is_integer() else f'{v:,.1f}'
    if isinstance(v, int) and abs(v) >= 1e6: return f'{v / 1e6:,.2f}M'
    return f'{v:,}' if isinstance(v, int) else str(v)


def markdown(rows: list[dict]) -> str:
    lines = ['| ' + ' | '.join(COLUMNS) + ' |', '|' + '---|' * len(COLUMNS)]
    for r in rows: lines.append('| ' + ' | '.join(fmt(r.get(c)) for c in COLUMNS) + ' |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', help='run directories or parents of run directories')
    ap.add_argument('--out', type=Path, required=True, help='output JSON')
    ap.add_argument('--markdown', type=Path, help='optional Markdown table')
    ap.add_argument('--ranks', type=int, help='TP size used to normalize LMCache volumes (default: each run spec.tp)')
    a = ap.parse_args(argv)
    rows = summarize(a.paths, a.ranks)
    a.out.write_text(json.dumps(dict(runs=rows), indent=1, default=str) + '\n')
    if a.markdown: a.markdown.write_text(markdown(rows))
    print(markdown(rows))


if __name__ == '__main__':
    main()
