"""Compare two replay runs of the same trace: run-level metrics side by side and paired per-request latencies.

Both runs replay the same request stream (same prompts, forced outputs, recorded gaps), so requests pair up by
(task, sequence number). Reports the fraction of requests faster/slower in the right run, latency-difference
quantiles, a breakdown by prompt length, and, when both runs recorded ``greedy_check.json``, whether the unforced
greedy outputs agree.

Usage::

    python -m efficientagent.analysis.compare_runs LEFT_RUN RIGHT_RUN --out compare.json [--markdown compare.md]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from efficientagent.analysis import run_metrics
from efficientagent.analysis.common import read_requests

BUCKETS = [(0, 16384), (16384, 32768), (32768, 65536), (65536, 131072), (131072, 10**9)]
METRICS = ('makespan_min', 'prefill_computed_tokens', 'retrieved_per_rank', 'stored_per_rank', 'preemptions', 'mean_queue_s',
           'mean_prefill_s', 'mean_decode_s')


def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * (len(v) - 1)))] if v else None


def greedy_agreement(left: Path, right: Path) -> dict | None:
    a, b = left / 'greedy_check.json', right / 'greedy_check.json'
    if not (a.exists() and b.exists()): return None
    ra = {(r['instance_id'], r['seq']): r['tokens'] for r in json.loads(a.read_text())}
    rb = {(r['instance_id'], r['seq']): r['tokens'] for r in json.loads(b.read_text())}
    keys = sorted(set(ra) & set(rb))
    return dict(pairs=len(keys), identical=sum(ra[k] == rb[k] for k in keys))


def compare(left, right, ranks: int | None = None) -> dict:
    left, right = Path(left), Path(right)
    ma, mb = run_metrics.compute(left, ranks), run_metrics.compute(right, ranks)
    ra = {(r['instance_id'], r['seq']): r for r in read_requests(left) if 'error' not in r}
    rb = {(r['instance_id'], r['seq']): r for r in read_requests(right) if 'error' not in r}
    keys = sorted(set(ra) & set(rb)); diffs, ratios = [], []; per_bucket = defaultdict(list)
    for k in keys:
        x, y = ra[k]['latency_s'], rb[k]['latency_s']; diffs.append(y - x)
        if x > 0: ratios.append(y / x)
        for lo, hi in BUCKETS:
            if lo <= ra[k]['prompt_len'] < hi: per_bucket[f'{lo}-{hi}'].append((x, y))
    n = len(diffs)
    paired = dict(paired_requests=n, frac_right_faster=sum(d < 0 for d in diffs) / n if n else None,
                  frac_right_slower=sum(d > 0 for d in diffs) / n if n else None,
                  diff_s_quantiles={str(p): q(diffs, p) for p in (.05, .25, .5, .75, .95)},
                  ratio_quantiles={str(p): q(ratios, p) for p in (.05, .25, .5, .75, .95)})
    buckets = {k: dict(n=len(v), mean_left_s=statistics.mean(x for x, _ in v), mean_right_s=statistics.mean(y for _, y in v),
                       median_diff_s=statistics.median(y - x for x, y in v), frac_right_faster=sum(y < x for x, y in v) / len(v))
               for k, v in per_bucket.items() if v}
    return dict(left=ma['run'], right=mb['run'], metrics={m: dict(left=ma.get(m), right=mb.get(m)) for m in METRICS},
                paired=paired, prompt_length_buckets=buckets, greedy_check=greedy_agreement(left, right),
                left_spec=ma['spec'], right_spec=mb['spec'])


def markdown(c: dict) -> str:
    f = lambda v: '-' if v is None else (f'{v:,.2f}' if isinstance(v, float) else str(v))
    lines = [f"| metric | {c['left']} | {c['right']} |", '|---|---:|---:|']
    for m, v in c['metrics'].items(): lines.append(f"| {m} | {f(v['left'])} | {f(v['right'])} |")
    p = c['paired']
    lines += ['', f"Paired requests: {p['paired_requests']}; right faster {f(p['frac_right_faster'])}, slower {f(p['frac_right_slower'])}.",
              'Latency difference (right - left) quantiles, s: ' + ', '.join(f'p{int(float(k) * 100)} {f(v)}' for k, v in p['diff_s_quantiles'].items())]
    if c['greedy_check']: lines.append(f"Greedy outputs identical: {c['greedy_check']['identical']} of {c['greedy_check']['pairs']}.")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('left', type=Path); ap.add_argument('right', type=Path)
    ap.add_argument('--out', type=Path, required=True); ap.add_argument('--markdown', type=Path)
    ap.add_argument('--ranks', type=int, help='TP size used to normalize LMCache volumes (default: each run spec.tp)')
    a = ap.parse_args(argv)
    c = compare(a.left, a.right, a.ranks)
    a.out.write_text(json.dumps(c, indent=1, default=str) + '\n')
    if a.markdown: a.markdown.write_text(markdown(c))
    print(markdown(c))


if __name__ == '__main__':
    main()
