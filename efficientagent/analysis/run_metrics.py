"""Per-run metrics of one replay run directory -> ``<run>/run_metrics.json``.

Workload-level measurements between the Prometheus snapshots at the start and end of the replay:

  makespan_s                 replay makespan (first dispatch to last task end; recorded gaps included)
  prefill_computed_tokens    delta of vllm:request_prefill_kv_computed_tokens_sum
  preemptions                delta of vllm:num_preemptions_total
  retrieved_per_rank         LMCache restored tokens (log lines summed over workers, divided by the TP size)
  stored_per_rank            LMCache stored tokens (same aggregation)
  gpu_prefix_hit_tokens      prompt tokens - computed prefill - restored per rank
  mean_queue_s, ...          means of vLLM's request-time histograms

plus request latency and task completion statistics, scheduler gauges, and connector counters.

Usage::

    python -m efficientagent.analysis.run_metrics RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from efficientagent.analysis import common as C

HIST = ('vllm:request_queue_time_seconds', 'vllm:request_prefill_time_seconds', 'vllm:request_decode_time_seconds',
        'vllm:e2e_request_latency_seconds', 'vllm:time_to_first_token_seconds', 'vllm:request_inference_time_seconds')
COUNTERS = ('vllm:request_success_total', 'vllm:prompt_tokens_total', 'vllm:generation_tokens_total', 'vllm:num_preemptions_total',
            'vllm:prefix_cache_queries_total', 'vllm:prefix_cache_hits_total', 'vllm:request_prefill_kv_computed_tokens_sum')


def gauges(run_dir) -> dict:
    rows = [r for r in C.read_jsonl(Path(run_dir) / 'load_samples.jsonl') if 'vllm' in r]
    if not rows: return dict(samples=0)
    run = [r['vllm'].get('vllm:num_requests_running', 0) for r in rows]; wait = [r['vllm'].get('vllm:num_requests_waiting', 0) for r in rows]
    kv = [r['vllm'].get('vllm:kv_cache_usage_perc', 0) for r in rows]; n = len(rows)
    return dict(samples=n, mean_running=sum(run) / n, mean_waiting=sum(wait) / n, mean_kv_usage=sum(kv) / n,
                frac_waiting_gt0=sum(w > 0 for w in wait) / n,
                running_hist={str(k): v for k, v in sorted(Counter(int(x) for x in run).items())})


def serving(run_dir) -> dict:
    o, f = C.read_prom(Path(run_dir) / 'metrics_start.prom'), C.read_prom(Path(run_dir) / 'metrics_end.prom')
    out = {k: C.prom_delta(o, f, k) for k in COUNTERS}
    for h in HIST:
        s, c = C.prom_delta(o, f, h + '_sum'), C.prom_delta(o, f, h + '_count')
        b = {le: f.get(h + '_bucket', {}).get(le, 0) - o.get(h + '_bucket', {}).get(le, 0) for le in f.get(h + '_bucket', {})}
        out[h] = dict(sum_s=s, count=c, mean_s=(s / c) if c else None, p50_le=C.hist_quantile(b, .5), p90_le=C.hist_quantile(b, .9),
                      p99_le=C.hist_quantile(b, .99))
    return out


def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * (len(v) - 1)))] if v else None


def compute(run_dir, ranks: int | None = None) -> dict:
    run_dir = Path(run_dir); run = C.read_run(run_dir); spec = run.get('spec') or {}
    ranks = ranks or int(spec.get('tp') or 1)
    summ = C.read_summary(run_dir); reqs = C.read_requests(run_dir); tasks = C.read_tasks(run_dir)
    srv = serving(run_dir); lm = C.lmcache_log_totals(run_dir / 'server.log', ranks)
    prompt = srv.get('vllm:prompt_tokens_total'); pre = srv.get('vllm:request_prefill_kv_computed_tokens_sum')
    lat = [r['latency_s'] for r in reqs if 'error' not in r]; jct = [t['jct_s'] for t in tasks]
    return dict(
        run=run_dir.name, spec=spec, status=run.get('status'),
        makespan_s=summ.get('makespan_s'), makespan_min=(summ['makespan_s'] / 60) if summ.get('makespan_s') is not None else None,
        tasks=len(tasks), requests=len(reqs), forced_mismatch=summ.get('forced_mismatch'), http_errors=summ.get('errors'),
        prompt_tokens=prompt, prefill_computed_tokens=pre, preemptions=srv.get('vllm:num_preemptions_total'),
        retrieved_per_rank=lm.get('retrieved_per_rank', 0.0), stored_per_rank=lm.get('stored_per_rank', 0.0),
        gpu_prefix_hit_tokens=(prompt - pre - lm.get('retrieved_per_rank', 0.0)) if prompt is not None and pre is not None else None,
        mean_queue_s=srv['vllm:request_queue_time_seconds']['mean_s'], mean_prefill_s=srv['vllm:request_prefill_time_seconds']['mean_s'],
        mean_decode_s=srv['vllm:request_decode_time_seconds']['mean_s'],
        request_latency=dict(n=len(lat), mean_s=statistics.mean(lat) if lat else None, p50_s=q(lat, .5), p90_s=q(lat, .9)),
        task_jct=dict(n=len(jct), mean_s=statistics.mean(jct) if jct else None, median_s=statistics.median(jct) if jct else None,
                      max_s=max(jct) if jct else None),
        gpu_kv_tokens=run.get('gpu_kv_tokens'), serving_delta=srv, lmcache_log=lm, gauges=gauges(run_dir),
        kvtier_counters=C.run_counters(run_dir), kvtier_first_hits=C.first_hits(run_dir / 'server.log'))


def write(run_dir, ranks: int | None = None) -> dict:
    out = compute(run_dir, ranks)
    (Path(run_dir) / 'run_metrics.json').write_text(json.dumps(out, indent=1, default=str) + '\n')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', type=Path, help='run directories')
    ap.add_argument('--ranks', type=int, help='TP size used to normalize LMCache volumes (default: spec.tp)')
    a = ap.parse_args(argv)
    for d in a.runs:
        m = write(d, a.ranks)
        print(json.dumps({k: m[k] for k in ('run', 'status', 'makespan_min', 'prefill_computed_tokens', 'retrieved_per_rank', 'stored_per_rank')}, default=str))


if __name__ == '__main__':
    main()
