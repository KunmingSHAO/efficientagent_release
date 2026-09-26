"""Build a dependency-preserving replay trace from agent logs with exact token IDs.

Input: an agent run directory with one record per task and one LLM-call log per attempt::

    RUN/tasks/<task_id>.json
        {"attempt_index": 0, "trial_id": "...", "task_timing": {"assigned_ns": ..., "finished_ns": ...}}
    RUN/attempts/<task_id>/attempt_<NN>/telemetry/llm_requests.jsonl      (one JSON object per call)
        {"trial_id": "...", "request_sequence": 3, "error": null, "token_identity_verified": true,
         "prompt_token_ids": [...], "generated_token_ids": [[...]],
         "start_wall_ns": ..., "end_wall_ns": ..., "client_elapsed_ns": ...}

The attempt used for a task is ``attempt_index`` of its task record. Only successful calls (``error`` is null and,
when present, ``token_identity_verified`` is true) become replay steps; the time of failed calls stays inside the
gaps. Per step the trace stores the exact prompt token IDs (delta-encoded against the task's previous prompt), the
exact completion token IDs to force, and the client-side gap before the call: for the first step, task assignment
to call start; later, previous call end to this call start (tool execution and agent-side processing). The
trailing gap from the last call to the task's finish is stored in the task header.

Output: ``OUT/NNN_<task_id>.jsonl.gz`` (header line, then one line per step, NNN = dispatch order) and
``OUT/trace_manifest.json``. Task order is given by ``--tasks`` (a JSON list of task IDs) or, by default, by
assignment time.

Usage::

    python -m efficientagent.replay.build_trace --run-dir RUN --out TRACE_DIR [--tasks tasks.json] [--limit N]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

SCHEMA = 'efficientagent.replay_trace.v1'


def _safe(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)


def task_order(run_dir: Path, tasks_file: Path | None) -> list[str]:
    if tasks_file is not None:
        return list(json.loads(Path(tasks_file).read_text()))
    recs = {p.stem: json.loads(p.read_text()) for p in sorted((run_dir / 'tasks').glob('*.json'))}
    return sorted(recs, key=lambda k: ((recs[k].get('task_timing') or {}).get('assigned_ns') or 0, k))


def build_task(run_dir: Path, inst: str) -> tuple[dict, list[dict]]:
    """-> (header, steps) of one task."""
    d = json.loads((run_dir / 'tasks' / f'{inst}.json').read_text())
    timing = d['task_timing']; att_rel = Path('attempts') / inst / ('attempt_%02d' % (d.get('attempt_index') or 0))
    calls = [json.loads(line) for line in open(run_dir / att_rel / 'telemetry/llm_requests.jsonl')]
    if any(c.get('trial_id') != d.get('trial_id') for c in calls):
        raise ValueError(f'{inst}: call log does not belong to the recorded trial')
    calls.sort(key=lambda c: c['request_sequence'])
    ok = [c for c in calls if c.get('error') is None and c.get('token_identity_verified', True)]
    steps = []; prev_prompt = []; prev_end = timing['assigned_ns']
    for c in ok:
        p = c['prompt_token_ids']; g = c['generated_token_ids'][0]
        shared = 0; n = min(len(p), len(prev_prompt))
        while shared < n and p[shared] == prev_prompt[shared]: shared += 1
        steps.append(dict(seq=c['request_sequence'], gap_before_s=max(0.0, (c['start_wall_ns'] - prev_end) / 1e9),
                          prompt_shared=shared, prompt_suffix=p[shared:], prompt_len=len(p), output=g, output_len=len(g),
                          recorded_client_elapsed_s=c['client_elapsed_ns'] / 1e9,
                          prompt_sha256=hashlib.sha256(json.dumps(p).encode()).hexdigest()))
        prev_prompt = p; prev_end = c['end_wall_ns']
    head = dict(instance_id=inst, trial_id=d.get('trial_id'), attempt=att_rel.as_posix(),
                tail_gap_s=max(0.0, (timing['finished_ns'] - prev_end) / 1e9),
                recorded_jct_s=(timing['finished_ns'] - timing['assigned_ns']) / 1e9,
                calls_total=len(calls), calls_replayed=len(ok))
    return head, steps


def build(run_dir, out, tasks_file=None, limit: int | None = None) -> dict:
    run_dir, out = Path(run_dir), Path(out)
    ids = task_order(run_dir, tasks_file)
    if limit: ids = ids[:limit]
    out.mkdir(parents=True, exist_ok=True); manifest = []
    for order, inst in enumerate(ids):
        head, steps = build_task(run_dir, inst)
        head = dict(head, order=order)
        f = out / ('%03d_%s.jsonl.gz' % (order, _safe(inst)))
        with gzip.open(f, 'wt') as z:
            z.write(json.dumps(head) + '\n')
            for s in steps: z.write(json.dumps(s) + '\n')
        manifest.append(dict(order=order, instance_id=inst, file=f.name, sha256=hashlib.sha256(f.read_bytes()).hexdigest(), steps=len(steps),
                             calls_total=head['calls_total'], prompt_tokens=sum(s['prompt_len'] for s in steps),
                             output_tokens=sum(s['output_len'] for s in steps),
                             gap_total_s=sum(s['gap_before_s'] for s in steps) + head['tail_gap_s'], recorded_jct_s=head['recorded_jct_s']))
    summary = dict(schema=SCHEMA, tasks=len(manifest), steps=sum(m['steps'] for m in manifest),
                   prompt_tokens=sum(m['prompt_tokens'] for m in manifest), output_tokens=sum(m['output_tokens'] for m in manifest),
                   gap_total_s=sum(m['gap_total_s'] for m in manifest),
                   task_list_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), files=manifest)
    (out / 'trace_manifest.json').write_text(json.dumps(summary, indent=1) + '\n')
    return {k: v for k, v in summary.items() if k != 'files'} | dict(skipped_calls=sum(m['calls_total'] - m['steps'] for m in manifest))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', type=Path, required=True, help='agent run directory (tasks/, attempts/)')
    ap.add_argument('--out', type=Path, required=True, help='output trace directory')
    ap.add_argument('--tasks', type=Path, help='JSON list of task IDs in dispatch order (default: by assignment time)')
    ap.add_argument('--limit', type=int, help='use only the first N tasks')
    a = ap.parse_args(argv)
    print(json.dumps(build(a.run_dir, a.out, a.tasks, a.limit), indent=1))


if __name__ == '__main__':
    main()
