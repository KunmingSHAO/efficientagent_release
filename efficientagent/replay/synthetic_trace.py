"""Generate a small synthetic agent trace in the replay trace format.

Each task starts from a system prompt shared by all tasks and a task-specific instruction; every call appends the
previous output and a new observation to the context, so consecutive prompts of a task share their prefix as in
append-only agent histories. Token IDs are drawn uniformly from ``[--token-min, --token-max)``; choose a range
inside the vocabulary of the served model. Useful for trying the replay pipeline end to end.

Usage::

    python -m efficientagent.replay.synthetic_trace --out /tmp/toy_trace --tasks 8 --steps 6 12
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path


def generate(out, tasks: int = 8, steps: tuple[int, int] = (6, 12), system_tokens: int = 2048, task_tokens: int = 1024,
             observation_tokens: tuple[int, int] = (200, 1500), output_tokens: tuple[int, int] = (20, 200),
             gap_s: tuple[float, float] = (0.05, 0.5), token_range: tuple[int, int] = (1000, 30000), seed: int = 0) -> dict:
    rng = random.Random(seed); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    tok = lambda n: [rng.randrange(*token_range) for _ in range(n)]
    system = tok(system_tokens); files = []
    for order in range(tasks):
        inst = f'task-{order:03d}'; context = system + tok(task_tokens); prev = []; rows = []
        for seq in range(1, rng.randint(*steps) + 1):
            if seq > 1: context = context + tok(rng.randint(*observation_tokens))
            shared = 0; n = min(len(context), len(prev))
            while shared < n and context[shared] == prev[shared]: shared += 1
            output = tok(rng.randint(*output_tokens))
            rows.append(dict(seq=seq, gap_before_s=round(rng.uniform(*gap_s), 3), prompt_shared=shared, prompt_suffix=context[shared:],
                             prompt_len=len(context), output=output, output_len=len(output), recorded_client_elapsed_s=0.0,
                             prompt_sha256=hashlib.sha256(json.dumps(context).encode()).hexdigest()))
            prev = list(context); context = context + output
        head = dict(instance_id=inst, order=order, tail_gap_s=round(rng.uniform(*gap_s), 3), recorded_jct_s=0.0,
                    calls_total=len(rows), calls_replayed=len(rows))
        f = out / f'{order:03d}_{inst}.jsonl.gz'
        with gzip.open(f, 'wt') as z:
            z.write(json.dumps(head) + '\n')
            for r in rows: z.write(json.dumps(r) + '\n')
        files.append(dict(order=order, instance_id=inst, file=f.name, steps=len(rows)))
    summary = dict(schema='efficientagent.replay_trace.v1', synthetic=True, seed=seed, tasks=tasks, steps=sum(f['steps'] for f in files), files=files)
    (out / 'trace_manifest.json').write_text(json.dumps(summary, indent=1) + '\n')
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tasks', type=int, default=8)
    ap.add_argument('--steps', type=int, nargs=2, default=(6, 12), metavar=('MIN', 'MAX'))
    ap.add_argument('--system-tokens', type=int, default=2048)
    ap.add_argument('--task-tokens', type=int, default=1024)
    ap.add_argument('--token-min', type=int, default=1000); ap.add_argument('--token-max', type=int, default=30000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    s = generate(a.out, a.tasks, tuple(a.steps), a.system_tokens, a.task_tokens, token_range=(a.token_min, a.token_max), seed=a.seed)
    print(json.dumps({k: v for k, v in s.items() if k != 'files'}))


if __name__ == '__main__':
    main()
