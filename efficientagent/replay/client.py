"""Dependency-preserving closed-loop trace replay client.

``workers`` task slots take tasks in trace order. Within a task, step i is sent only after step i-1 has completed and
the recorded gap before step i (tool execution and agent-side processing) has elapsed; after the last step the
task's recorded trailing gap elapses before the slot takes the next task. Prompts are the exact recorded token IDs,
and outputs are forced to the recorded completion tokens through the server's forced-output logits processor
(``max_tokens`` = recorded length, ``ignore_eos``). Every run therefore submits the same token work.

Outputs in ``--out``: ``requests.jsonl`` (one row per request), ``tasks.jsonl`` (one row per task) and
``replay_summary.json``.

Usage::

    python -m efficientagent.replay.client --trace TRACE_DIR --out OUT_DIR --base http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import gzip
import json
import threading
import time
import urllib.request
from pathlib import Path
from queue import Empty, Queue


def load_task(path) -> tuple[dict, list[dict]]:
    with gzip.open(path, 'rt') as z:
        head = json.loads(z.readline()); steps = [json.loads(line) for line in z]
    return head, steps


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(base: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(base + '/v1/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with _opener().open(req, timeout=timeout) as r: return json.loads(r.read().decode())


def served_model(base: str, timeout: float = 30) -> str:
    """Name of the first model served at ``base`` (``/v1/models``)."""
    with _opener().open(base + '/v1/models', timeout=timeout) as r:
        return json.loads(r.read().decode())['data'][0]['id']


def request_body(model: str, prompt: list[int], step: dict, task_id: str) -> dict:
    """Completion request for one recorded step: exact prompt, forced recorded output, task ID for the connector."""
    forced = step['output'] if step['output_len'] > 0 else None
    body = dict(model=model, prompt=prompt, max_tokens=max(1, step['output_len']), temperature=0, top_p=1, ignore_eos=True,
                return_token_ids=True, skip_special_tokens=False,
                kv_transfer_params={'efficientagent.task_id': task_id, 'efficientagent.request_sequence': step['seq']})
    if forced: body['vllm_xargs'] = {'ea_forced_output': ','.join(map(str, forced))}
    return body


def run(trace_dir, out, base: str, model: str | None = None, workers: int = 16, gap_scale: float = 1.0,
        limit: int | None = None, timeout: float = 7200, stop: threading.Event | None = None) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(trace_dir).glob('*.jsonl.gz'))
    if limit: files = files[:limit]
    if not files: raise FileNotFoundError(f'no *.jsonl.gz trace files in {trace_dir}')
    model = model or served_model(base)
    q = Queue(); [q.put(f) for f in files]
    lock = threading.Lock(); stats = dict(requests=0, forced_match=0, forced_mismatch=0, errors=0, prompt_len_mismatch=0)
    reqf = (out / 'requests.jsonl').open('a'); taskf = (out / 'tasks.jsonl').open('a')
    t0 = time.time()

    def worker(wid):
        while not (stop and stop.is_set()):
            try: f = q.get_nowait()
            except Empty: return
            head, steps = load_task(f); inst = head['instance_id']; assigned = time.time(); prompt = []; ok = True
            for s in steps:
                if stop and stop.is_set(): return
                time.sleep(s['gap_before_s'] * gap_scale)
                prompt = prompt[:s['prompt_shared']] + s['prompt_suffix']
                body = request_body(model, prompt, s, inst); forced = s['output'] if s['output_len'] > 0 else None
                st = time.time(); row = dict(instance_id=inst, seq=s['seq'], worker=wid, start=st, prompt_len=len(prompt), output_len=s['output_len'])
                try:
                    r = post(base, body, timeout); ch = r['choices'][0]; ids = ch.get('token_ids')
                    row.update(end=time.time(), usage=r.get('usage'), finish_reason=ch.get('finish_reason'),
                               forced_match=(ids == forced) if forced else None,
                               prompt_ok=(r.get('usage') or {}).get('prompt_tokens') == len(prompt) == s['prompt_len'])
                except Exception as e:
                    row.update(end=time.time(), error=repr(e)[:500]); ok = False
                row['latency_s'] = row['end'] - st
                with lock:
                    stats['requests'] += 1
                    if 'error' in row: stats['errors'] += 1
                    elif row['forced_match'] is True: stats['forced_match'] += 1
                    elif row['forced_match'] is False: stats['forced_mismatch'] += 1
                    if 'error' not in row and not row['prompt_ok']: stats['prompt_len_mismatch'] += 1
                    reqf.write(json.dumps(row) + '\n'); reqf.flush()
            time.sleep(head.get('tail_gap_s', 0.0) * gap_scale)
            with lock:
                taskf.write(json.dumps(dict(instance_id=inst, order=head['order'], worker=wid, assigned=assigned, finished=time.time(),
                                            jct_s=time.time() - assigned, recorded_jct_s=head.get('recorded_jct_s'), steps=len(steps), ok=ok)) + '\n')
                taskf.flush()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(min(workers, len(files)))]
    [t.start() for t in threads]; [t.join() for t in threads]
    reqf.close(); taskf.close()
    end = time.time()
    summary = dict(stats, tasks=len(files), workers=workers, gap_scale=gap_scale, model=model, start=t0, end=end, makespan_s=end - t0,
                   all_outputs_forced_exactly=stats['forced_mismatch'] == 0 and stats['errors'] == 0,
                   interrupted=bool(stop and stop.is_set()))
    (out / 'replay_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trace', type=Path, required=True, help='trace directory (*.jsonl.gz)')
    ap.add_argument('--out', type=Path, required=True, help='output directory')
    ap.add_argument('--base', default='http://127.0.0.1:8000', help='server base URL')
    ap.add_argument('--model', help='served model name (default: first entry of /v1/models)')
    ap.add_argument('--workers', type=int, default=16, help='active task slots (active pool)')
    ap.add_argument('--gap-scale', type=float, default=1.0, help='multiplier on recorded gaps')
    ap.add_argument('--limit', type=int, help='replay only the first N tasks')
    ap.add_argument('--timeout', type=float, default=7200, help='per-request HTTP timeout, seconds')
    a = ap.parse_args(argv)
    print(json.dumps(run(a.trace, a.out, a.base, a.model, a.workers, a.gap_scale, a.limit, a.timeout), indent=1))


if __name__ == '__main__':
    main()
