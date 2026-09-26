"""Launch one replay run: a fresh vLLM (+ LMCache) server, the closed-loop replay, then per-run analysis.

Each run starts a new serving process with a cold cache, replays the trace with forced outputs, records Prometheus
snapshots at the start and end of the replay, scheduler gauges every 2 s, connector counters and (for
capacity-conditioned admission) the host-tier telemetry, stops the server and writes ``run.json`` and
``run_metrics.json`` into ``--out``.

Admission options (``--admission``):

  none         offload without write admission (native LMCache path; ``--use-connector`` routes it through the
               EfficientAgent connector with admission disabled). ``--host-gib 0`` runs without a host tier.
  fixed        fixed write admission (request filter on every request, plus copy-time deduplication).
  conditioned  capacity-conditioned admission (request filter under pressure, plus copy-time deduplication).

``--host-gib`` is the host (CPU) tier capacity per tensor-parallel rank. Without ``--execute`` the launcher only
prints the server command and environment.

Usage::

    python -m efficientagent.replay.launch --admission conditioned --host-gib 5 \\
        --model /models/Qwen3-Coder-30B-A3B-Instruct --tp 8 --trace TRACE_DIR --out runs/conditioned_5gib --execute
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

from efficientagent.kvtier.policies import (ADMISSION_OPTIONS, DESCRIPTIONS, AdmissionConfig, TierPlan,
                                            kv_bytes_per_token_from_config)
from efficientagent.replay import client

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OBSERVER_DIR = Path(__file__).resolve().parent / 'lmcache_observer'
LOGITS_PROCESSOR = 'efficientagent.replay.forced_output:ForcedOutputLogitsProcessor'
GAUGES = ('vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:kv_cache_usage_perc', 'vllm:num_preemptions_total',
          'vllm:request_success_total', 'vllm:prompt_tokens_total', 'vllm:generation_tokens_total', 'vllm:prefix_cache_queries_total',
          'vllm:prefix_cache_hits_total')
ENV_RECORDED = ('EA_', 'LMCACHE', 'VLLM_PLUGINS', 'PYTHONPATH', 'PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES')


# ---------------------------------------------------------------------------------------------------- small utilities
def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial'); tmp.write_text(json.dumps(value, indent=2, default=str) + '\n'); tmp.replace(path)


def append_jsonl(path, value) -> None:
    with Path(path).open('a') as f: f.write(json.dumps(value, default=str) + '\n')


def http_get(base: str, path: str, timeout: float = 10) -> str:
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with op.open(base + path, timeout=timeout) as r: return r.read().decode()


def parse_prom_flat(raw: str) -> dict:
    vals = {}
    for line in raw.splitlines():
        if not line or line[0] == '#': continue
        try:
            k, v = line.rsplit(' ', 1); stem = k.split('{')[0]
            vals[stem] = vals.get(stem, 0.0) + float(v)
        except ValueError: pass
    return vals


def compute_apps() -> str | None:
    """Compute processes on the visible GPUs (nvidia-smi), or None when nvidia-smi is unavailable."""
    try:
        q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], capture_output=True, text=True, timeout=30)
        return q.stdout.strip() if q.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def wait_for_idle_gpus(seconds: float) -> dict:
    """Wait until no compute process runs on the GPUs, at most ``seconds``."""
    deadline = time.monotonic() + seconds
    while True:
        apps = compute_apps()
        if apps == '': return dict(idle=True, time=time.time())
        if time.monotonic() > deadline: raise RuntimeError(f'GPUs not idle after {seconds:.0f} s: {apps!r}')
        time.sleep(15)


# ---------------------------------------------------------------------------------------------------- server command
def build_plan(a: argparse.Namespace) -> TierPlan:
    bpt = a.kv_bytes_per_token
    if bpt is None and Path(a.model).is_dir() and (Path(a.model) / 'config.json').exists():
        bpt = kv_bytes_per_token_from_config(a.model, a.tp)
    params = AdmissionConfig(write_threshold=a.write_threshold, occupancy_threshold=a.occupancy_threshold, window_s=a.window_s,
                             prompt_window=a.prompt_window, telemetry_max_age_s=a.telemetry_max_age_s, chunk_tokens=a.chunk_tokens,
                             bytes_per_token=bpt)
    return TierPlan(a.admission, a.host_gib, tp=a.tp, use_connector=a.use_connector, params=params)


def server_command(a: argparse.Namespace, plan: TierPlan, out: Path) -> tuple[list[str], dict]:
    cmd = [a.python, '-m', 'vllm.entrypoints.openai.api_server', '--model', a.model, '--served-model-name', a.served_model_name or a.model,
           '--enable-prefix-caching', '--disable-hybrid-kv-cache-manager', '--host', a.host, '--port', str(a.port),
           '--tensor-parallel-size', str(a.tp)]
    if a.gpu_memory_utilization is not None: cmd += ['--gpu-memory-utilization', str(a.gpu_memory_utilization)]
    if a.max_model_len is not None: cmd += ['--max-model-len', str(a.max_model_len)]
    if a.max_num_seqs is not None: cmd += ['--max-num-seqs', str(a.max_num_seqs)]
    cmd += plan.server_args() + ['--logits-processors', LOGITS_PROCESSOR] + list(a.server_arg or [])
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LMCACHE_', 'VLLM_', 'EA_'))}
    paths = ([str(OBSERVER_DIR)] if plan.host_tier and not a.no_observer else []) + [str(PACKAGE_ROOT)]
    if env.get('PYTHONPATH'): paths.append(env['PYTHONPATH'])
    env.update(PYTHONPATH=os.pathsep.join(paths), PYTHONUNBUFFERED='1', PYTHONHASHSEED='0', TMPDIR=str(a.tmpdir))
    env.update(plan.lmcache_env())
    env.update(plan.connector_env(stats_dir=out / 'kvtier_stats', telemetry_file=out / 'kvtier_stats/telemetry_rank0.json'))
    if plan.host_tier and not a.no_observer: env['EA_LMCACHE_OBSERVER_DIR'] = str(out / 'lmcache_observer')
    return cmd, env


def recorded_env(env: dict) -> dict:
    return {k: env[k] for k in sorted(env) if k.startswith(ENV_RECORDED)}


def spec_of(a: argparse.Namespace, plan: TierPlan) -> dict:
    return dict(admission=plan.admission, description=DESCRIPTIONS[plan.admission] if plan.host_tier else 'no host tier (recompute)',
                host_gib=plan.host_gib, use_connector=plan.use_connector, native_lmcache=plan.native, workers=a.workers, tp=a.tp,
                model=a.model, served_model_name=a.served_model_name or a.model, write_threshold=plan.params.write_threshold,
                occupancy_threshold=plan.params.occupancy_threshold, window_s=plan.params.window_s, prompt_window=plan.params.prompt_window,
                telemetry_max_age_s=plan.params.telemetry_max_age_s, bytes_per_token=plan.params.bytes_per_token,
                chunk_tokens=plan.params.chunk_tokens, trace_dir=str(a.trace), limit=a.limit, max_seconds=a.max_seconds)


# ---------------------------------------------------------------------------------------------------- run-time recorders
def metrics_sampler(base: str, out: Path, stop: threading.Event, interval: float = 2.0) -> None:
    with gzip.open(out / 'metrics.jsonl.gz', 'at', compresslevel=1) as raw_log:
        while not stop.is_set():
            t0 = time.monotonic(); row = dict(time=time.time())
            try:
                raw = http_get(base, '/metrics', timeout=5); raw_log.write(json.dumps(dict(time=row['time'], prometheus=raw)) + '\n')
                v = parse_prom_flat(raw); row['vllm'] = {k: v[k] for k in GAUGES if k in v}
                row['lmcache'] = {k: x for k, x in v.items() if k.startswith('lmcache')}
            except Exception as e:
                row['error'] = repr(e)
            append_jsonl(out / 'load_samples.jsonl', row)
            stop.wait(max(0.1, interval - (time.monotonic() - t0)))


def telemetry_history(src: Path, dst: Path, stop: threading.Event, interval: float = 5.0) -> None:
    """Copy the rank-0 telemetry report every ``interval`` seconds (read only; the server owns ``src``)."""
    last = None
    while not stop.wait(interval):
        try: row = json.loads(Path(src).read_text())
        except (OSError, ValueError): continue
        if row.get('t') != last:
            last = row.get('t'); row['read_t'] = time.time(); append_jsonl(dst, row)


def greedy_check(base: str, model: str, trace_dir: Path, tasks: int, out: Path, every: int = 5, max_tokens: int = 8) -> int:
    """Unforced greedy generations on recorded prompts (every ``every``-th and the last step of the first ``tasks``
    tasks); comparing ``greedy_check.json`` across runs checks that restored KV state yields the same tokens."""
    rows = []
    for f in sorted(Path(trace_dir).glob('*.jsonl.gz'))[:tasks]:
        head, steps = client.load_task(f); prompt = []
        for i, s in enumerate(steps):
            prompt = prompt[:s['prompt_shared']] + s['prompt_suffix']
            if i % every == every - 1 or i == len(steps) - 1:
                r = client.post(base, dict(model=model, prompt=prompt, max_tokens=max_tokens, temperature=0, return_token_ids=True,
                                           skip_special_tokens=False), timeout=1800)
                rows.append(dict(instance_id=head['instance_id'], seq=s['seq'], prompt_len=len(prompt), tokens=r['choices'][0].get('token_ids')))
    write_json(out / 'greedy_check.json', rows)
    return len(rows)


def observer_summary(out: Path) -> dict:
    res = {}
    d = out / 'lmcache_observer'
    for q in sorted(d.glob('cpu-*.jsonl')) if d.exists() else []:
        rows = [json.loads(line) for line in open(q) if '"keys"' in line]
        if rows:
            res[q.name] = dict(samples=len(rows), max_keys=max(r['keys'] for r in rows), mean_keys=sum(r['keys'] for r in rows) / len(rows),
                               evicted_total=rows[-1]['evicted'], max_pinned=max(r['pinned'] for r in rows),
                               max_rss_gib=max(r['rss_bytes'] for r in rows) / 2**30, capacity_gib=rows[-1]['capacity_gb'])
    return res


# ---------------------------------------------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group('policy')
    g.add_argument('--admission', choices=ADMISSION_OPTIONS, required=True, help='host-tier write admission')
    g.add_argument('--host-gib', type=float, required=True, help='host (CPU) tier capacity per TP rank, GiB; 0 = no host tier')
    g.add_argument('--use-connector', action='store_true', help="with --admission none: serve through the EfficientAgent connector")
    g.add_argument('--write-threshold', type=int, default=8, help='kappa: largest number of new full chunks a request may write under pressure')
    g.add_argument('--occupancy-threshold', type=float, default=0.95, help='theta: occupancy at which an evicting tier counts as full')
    g.add_argument('--window-s', type=float, default=60.0, help='task-activity and eviction window, seconds')
    g.add_argument('--prompt-window', type=int, default=256, help='first lookups averaged for the mean prompt length')
    g.add_argument('--telemetry-max-age-s', type=float, default=5.0, help='maximum age of a fresh telemetry report, seconds')
    g.add_argument('--chunk-tokens', type=int, default=1024, help='LMCache chunk size, tokens')
    g.add_argument('--kv-bytes-per-token', type=int, help='KV bytes per token per TP rank (default: from the model config.json)')
    g = ap.add_argument_group('server')
    g.add_argument('--model', required=True, help='model path or Hugging Face ID')
    g.add_argument('--served-model-name', help='name the server exposes (default: --model)')
    g.add_argument('--tp', type=int, default=1, help='tensor-parallel size')
    g.add_argument('--gpu-memory-utilization', type=float, help='vLLM GPU memory fraction')
    g.add_argument('--max-model-len', type=int, help='vLLM maximum model length')
    g.add_argument('--max-num-seqs', type=int, help='vLLM running-request cap')
    g.add_argument('--host', default='127.0.0.1'); g.add_argument('--port', type=int, default=8000)
    g.add_argument('--python', default=sys.executable, help='Python interpreter of the vLLM environment')
    g.add_argument('--tmpdir', type=Path, default=Path('/tmp/ea'), help='short TMPDIR for LMCache sockets')
    g.add_argument('--server-arg', action='append', help='extra vLLM argument (repeatable, e.g. --server-arg=--enforce-eager)')
    g.add_argument('--startup-timeout', type=float, default=1800, help='seconds to wait for /health')
    g.add_argument('--no-observer', action='store_true', help='do not load the read-only LMCache tier sampler')
    g = ap.add_argument_group('replay')
    g.add_argument('--trace', type=Path, required=True, help='trace directory (*.jsonl.gz)')
    g.add_argument('--out', type=Path, required=True, help='run directory (must be new or empty)')
    g.add_argument('--workers', type=int, default=16, help='active task slots (active pool)')
    g.add_argument('--limit', type=int, help='replay only the first N tasks')
    g.add_argument('--max-seconds', type=float, help='stop dispatching new steps after this many seconds')
    g.add_argument('--greedy-check', type=int, default=0, metavar='N', help='after the replay, record unforced greedy outputs for N tasks')
    g.add_argument('--wait-idle-gpus', type=float, default=0, metavar='S', help='wait up to S seconds for GPUs without compute processes')
    g.add_argument('--execute', action='store_true', help='start the server and replay (default: print the plan)')
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv); out = a.out.resolve()
    plan = build_plan(a); cmd, env = server_command(a, plan, out)
    spec = spec_of(a, plan)
    if not a.execute:
        print(json.dumps(dict(spec=spec, server=' '.join(shlex.quote(x) for x in cmd), env=recorded_env(env)), indent=1)); return 0
    if out.exists() and any(out.iterdir()): raise SystemExit(f'--out must be a new or empty directory: {out}')
    out.mkdir(parents=True, exist_ok=True); Path(a.tmpdir).mkdir(parents=True, exist_ok=True)
    base = f'http://{a.host}:{a.port}'
    stop = threading.Event()

    def on_signal(s, f): stop.set(); raise InterruptedError('signal %s' % s)
    for s in (signal.SIGTERM, signal.SIGINT): signal.signal(s, on_signal)
    res = dict(spec=spec, status='started', errors=[], started=time.time())
    srv = None; rec_stop = threading.Event()
    here = Path(__file__).resolve().parent
    try:
        if a.wait_idle_gpus > 0: res['gpus_idle'] = wait_for_idle_gpus(a.wait_idle_gpus)
        write_json(out / 'manifest.json', dict(server_command=cmd, env=recorded_env(env), spec=spec, sources_sha256={
            'kvtier/connector.py': sha256(here.parent / 'kvtier/connector.py'), 'kvtier/policies.py': sha256(here.parent / 'kvtier/policies.py'),
            'replay/client.py': sha256(here / 'client.py'), 'replay/forced_output.py': sha256(here / 'forced_output.py'),
            'replay/launch.py': sha256(__file__)}))
        with (out / 'server.log').open('w') as lf:
            srv = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf, stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(out))
        deadline = time.monotonic() + a.startup_timeout
        while True:
            if srv.poll() is not None: raise RuntimeError('server exited at startup with code %s' % srv.returncode)
            try: http_get(base, '/health', 3); break
            except Exception:
                if time.monotonic() > deadline: raise TimeoutError('server startup')
                time.sleep(5)
        text = (out / 'server.log').read_text(errors='replace')
        slots = [int(x.replace(',', '')) for x in re.findall(r'GPU KV cache size: ([\d,]+) tokens', text)]
        res['gpu_kv_tokens'] = slots[-1] if slots else None
        (out / 'metrics_start.prom').write_text(http_get(base, '/metrics'))
        threading.Thread(target=metrics_sampler, args=(base, out, rec_stop), daemon=True).start()
        tf = env.get('EA_KVTIER_TELEMETRY_FILE')
        if tf: threading.Thread(target=telemetry_history, args=(Path(tf), out / 'telemetry_history.jsonl', rec_stop), daemon=True).start()
        if a.max_seconds:
            tm = threading.Timer(a.max_seconds, stop.set); tm.daemon = True; tm.start()
        model = a.served_model_name or a.model
        res['replay'] = client.run(a.trace, out / 'replay', base, model, a.workers, 1.0, a.limit, stop=stop)
        (out / 'metrics_end.prom').write_text(http_get(base, '/metrics'))
        if a.greedy_check: res['greedy_check_requests'] = greedy_check(base, model, a.trace, a.greedy_check, out)
        if srv.poll() is not None: res['errors'].append('server exited during replay')
        rp = res['replay']
        res['status'] = 'passed' if rp['forced_mismatch'] == 0 and rp['errors'] == 0 and not res['errors'] else 'completed_with_issues'
    except BaseException as e:
        res['status'] = 'failed'; res['errors'].append(repr(e)); res['traceback'] = traceback.format_exc()
    finally:
        rec_stop.set()
        if srv is not None and srv.poll() is None:
            try: os.killpg(srv.pid, signal.SIGTERM); srv.wait(timeout=120)
            except subprocess.TimeoutExpired: os.killpg(srv.pid, signal.SIGKILL); srv.wait(timeout=30)
            except ProcessLookupError: pass
        from efficientagent.analysis import common, run_metrics
        res['kvtier_counters'] = common.run_counters(out)
        res['kvtier_first_hits'] = common.first_hits(out / 'server.log')
        tfile = out / 'kvtier_stats/telemetry_rank0.json'
        if tfile.exists():
            try: res['telemetry_last'] = json.loads(tfile.read_text())
            except (OSError, ValueError): pass
        res['lmcache_observer'] = observer_summary(out)
        res['finished'] = time.time()
        write_json(out / 'run.json', res)
        try: run_metrics.write(out)
        except Exception as e: append_jsonl(out / 'launcher_events.jsonl', dict(t=time.time(), msg='run_metrics failed: ' + repr(e)))
    return 0 if res['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
