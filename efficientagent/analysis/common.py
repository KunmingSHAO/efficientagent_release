"""Shared readers for replay traces and run directories, LMCache log parsing, and stack distances.

Run directory layout (written by ``efficientagent.replay.launch``)::

    run.json                    spec (admission, host_gib, workers, ...), status and client summary
    manifest.json               server command, environment, source checksums
    server.log                  vLLM + LMCache log
    metrics_start.prom          Prometheus snapshot when the replay starts
    metrics_end.prom            Prometheus snapshot when the replay ends
    load_samples.jsonl          scheduler gauges every 2 s
    replay/requests.jsonl       one row per replayed request
    replay/tasks.jsonl          one row per task
    replay/replay_summary.json  client summary
    kvtier_stats/*.json         connector counters per process (plus the telemetry file of rank 0)
    telemetry_history.jsonl     5-s copies of the rank-0 host-tier telemetry (conditioned admission)
    lmcache_observer/*.jsonl    10-s host-tier samples per rank (resident keys, evictions)
"""
from __future__ import annotations

import collections
import datetime
import gzip
import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Iterable

import numpy as np

CHUNK = 1024          # LMCache chunk size in tokens (only full chunks are stored)
GIB = 2 ** 30


# ---------------------------------------------------------------------------------------------------- basic helpers
def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=float), q)) if len(a) else None


def dist(a: Iterable[float]) -> dict:
    a = list(a)
    if not a: return dict(n=0)
    return dict(n=len(a), mean=float(np.mean(a)), p50=pct(a, 50), p90=pct(a, 90), p99=pct(a, 99), max=float(max(a)), sum=float(sum(a)))


def read_jsonl(path) -> list[dict]:
    rows = []
    p = Path(path)
    if not p.exists(): return rows
    with open(p) as f:
        for line in f:
            try: rows.append(json.loads(line))
            except ValueError: continue
    return rows


def chunks_of(gib: float, bytes_per_token: int, chunk: int = CHUNK) -> int:
    """Chunk capacity K_H of a host tier of ``gib`` GiB per rank."""
    return int(gib * GIB // (chunk * bytes_per_token))


# ---------------------------------------------------------------------------------------------------- trace
def load_trace(trace_dir) -> list[dict]:
    """Read a replay trace directory (``*.jsonl.gz``, one task per file, sorted by file name).

    Returns tasks ``dict(inst, order, tail_gap, calls=[dict(seq, gap, shared, plen, olen, prompt, out)])`` with the exact
    prompt and output token IDs as int32 arrays.
    """
    tasks = []
    for f in sorted(Path(trace_dir).glob('*.jsonl.gz')):
        with gzip.open(f, 'rt') as z:
            lines = [json.loads(line) for line in z]
        head, steps = lines[0], lines[1:]
        prev = np.zeros(0, dtype=np.int32); calls = []
        for s in steps:
            p = np.concatenate([prev[:s['prompt_shared']], np.asarray(s['prompt_suffix'], dtype=np.int32)])
            if len(p) != s['prompt_len']:
                raise ValueError(f'{f.name}: prompt length mismatch at seq {s["seq"]}')
            calls.append(dict(seq=s['seq'], gap=s['gap_before_s'], shared=s['prompt_shared'], plen=s['prompt_len'],
                              olen=s['output_len'], prompt=p, out=np.asarray(s['output'], dtype=np.int32)))
            prev = p
        tasks.append(dict(inst=head['instance_id'], order=head['order'], tail_gap=head.get('tail_gap_s', 0.0), calls=calls))
    return tasks


def call_map(tasks: list[dict]) -> dict:
    return {(t['inst'], c['seq']): c for t in tasks for c in t['calls']}


def lcp(a, b) -> int:
    """Length of the longest common prefix of two token arrays."""
    if a is None or b is None: return 0
    n = min(len(a), len(b))
    if n == 0: return 0
    d = np.nonzero(a[:n] != b[:n])[0]
    return int(d[0]) if len(d) else n


def chunk_keys(seq, parent_keys=None, parent_valid: int = 0, chunk: int = CHUNK) -> list[bytes]:
    """Prefix-dependent keys of the full chunks of ``seq`` (each key hashes its chunk and the previous key).

    The first ``parent_valid`` keys are reused from ``parent_keys`` (the caller guarantees an identical prefix).
    """
    keys = list(parent_keys[:parent_valid]) if parent_keys is not None else []
    h = keys[-1] if keys else b''
    for c in range(len(keys), len(seq) // chunk):
        h = hashlib.blake2b(h + seq[c * chunk:(c + 1) * chunk].tobytes(), digest_size=12).digest()
        keys.append(h)
    return keys


def prompt_chunk_keys(tasks: list[dict], chunk: int = CHUNK) -> dict:
    """{(task, seq): prefix-dependent keys of the full prompt chunks}; also stored as ``call['keys']``."""
    keys = {}
    for t in tasks:
        pk = pfull = None
        for c in t['calls']:
            full = c['prompt']
            valid = (lcp(pfull, full) // chunk) if pfull is not None else 0
            c['keys'] = chunk_keys(full, pk, valid, chunk); pk, pfull = c['keys'], full
            keys[(t['inst'], c['seq'])] = c['keys']
    return keys


# ---------------------------------------------------------------------------------------------------- stack distances
class BIT:
    """Fenwick tree over reference positions (prefix sums of live 'last reference' markers)."""

    def __init__(self, n: int):
        self.n = n; self.t = [0] * (n + 1)

    def add(self, i: int, v: int) -> None:
        i += 1; t = self.t; n = self.n
        while i <= n: t[i] += v; i += i & -i

    def pre(self, i: int) -> int:
        """Sum over positions [0, i)."""
        r = 0; t = self.t
        while i > 0: r += t[i]; i -= i & -i
        return r


def stack_distances(order: list, keys: dict, tmap: dict, chunk: int = CHUNK) -> dict:
    """Single-pass LRU stack distances (Mattson et al.) of a reference stream.

    Each call in ``order`` references all full chunks of its prompt, tail first and head last (the head is most
    recent). Returns {call: array of per-chunk stack distances, head..tail}, ``inf`` for a first reference. A chunk is
    resident in an LRU tier of C chunks iff its stack distance is < C.
    """
    total = sum(tmap[c]['plen'] // chunk for c in order)
    bit = BIT(total + 1); last = {}; pos = 0; out = {}
    for call in order:
        k = keys[call][:tmap[call]['plen'] // chunk]; sd = np.full(len(k), np.inf)
        for c in reversed(range(len(k))):
            key = k[c]
            if key in last:
                p = last[key]; sd[c] = bit.pre(pos) - bit.pre(p + 1); bit.add(p, -1)
            bit.add(pos, 1); last[key] = pos; pos += 1
        out[call] = sd
    return out


def prefix_hit_chunks(sd, capacity_chunks: int) -> int:
    """Consecutive host coverage from the first chunk: a lookup stops at the first absent chunk."""
    if not len(sd): return 0
    return int(np.sum(np.maximum.accumulate(sd) < capacity_chunks))


# ---------------------------------------------------------------------------------------------------- LMCache log
TS = r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3})\] LMCache INFO:\S* '
R_LOOK = re.compile(TS + r'Reqid: (\S+), Total tokens (\d+), LMCache hit tokens: (\d+), need to load: (-?\d+)')
R_STORING = re.compile(TS + r'Storing KV cache for (\d+) out of (\d+) tokens \(skip_leading_tokens=(\d+)\) for request (\S+)')
R_STORED = re.compile(TS + r'Stored (\d+) out of total (\d+) tokens\. size: [\d.]+ gb, cost ([\d.]+) ms')
R_RETR = re.compile(TS + r'Retrieved (\d+) out of (\d+) required tokens \(from (\d+) total tokens\)\. size: [\d.]+ gb, cost ([\d.]+) ms')
R_FIRST_HIT = re.compile(r'EA_KVTIER first-hit (\w+)')


def ts(s: str) -> float:
    return datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S,%f').timestamp()


def parse_lmcache_log(path):
    """-> lookups (t, rid, total, hit, need), storing (t, rid, Y, S, X), stored (t, tokens, cost_ms), retrieved (t, R, T, cost_ms).

    Storing rows: the request stores X tokens [S, Y) of its first Y tokens, S = skip_leading_tokens.
    """
    look, storing, stored, retr = [], [], [], []
    with open(path, errors='replace') as f:
        for line in f:
            if 'LMCache INFO' not in line: continue
            for m in R_LOOK.finditer(line): look.append((ts(m[1]), m[2], int(m[3]), int(m[4]), int(m[5])))
            for m in R_STORING.finditer(line): storing.append((ts(m[1]), m[5], int(m[3]), int(m[4]), int(m[2])))
            for m in R_STORED.finditer(line): stored.append((ts(m[1]), int(m[2]), float(m[4])))
            for m in R_RETR.finditer(line): retr.append((ts(m[1]), int(m[2]), int(m[4]), float(m[5])))
    return look, storing, stored, retr


def group_events(items, key, tol: float = 0.25, size: int = 8):
    """Collapse per-rank duplicate log lines into events: same key, within ``tol`` seconds, up to ``size`` lines."""
    items = sorted(items, key=lambda x: x[0]); open_ = {}; events = []
    for it in items:
        k = key(it); ev = open_.get(k)
        if ev is not None and it[0] - ev['t0'] <= tol and len(ev['lines']) < size:
            ev['lines'].append(it)
        else:
            ev = dict(t0=it[0], lines=[it]); open_[k] = ev; events.append(ev)
    return events


def lmcache_log_totals(path, ranks: int) -> dict:
    """Stored / retrieved token totals from LMCache log lines (one line per rank per operation), and per-rank volumes."""
    out = dict(retrieved_lines=0, retrieved_tokens=0, stored_lines=0, stored_tokens=0)
    costs = dict(retrieved=[], stored=[])
    if not Path(path).exists(): return out
    ret = re.compile(r'Retrieved (\d+) out of (?:total )?(\d+)[^\n]*?cost ([\d.]+) ms')
    sto = re.compile(r'Stored (\d+) out of total (\d+) tokens[^\n]*?cost ([\d.]+) ms')
    with open(path, errors='replace') as f:
        for line in f:
            if 'LMCache' not in line: continue
            m = ret.search(line)
            if m: out['retrieved_lines'] += 1; out['retrieved_tokens'] += int(m.group(1)); costs['retrieved'].append(float(m.group(3)))
            m = sto.search(line)
            if m: out['stored_lines'] += 1; out['stored_tokens'] += int(m.group(1)); costs['stored'].append(float(m.group(3)))
    for k, v in costs.items():
        out[f'{k}_cost_ms_sum'] = sum(v); out[f'{k}_cost_ms_median'] = statistics.median(v) if v else None
    out['retrieved_per_rank'] = out['retrieved_tokens'] / ranks
    out['stored_per_rank'] = out['stored_tokens'] / ranks
    return out


def first_hits(path) -> list[str]:
    if not Path(path).exists(): return []
    return sorted(set(R_FIRST_HIT.findall(Path(path).read_text(errors='replace'))))


# ---------------------------------------------------------------------------------------------------- Prometheus
def read_prom(path) -> dict:
    """Sum every sample of each metric over its label sets; histogram buckets are kept per ``le``."""
    vals = {}
    if not Path(path).exists(): return vals
    for line in Path(path).read_text().splitlines():
        if not line or line[0] == '#': continue
        try: k, v = line.rsplit(' ', 1)
        except ValueError: continue
        stem = k.split('{')[0]
        try: x = float(v)
        except ValueError: continue
        if 'le="' in k:
            le = re.search(r'le="([^"]+)"', k).group(1); vals.setdefault(stem, {}); vals[stem][le] = vals[stem].get(le, 0) + x
        elif not isinstance(vals.get(stem), dict):
            vals[stem] = vals.get(stem, 0.0) + x
    return vals


def prom_delta(a: dict, b: dict, k: str):
    x, y = a.get(k, 0), b.get(k, 0)
    return (y - x) if not isinstance(x, dict) and not isinstance(y, dict) else None


def hist_quantile(buckets: dict, q: float):
    items = sorted(((float('inf') if le == '+Inf' else float(le), c) for le, c in buckets.items()))
    total = items[-1][1] if items else 0
    if not total: return None
    for le, c in items:
        if c >= q * total: return le
    return None


# ---------------------------------------------------------------------------------------------------- run directories
def read_run(run_dir) -> dict:
    p = Path(run_dir) / 'run.json'
    return json.loads(p.read_text()) if p.exists() else {}


def run_spec(run_dir) -> dict:
    return read_run(run_dir).get('spec') or {}


def read_summary(run_dir) -> dict:
    p = Path(run_dir) / 'replay/replay_summary.json'
    return json.loads(p.read_text()) if p.exists() else {}


def read_requests(run_dir) -> list[dict]:
    return read_jsonl(Path(run_dir) / 'replay/requests.jsonl')


def read_tasks(run_dir) -> list[dict]:
    return read_jsonl(Path(run_dir) / 'replay/tasks.jsonl')


def run_counters(run_dir) -> dict:
    """Connector counters summed over processes, keyed '<role>.<counter>'; non-counter JSON files are skipped."""
    tot = {}
    d = Path(run_dir) / 'kvtier_stats'
    for p in sorted(d.glob('*.json')) if d.exists() else []:
        try: x = json.loads(p.read_text())
        except (OSError, ValueError): continue
        if not (isinstance(x, dict) and 'role' in x and isinstance(x.get('counters'), dict)): continue
        for k, v in x['counters'].items(): tot[f'{x["role"]}.{k}'] = tot.get(f'{x["role"]}.{k}', 0) + v
    return tot


def map_lookups(run_dir, tmap: dict, look=None, storing=None) -> dict:
    """Map LMCache request IDs to trace calls and return each call's scheduling lookup.

    A request ID maps to the trace call whose client request has the same prompt length and whose send window
    contains the ID's first lookup (nearest to the send time). The scheduling lookup is the last lookup before the
    request's first store (or the last lookup). Returns dict(final={call: (t, hit, need)}, mapped, n, rid={call: rid}).
    """
    run_dir = Path(run_dir)
    req = read_requests(run_dir); req.sort(key=lambda r: r['start'])
    if look is None:
        look, storing, _, _ = parse_lmcache_log(run_dir / 'server.log')
    first = {}
    for t_, rid, tot, hit, need in look:
        first.setdefault(rid, (t_, tot))
    by_len = collections.defaultdict(list)
    for rid, (t_, tot) in first.items(): by_len[tot].append((t_, rid))
    used, rid2call = set(), {}
    for r in req:
        c = [(t_ - r['start'], rid) for t_, rid in by_len.get(r['prompt_len'], []) if rid not in used and r['start'] - 0.5 <= t_ <= r.get('end', r['start'])]
        if c:
            rid = min(c)[1]; used.add(rid); rid2call[rid] = (r['instance_id'], r['seq'])
    fs = {}
    for row in storing or []:
        fs.setdefault(row[1], row[0])
    lb = collections.defaultdict(list)
    for x in look: lb[x[1]].append(x)
    fl, rids = {}, {}
    for rid, call in rid2call.items():
        L = lb[rid]; cand = [x for x in L if rid not in fs or x[0] <= fs[rid] + 0.05]
        x = (cand or L)[-1]; fl[call] = (x[0], x[3], x[4]); rids[call] = rid
    return dict(final=fl, mapped=len(rid2call), n=len(req), rid=rids)
