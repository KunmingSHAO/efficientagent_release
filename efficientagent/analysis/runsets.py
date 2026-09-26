"""Run arguments for the analysis command lines: ``RUN_DIR`` or ``RUN_DIR@HOST_GIB``, resolved against ``run.json``."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from efficientagent.analysis import common as C

DEFAULT_BYTES_PER_TOKEN = 24576   # BF16 KV bytes per token per rank of Qwen3-Coder-30B-A3B-Instruct at TP8


@dataclass
class RunInfo:
    """One replay run and the settings the analyses need."""
    name: str
    dir: Path
    admission: str
    host_gib: float
    workers: int
    ranks: int
    write_threshold: int
    bytes_per_token: int
    chunk: int

    @property
    def capacity_chunks(self) -> int:
        return C.chunks_of(self.host_gib, self.bytes_per_token, self.chunk)


def fmt_gib(x: float) -> str:
    """Key used for capacities in outputs ('5', '2.5', '0.5')."""
    return '%g' % x


def resolve_run(arg: str, bytes_per_token: int | None = None, chunk: int | None = None) -> RunInfo:
    """Parse ``DIR`` or ``DIR@GIB``; the remaining settings come from ``DIR/run.json`` (``spec``)."""
    path, _, gib = str(arg).partition('@')
    d = Path(path)
    spec = C.run_spec(d)
    summary = C.read_summary(d)
    host = float(gib) if gib else spec.get('host_gib')
    if host is None:
        raise ValueError(f'{d}: host capacity unknown; pass {d}@GIB or write spec.host_gib in run.json')
    workers = spec.get('workers') or summary.get('workers')
    if not workers:
        raise ValueError(f'{d}: active pool unknown (spec.workers or replay_summary.workers)')
    return RunInfo(name=d.name, dir=d, admission=spec.get('admission', 'none'), host_gib=float(host), workers=int(workers),
                   ranks=int(spec.get('tp') or 1), write_threshold=int(spec.get('write_threshold') or 8),
                   bytes_per_token=int(bytes_per_token or spec.get('bytes_per_token') or DEFAULT_BYTES_PER_TOKEN),
                   chunk=int(chunk or spec.get('chunk_tokens') or C.CHUNK))


def write_json(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=float) + '\n')


def makespan(run: RunInfo):
    return C.read_summary(run.dir).get('makespan_s')


def prefill_computed(run: RunInfo):
    """Computed prefill tokens: delta of ``vllm:request_prefill_kv_computed_tokens_sum`` over the replay."""
    o, f = C.read_prom(run.dir / 'metrics_start.prom'), C.read_prom(run.dir / 'metrics_end.prom')
    return C.prom_delta(o, f, 'vllm:request_prefill_kv_computed_tokens_sum')
