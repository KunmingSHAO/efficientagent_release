"""Descriptive admission options and the serving configuration they map to.

Three options are exposed:

  none         offload without write admission (native LMCache path, or the EfficientAgent connector with
               admission disabled when ``use_connector=True``);
  fixed        fixed write admission (request filter on every request, plus copy-time deduplication);
  conditioned  capacity-conditioned admission (request filter under pressure, plus copy-time deduplication).

A host capacity of 0 GiB with option ``none`` means no host tier (GPU prefix caching and recomputation only).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ADMISSION_OPTIONS = ('none', 'fixed', 'conditioned')
CONNECTOR_CLASS = 'EfficientAgentConnector'
CONNECTOR_MODULE = 'efficientagent.kvtier.connector'
NATIVE_CONNECTOR = 'LMCacheConnectorV1'
DESCRIPTIONS = {
    'none': 'offload without write admission',
    'fixed': 'fixed write admission',
    'conditioned': 'capacity-conditioned admission',
}


@dataclass
class AdmissionConfig:
    """Parameters of the write-admission rule (defaults as in the paper's appendix)."""
    write_threshold: int = 8               # kappa, full chunks
    occupancy_threshold: float = 0.95      # theta
    window_s: float = 60.0                 # task-activity and eviction window
    prompt_window: int = 256               # first lookups averaged for the mean prompt length
    telemetry_interval_s: float = 1.0      # telemetry publication interval
    telemetry_read_s: float = 0.5          # scheduler telemetry re-read interval
    telemetry_max_age_s: float = 5.0       # maximum age of a fresh report
    chunk_tokens: int = 1024               # LMCache chunk size b
    bytes_per_token: int | None = None     # beta, KV bytes per token per rank (required for 'conditioned')

    def env(self) -> dict[str, str]:
        out = {
            'EA_KVTIER_WRITE_THRESHOLD': str(self.write_threshold),
            'EA_KVTIER_OCCUPANCY_THRESHOLD': repr(float(self.occupancy_threshold)),
            'EA_KVTIER_WINDOW_S': repr(float(self.window_s)),
            'EA_KVTIER_PROMPT_WINDOW': str(self.prompt_window),
            'EA_KVTIER_TELEMETRY_INTERVAL_S': repr(float(self.telemetry_interval_s)),
            'EA_KVTIER_TELEMETRY_READ_S': repr(float(self.telemetry_read_s)),
            'EA_KVTIER_TELEMETRY_MAX_AGE_S': repr(float(self.telemetry_max_age_s)),
        }
        if self.bytes_per_token is not None:
            out['EA_KVTIER_BYTES_PER_TOKEN'] = str(int(self.bytes_per_token))
        return out


@dataclass
class TierPlan:
    """Server-side settings for one admission option and host capacity."""
    admission: str
    host_gib: float
    tp: int = 1
    use_connector: bool = False
    params: AdmissionConfig = field(default_factory=AdmissionConfig)

    def __post_init__(self):
        if self.admission not in ADMISSION_OPTIONS:
            raise ValueError(f'admission must be one of {ADMISSION_OPTIONS}, got {self.admission!r}')
        if self.host_gib < 0:
            raise ValueError('host_gib must be >= 0')
        if self.host_gib == 0 and (self.admission != 'none' or self.use_connector):
            raise ValueError('write admission needs a host tier: use host_gib > 0')
        if self.admission == 'conditioned' and self.params.bytes_per_token is None:
            raise ValueError("'conditioned' needs bytes_per_token (KV bytes per token per rank)")

    @property
    def host_tier(self) -> bool:
        return self.host_gib > 0

    @property
    def native(self) -> bool:
        """True when the unmodified LMCache connector serves the host tier."""
        return self.host_tier and self.admission == 'none' and not self.use_connector

    def lmcache_env(self) -> dict[str, str]:
        if not self.host_tier:
            return {'VLLM_PLUGINS': ''}
        return {'VLLM_PLUGINS': 'lmcache.vllm_plugin', 'LMCACHE_LOCAL_CPU': 'True',
                'LMCACHE_CHUNK_SIZE': str(self.params.chunk_tokens),
                'LMCACHE_MAX_LOCAL_CPU_SIZE': f'{float(self.host_gib):.1f}',
                'LMCACHE_PRE_CACHING_HASH_ALGORITHM': 'sha256_cbor'}

    def server_args(self) -> list[str]:
        """vLLM arguments that select the host tier and connector."""
        if not self.host_tier:
            return []
        if self.native:
            return ['--kv-offloading-backend', 'lmcache', '--kv-offloading-size', f'{self.tp * float(self.host_gib):.1f}',
                    '--kv-transfer-config', json.dumps(dict(kv_connector=NATIVE_CONNECTOR, kv_role='kv_both'))]
        cfg = dict(kv_connector=CONNECTOR_CLASS, kv_connector_module_path=CONNECTOR_MODULE, kv_role='kv_both',
                   kv_connector_extra_config={'lmcache.local_cpu': True, 'lmcache.max_local_cpu_size': float(self.host_gib)})
        return ['--kv-transfer-config', json.dumps(cfg)]

    def connector_env(self, stats_dir: str | Path | None = None, telemetry_file: str | Path | None = None) -> dict[str, str]:
        """Environment for the EfficientAgent connector (empty on the native path or without a host tier)."""
        if not self.host_tier or self.native:
            return {}
        env = {'EA_KVTIER_ADMISSION': self.admission, **self.params.env()}
        if stats_dir is not None:
            env['EA_KVTIER_STATS_DIR'] = str(stats_dir)
        if self.admission == 'conditioned' and telemetry_file is not None:
            env['EA_KVTIER_TELEMETRY_FILE'] = str(telemetry_file)
        return env


def kv_bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int, tp: int = 1, dtype_bytes: int = 2) -> int:
    """KV bytes per token per tensor-parallel rank: K and V for every layer, with KV heads replicated when tp > heads."""
    heads_per_rank = -(-num_kv_heads // tp)
    return 2 * num_layers * heads_per_rank * head_dim * dtype_bytes


def kv_bytes_per_token_from_config(config_path: str | Path, tp: int = 1, dtype_bytes: int = 2) -> int:
    """Read a Hugging Face ``config.json`` (or a model directory containing one) and apply ``kv_bytes_per_token``."""
    p = Path(config_path)
    if p.is_dir():
        p = p / 'config.json'
    c = json.loads(p.read_text())
    c = c.get('text_config', c)
    heads = c.get('num_key_value_heads') or c['num_attention_heads']
    head_dim = c.get('head_dim') or c['hidden_size'] // c['num_attention_heads']
    return kv_bytes_per_token(c['num_hidden_layers'], heads, head_dim, tp, dtype_bytes)
