"""vLLM v1 logits processor that forces a request's output to a recorded token sequence.

Enable it on the server with::

    --logits-processors efficientagent.replay.forced_output:ForcedOutputLogitsProcessor

A request opts in through ``vllm_xargs={'ea_forced_output': '<comma-separated token ids>'}``; requests without it are
untouched. At output position n the row's logits become -inf except for the recorded token f[n], so greedy sampling
emits f[n]. After len(f) tokens the row is left untouched; the replay client sets ``max_tokens=len(f)`` and
``ignore_eos=True``, so generation stops exactly there. Each step costs one masked fill on the forced rows.
"""
from __future__ import annotations

import torch
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor, MoveDirectionality

KEY = 'ea_forced_output'


def parse(params) -> list[int] | None:
    """Recorded output token IDs of a request, or None when the request does not opt in."""
    extra = getattr(params, 'extra_args', None) or {}
    raw = extra.get(KEY)
    if raw is None: return None
    if isinstance(raw, str): return [int(x) for x in raw.split(',') if x]
    return [int(x) for x in raw]


class ForcedOutputLogitsProcessor(LogitsProcessor):
    """Constrains opted-in requests to their recorded output tokens."""

    @classmethod
    def validate_params(cls, sampling_params):
        try: parse(sampling_params)
        except (TypeError, ValueError) as e: raise ValueError('invalid ea_forced_output: %s' % e)

    def __init__(self, vllm_config, device, is_pin_memory):
        self.device = device; self.rows = {}  # batch index -> (forced list, live output list)

    def is_argmax_invariant(self) -> bool: return False

    def update_state(self, batch_update: BatchUpdate | None):
        if not batch_update: return
        for i in batch_update.removed: self.rows.pop(i, None)
        for i, params, _prompt, out in batch_update.added:
            f = parse(params) if params is not None else None
            if f: self.rows[i] = (f, out)
            else: self.rows.pop(i, None)
        for a, b, direction in batch_update.moved:
            ra, rb = self.rows.pop(a, None), self.rows.pop(b, None)
            if ra is not None: self.rows[b] = ra
            if direction == MoveDirectionality.SWAP and rb is not None: self.rows[a] = rb

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.rows: return logits
        idx, tok = [], []
        for i, (f, out) in self.rows.items():
            n = len(out)
            if n < len(f) and i < logits.shape[0]: idx.append(i); tok.append(f[n])
        if not idx: return logits
        r = torch.tensor(idx, device=logits.device, dtype=torch.long); t = torch.tensor(tok, device=logits.device, dtype=torch.long)
        logits[r] = float('-inf'); logits[r, t] = 0.0
        return logits
