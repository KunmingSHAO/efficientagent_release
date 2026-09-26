"""Workload profile and prefix-reuse structure of a replay trace (CPU only, token IDs only).

Profile: calls per task, prompt and output tokens, the longest prompt, and two shared-prefix sums over all calls:
the longest common prefix (on token IDs) of each prompt with the same task's previous prompt, and with that previous
prompt followed by its recorded output (the cache-stable prompt length). A task's first call counts as unshared.

Reuse of newly written KV: for each call, the new KV is its prompt suffix beyond the shared prefix plus its output.
The command reports how much of it the task's next call (or any later call of the task) uses as a prefix, and the
remainder that no later call uses, split by turn position, suffix length, and whether the next prompt diverges inside
the current prompt (a context edit).

Example::

    python -m efficientagent.analysis.trace_profile --trace TRACE --out profile.json
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np

from efficientagent.analysis import common as C
from efficientagent.analysis.runsets import write_json

SUFFIX_BINS = [(0, 256), (256, 1024), (1024, 4096), (4096, 16384), (16384, 10 ** 9)]


def profile(tasks: list) -> dict:
    calls = [len(t['calls']) for t in tasks]
    ptok = [sum(c['plen'] for c in t['calls']) for t in tasks]
    otok = [sum(c['olen'] for c in t['calls']) for t in tasks]
    shared_prompt = shared_processed = 0
    for t in tasks:
        prev = None
        for c in t['calls']:
            if prev is not None:
                shared_prompt += C.lcp(c['prompt'], prev['prompt'])
                shared_processed += C.lcp(c['prompt'], np.concatenate([prev['prompt'], prev['out']]))
            prev = c
    return dict(tasks=len(tasks), calls=sum(calls), prompt_tokens=sum(ptok), output_tokens=sum(otok),
                calls_per_task_mean=statistics.mean(calls), calls_per_task_median=statistics.median(calls),
                calls_per_task_min=min(calls), calls_per_task_max=max(calls),
                prompt_tokens_per_task_mean=statistics.mean(ptok), output_tokens_per_task_mean=statistics.mean(otok),
                max_prompt_tokens=max(c['plen'] for t in tasks for c in t['calls']),
                prompt_tokens_shared_with_previous_prompt=shared_prompt,
                prompt_tokens_shared_with_previous_processed=shared_processed)


def reuse_rows(tasks: list) -> list[dict]:
    rows = []
    for t in tasks:
        cs = t['calls']; n = len(cs)
        full = [np.concatenate([c['prompt'], c['out']]) for c in cs]
        for i, c in enumerate(cs):
            l_next = C.lcp(full[i], cs[i + 1]['prompt']) if i + 1 < n else 0
            l_any = max([C.lcp(full[i], cs[j]['prompt']) for j in range(i + 1, n)] or [0])

            def split(L):  # reused part of the new region [shared, plen + olen)
                rp = max(0, min(L, c['plen']) - c['shared']); ro = max(0, L - c['plen']) if L > c['plen'] else 0
                return rp, ro
            rp_n, ro_n = split(l_next); rp_a, ro_a = split(l_any)
            edited = i + 1 < n and C.lcp(c['prompt'], cs[i + 1]['prompt']) < c['plen']
            rows.append(dict(inst=t['inst'], seq=c['seq'], i=i, n=n, plen=c['plen'], shared=c['shared'], new_p=c['plen'] - c['shared'],
                             new_o=c['olen'], reuse_next_p=rp_n, reuse_next_o=ro_n, reuse_any_p=rp_a, reuse_any_o=ro_a, edited=bool(edited)))
    return rows


def summarize_reuse(rows: list[dict]) -> dict:
    def agg(sel):
        sel = list(sel)
        new = sum(r['new_p'] + r['new_o'] for r in sel)
        ra = sum(r['reuse_any_p'] + r['reuse_any_o'] for r in sel); rn = sum(r['reuse_next_p'] + r['reuse_next_o'] for r in sel)
        return dict(calls=len(sel), new_tokens=new, new_prompt=sum(r['new_p'] for r in sel), new_output=sum(r['new_o'] for r in sel),
                    reused_next=rn, reused_any=ra, reused_output_any=sum(r['reuse_any_o'] for r in sel), dead=new - ra,
                    dead_frac=(new - ra) / new if new else None)
    out = dict(all=agg(rows))
    out['position'] = {
        'first': agg(r for r in rows if r['i'] == 0),
        'middle': agg(r for r in rows if 0 < r['i'] < r['n'] - 3),
        'last3_not_last': agg(r for r in rows if r['n'] - 3 <= r['i'] < r['n'] - 1 and r['i'] > 0),
        'last': agg(r for r in rows if r['i'] == r['n'] - 1)}
    out['suffix_len'] = {f'{a}-{b}': agg(r for r in rows if a <= r['new_p'] < b) for a, b in SUFFIX_BINS}
    out['context_edit'] = {'next_diverges_inside_prompt': agg(r for r in rows if r['edited']),
                           'next_extends_prompt': agg(r for r in rows if r['i'] < r['n'] - 1 and not r['edited']),
                           'last_call': agg(r for r in rows if r['i'] == r['n'] - 1)}
    last = sum(r['new_p'] + r['new_o'] for r in rows if r['i'] == r['n'] - 1)
    edit = sum(r['new_p'] + r['new_o'] - r['reuse_any_p'] - r['reuse_any_o'] for r in rows if r['edited'])
    outd = sum(r['new_o'] - r['reuse_any_o'] for r in rows if r['i'] < r['n'] - 1 and not r['edited'])
    out['dead_decomposition'] = dict(last_call=last, context_edit=edit, output_not_reused_on_extend=outd,
                                     other=out['all']['dead'] - last - edit - outd)
    out['context_edit_events'] = sum(r['edited'] for r in rows)
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trace', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    tasks = C.load_trace(a.trace)
    res = dict(schema='efficientagent.trace_profile', version=1, profile=profile(tasks), new_kv_reuse=summarize_reuse(reuse_rows(tasks)))
    write_json(a.out, res)
    return res


if __name__ == '__main__':
    main()
