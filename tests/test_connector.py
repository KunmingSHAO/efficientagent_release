"""CPU tests of the write-admission connector: request filter, pressure rule, deduplication and telemetry."""
import json
import threading
import time
from collections import deque
from types import SimpleNamespace as N

import pytest

GIB = 2 ** 30


def make(mod, admission, host_gib=5.0):
    """Scheduler-side connector instance without a server (the constructor needs a live vLLM config)."""
    mod.ADMISSION = admission
    c = object.__new__(mod.EfficientAgentConnector)
    c._ea_stats = mod.Stats('scheduler'); c._ea_decided = {}
    c._ea_tasks = {}; c._ea_lens = deque(maxlen=mod.PROMPT_WINDOW); c._ea_telemetry = (0.0, None)
    c._ea_host_bytes = host_gib * GIB; c._lmcache_engine = N(load_specs={})
    return c


def request(c, rid, n_tokens, task='t0', hit=None):
    if hit is not None:
        c._lmcache_engine.load_specs[rid] = N(lmcache_cached_tokens=hit)
    return N(request_id=rid, num_tokens=n_tokens,
             sampling_params=N(extra_args={'kv_transfer_params': {'efficientagent.task_id': task}}))


def skipped(r):
    return bool(((r.sampling_params.extra_args or {}).get('kv_transfer_params') or {}).get('lmcache.skip_save'))


def telemetry(mod, evicted, registered, capacity, age=0.0):
    with open(mod.TELEMETRY_FILE, 'w') as f:
        json.dump(dict(t=time.time() - age, evicted_in_window=evicted, registered=registered, capacity=capacity), f)


def decide(mod, c, r):
    c._ea_telemetry = (0.0, None)   # force a telemetry re-read
    c.get_num_new_matched_tokens(r, 0)
    return skipped(r)


def test_defaults_match_paper_parameters(connector):
    m = connector
    assert (m.KAPPA, m.THETA, m.WINDOW_S, m.PROMPT_WINDOW) == (8, 0.95, 60.0, 256)
    assert (m.TELEMETRY_INTERVAL_S, m.TELEMETRY_READ_S, m.TELEMETRY_MAX_AGE_S, m.CHUNK) == (1.0, 0.5, 5.0, 1024)
    assert m.capacity_chunks(5.0) == 5 * GIB // (m.CHUNK * m.BYTES_PER_TOKEN)


def test_none_makes_no_decisions(connector):
    c = make(connector, 'none')
    r = request(c, 'a', 40 * 1024, hit=0)
    assert not decide(connector, c, r) and c._ea_stats.c['decisions'] == 0


def test_fixed_filter(connector):
    c = make(connector, 'fixed'); b = connector.CHUNK
    save = request(c, 'save', 20 * b + 5, hit=12 * b)      # u = 8 -> saved
    skip = request(c, 'skip', 20 * b + 5, hit=11 * b)      # u = 9 -> skipped
    short = request(c, 'short', 3 * b)                     # no host match: u = 3 -> saved
    assert [decide(connector, c, r) for r in (save, skip, short)] == [False, True, False]
    decide(connector, c, request(c, 'skip', 20 * b + 5))   # later lookups keep the first decision
    s = c._ea_stats.c
    assert (s['decisions'], s['skipped_requests'], s['saved_requests']) == (3, 1, 2)
    assert (s['skipped_new_chunks'], s['saved_new_chunks']) == (9, 11)
    assert 'SKIP_WRITE' in c._ea_stats.first


def test_fixed_threshold_is_configurable(connector, monkeypatch):
    monkeypatch.setattr(connector, 'KAPPA', 2)
    c = make(connector, 'fixed'); b = connector.CHUNK
    assert decide(connector, c, request(c, 'x', 3 * b, hit=0))


def load_pool(mod, c, tasks, tokens, prefix='p'):
    """First lookups of ``tasks`` distinct tasks with ``tokens`` prompt tokens each; returns the skip decisions."""
    return [decide(mod, c, request(c, f'{prefix}{i}', tokens, task=f'task{i}', hit=0)) for i in range(tasks)]


def estimate_bytes(mod, tasks, tokens):
    return (tasks - 1) * tokens * mod.BYTES_PER_TOKEN


def test_conditioned_requires_estimate_and_full_evicting_tier(connector):
    m = connector; cap = m.capacity_chunks(5.0); tokens = 30 * m.CHUNK
    assert estimate_bytes(m, 16, tokens) > 5 * GIB
    c = make(m, 'conditioned', 5.0)
    telemetry(m, evicted=5, registered=cap, capacity=cap)          # full and evicting
    out = load_pool(m, c, 16, tokens)
    first_above = next(i for i in range(1, 17) if estimate_bytes(m, i, tokens) > 5 * GIB)
    assert out == [i + 1 >= first_above for i in range(16)]
    s = c._ea_stats.c
    assert s['pressure_requests'] == s['estimate_above_capacity'] == s['skipped_requests'] == 16 - first_above + 1
    assert s['telemetry_full_evicting'] == 16 and s['telemetry_stale'] == 0
    assert {'PRESSURE_ON', 'PRESSURE_OFF', 'SKIP_WRITE'} <= c._ea_stats.first


def test_conditioned_tier_holding_working_set_stays_unrestricted(connector):
    m = connector; cap = m.capacity_chunks(40.0)
    c = make(m, 'conditioned', 40.0)
    telemetry(m, evicted=50, registered=cap, capacity=cap)         # full and evicting, but the estimate fits
    assert not any(load_pool(m, c, 16, 30 * m.CHUNK))
    s = c._ea_stats.c
    assert s['pressure_requests'] == 0 and s['telemetry_full_evicting'] == 16 and s['estimate_above_capacity'] == 0


@pytest.mark.parametrize('evicted,occupancy', [(0, 1.0), (5, 0.9)])
def test_conditioned_needs_evictions_at_high_occupancy(connector, evicted, occupancy):
    m = connector; cap = m.capacity_chunks(5.0)
    c = make(m, 'conditioned', 5.0)
    telemetry(m, evicted=evicted, registered=int(occupancy * cap), capacity=cap)
    assert not any(load_pool(m, c, 16, 30 * m.CHUNK))
    assert c._ea_stats.c['estimate_above_capacity'] > 0 and c._ea_stats.c['pressure_requests'] == 0


def test_conditioned_estimate_alone_without_fresh_report(connector):
    m = connector; tokens = 30 * m.CHUNK
    c = make(m, 'conditioned', 5.0)
    telemetry(m, evicted=0, registered=0, capacity=1, age=10 * m.TELEMETRY_MAX_AGE_S)   # stale report
    out = load_pool(m, c, 16, tokens)
    assert any(out) and c._ea_stats.c['telemetry_stale'] == 16
    assert c._ea_stats.c['pressure_requests'] == c._ea_stats.c['estimate_above_capacity']
    import os
    os.remove(m.TELEMETRY_FILE)                                                        # no report at all
    assert decide(m, c, request(c, 'late', tokens, task='task3', hit=0))


def test_conditioned_small_uncached_extension_is_saved(connector):
    m = connector; cap = m.capacity_chunks(5.0); b = m.CHUNK
    c = make(m, 'conditioned', 5.0)
    telemetry(m, evicted=5, registered=cap, capacity=cap)
    load_pool(m, c, 16, 30 * b)
    assert not decide(m, c, request(c, 'ext', 30 * b, task='task1', hit=22 * b))     # u = 8 under pressure -> saved
    assert decide(m, c, request(c, 'refill', 30 * b, task='task2', hit=21 * b))      # u = 9 under pressure -> skipped


def test_task_window_expires_inactive_tasks(connector):
    m = connector; tokens = 30 * m.CHUNK
    c = make(m, 'conditioned', 5.0)
    telemetry(m, evicted=0, registered=0, capacity=1, age=10 * m.TELEMETRY_MAX_AGE_S)
    old = time.time() - 2 * m.WINDOW_S
    c._ea_tasks = {f'old{i}': old for i in range(32)}
    c._ea_lens.extend([tokens] * 8)
    assert not decide(m, c, request(c, 'new', tokens, task='fresh', hit=0))            # only one active task
    assert set(c._ea_tasks) == {'fresh'}


def test_dedup_stores_only_missing_runs(connector):
    torch = pytest.importorskip('torch')
    b = 1024

    class TokenDB:
        def process_tokens(self, tokens=None, mask=None, request_configs=None, **k):
            n = len(tokens); first = int((~mask).sum())
            for s in range(first, n - n % b if n % b else n, b): yield s, s + b, ('key', tuple(tokens[:s + b]))

    calls = []; present = set(); tokens = list(range(6 * b)); td = TokenDB()
    for s, e, k in td.process_tokens(tokens=tokens, mask=torch.ones(len(tokens), dtype=torch.bool)):
        if s // b in (1, 2, 4): present.add(k)
    eng = N(store=lambda t, h, o, m, **kw: calls.append((len(t), int((~m).sum()), int(m.sum()), len(kw['slot_mapping']), kw.get('offset'))),
            token_database=td, storage_manager=N(contains=lambda k: 'LocalCPUBackend' if k in present else None))
    st = connector.Stats('worker'); connector.install_dedup(eng, st)
    mask = torch.ones(len(tokens), dtype=torch.bool); mask[:b] = False
    eng.store(tokens, mask=mask, slot_mapping=torch.arange(len(tokens)), offset=b, kvcaches=None)
    assert calls == [(4 * b, 3 * b, b, 4 * b, 3 * b), (6 * b, 5 * b, b, 6 * b, 5 * b)]    # runs [3], [5]
    assert (st.c['dedup_skipped_chunks'], st.c['store_chunks_considered']) == (3, 5)
    calls.clear(); present.clear(); eng.store(tokens, mask=mask, slot_mapping=torch.arange(len(tokens)))
    assert len(calls) == 1                                                            # nothing present: one native call


def test_telemetry_publisher_counts_capacity_evictions(connector, monkeypatch):
    m = connector
    monkeypatch.setattr(m, 'TELEMETRY_INTERVAL_S', 0.05)
    be = N(hot_cache={i: 0 for i in range(40)}, config=N(max_local_cpu_size=5.0),
           batched_remove=lambda keys, force=True: len(keys))
    st = m.Stats('worker'); m.install_telemetry_publisher(N(storage_manager=N(local_cpu_backend=be)), st)
    be.batched_remove([1, 2, 3], force=False); be.batched_remove([4], force=True)
    deadline = time.time() + 5; row = None
    while time.time() < deadline:
        try:
            row = json.loads(open(m.TELEMETRY_FILE).read())
            if row['evicted_in_window'] == 3: break
        except (OSError, ValueError): pass
        time.sleep(0.05)
    assert row and (row['evicted_in_window'], row['registered'], row['capacity']) == (3, 40, m.capacity_chunks(5.0))
    assert st.c['cpu_evicted_chunks'] == 3


def test_stats_flush(connector, tmp_path):
    st = connector.Stats('scheduler'); st.hit('X', decisions=2); st.flush()
    data = json.loads(next((tmp_path / 'stats').glob('scheduler-*.json')).read_text())
    assert data['counters'] == {'decisions': 2} and data['first_hits'] == ['X'] and data['role'] == 'scheduler'


def test_skip_save_is_not_part_of_the_cache_key():
    pytest.importorskip('lmcache')
    torch = pytest.importorskip('torch')
    from lmcache.utils import CacheEngineKey
    k1 = CacheEngineKey('vllm', 'm', 8, 0, 123, torch.bfloat16, {'lmcache.skip_save': True})
    k2 = CacheEngineKey('vllm', 'm', 8, 0, 123, torch.bfloat16, None)
    assert k1 == k2 and hash(k1) == hash(k2)
