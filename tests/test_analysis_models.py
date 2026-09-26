"""CPU tests of the trace and admission analyses on tiny synthetic traces and runs (see tests/synthetic.py)."""
import json

import pytest

import synthetic as S
from efficientagent.analysis import admission_sensitivity, capacity_model, declined_reuse, pressure_estimate, trace_profile
from efficientagent.analysis import common as C


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp('synthetic')
    trace = S.make_trace(d / 'trace')
    runs = {}
    for name, adm, gib in (('offload', 'none', 0.25), ('fixed', 'fixed', 0.25), ('conditioned', 'conditioned', 0.25), ('recompute', 'none', 0),
                           ('offload_large', 'none', 4.0)):
        runs[name] = (d / name, S.make_run(d / name, trace, admission=adm, host_gib=gib))
    return d, trace, runs


def test_trace_profile_totals(tmp_path):
    trace = S.make_trace(tmp_path / 'trace', n_tasks=3, edit_task=1)
    res = trace_profile.main(['--trace', str(trace), '--out', str(tmp_path / 'p.json')])
    tasks = C.load_trace(trace)
    p = res['profile']
    assert p['calls'] == sum(len(t['calls']) for t in tasks)
    assert p['prompt_tokens'] == sum(c['plen'] for t in tasks for c in t['calls'])
    assert p['prompt_tokens_shared_with_previous_prompt'] == sum(c['shared'] for t in tasks for c in t['calls'])
    assert p['prompt_tokens_shared_with_previous_processed'] >= p['prompt_tokens_shared_with_previous_prompt']
    r = res['new_kv_reuse']
    assert r['context_edit_events'] == 1
    a = r['all']
    assert a['new_tokens'] == a['new_prompt'] + a['new_output'] and a['dead'] == a['new_tokens'] - a['reused_any']
    assert sum(r['dead_decomposition'].values()) == a['dead']
    assert sum(v['calls'] for v in r['position'].values()) == p['calls']
    assert json.loads((tmp_path / 'p.json').read_text())['profile'] == p


def test_capacity_curve_matches_simulated_tier(world, tmp_path):
    d, trace, runs = world
    run_dir, sim = runs['offload']
    big_dir, big = runs['offload_large']
    res = capacity_model.main(['curve', '--trace', str(trace), '--run', str(run_dir), '--run', str(big_dir), '--run', str(runs['recompute'][0]),
                               '--grid', '0.125', '0.25', '1', '--noqueue-scales', '0', '--out', str(tmp_path / 'c.json'), '--md', str(tmp_path / 'c.md')])
    r = res['runs']['offload']
    assert r['mapped'] == r['n'] == sum(1 for _ in C.read_requests(run_dir))
    assert 0 <= r['exact_hit_match_frac'] <= 1
    assert r['logged_retrieved_per_rank'] == sim['truth']['retrieved_per_rank']
    assert r['logged_stored_per_rank'] == sim['truth']['stored_per_rank']
    assert r['logged_prefill_computed'] == sim['truth']['prefill_computed']
    cv = [r['curve'][g]['cpu_prefix_hit_tokens'] for g in ('0.125', '0.25', '1')]
    assert cv == sorted(cv)                                    # LRU inclusion: coverage grows with capacity
    # a tier that never evicts: the stack-distance prediction reproduces every logged host hit and the restored volume
    rb = res['runs']['offload_large']
    assert big['truth']['evictions'] == 0 and rb['exact_hit_match_frac'] == 1.0
    assert rb['predicted_at_own']['retrieved'] == rb['logged_retrieved_per_rank'] == big['truth']['retrieved_per_rank']
    assert res['recompute_runs']['recompute']['logged_prefill_computed'] == runs['recompute'][1]['truth']['prefill_computed']
    assert set(res['timelines']) == {'offload', 'offload_large', 'noqueue_x0'}
    assert res['workload']['working_set_gib'] > 0 and (tmp_path / 'c.md').exists()
    with pytest.raises(SystemExit):
        capacity_model.main(['curve', '--trace', str(trace), '--run', str(run_dir), '--noqueue-scales', '1', '--out', str(tmp_path / 'x.json')])


def test_capacity_pool_is_seeded_and_ordered(world, tmp_path):
    d, trace, runs = world
    args = ['pool', '--trace', str(trace), '--run', str(runs['offload'][0]), '--run', str(runs['recompute'][0]), '--active-pool', '2',
            '--engine-slots', '1', '2', '--prefill-tok-s', '20000', '--decode-s-per-token', '0.01', '--host-gib', '0.125', '0.25', '0.5',
            '--seed', '3', '--capacity-samples', '200', '--validate']
    a = capacity_model.main(args + ['--out', str(tmp_path / 'a.json')])
    b = capacity_model.main(args + ['--out', str(tmp_path / 'b.json')])
    assert a['prediction'] == b['prediction']
    assert set(a['timelines']) == {'noqueue', 'queued_slots1', 'queued_slots2'}
    for g in ('0.125', '0.25', '0.5'):
        lo, hi = a['prediction'][g]['retrieved_per_rank']
        assert 0 <= lo <= hi
    assert a['prediction']['no_host_tier']['retrieved_per_rank'] == [0.0, 0.0]
    assert a['validation_reference_pool']['measured']['offload']['retrieved_per_rank'] == runs['offload'][1]['truth']['retrieved_per_rank']


def test_admission_sensitivity_reproduces_counters(world, tmp_path):
    d, trace, runs = world
    res = admission_sensitivity.main(['--run', str(runs['fixed'][0]), '--run', str(runs['conditioned'][0]), '--max-kappa', '16',
                                      '--out', str(tmp_path / 's.json')])
    f = res['kappa']['fixed']; truth = runs['fixed'][1]['truth']['counters']
    assert f['validated_against_counters'] is True
    k = f['curve'][f['write_threshold']]
    assert (k['selected_requests'], k['declined_chunks']) == (truth['skipped_requests'], truth['skipped_new_chunks'])
    assert [r['selected_requests'] for r in f['curve']] == sorted((r['selected_requests'] for r in f['curve']), reverse=True)
    th = res['theta']['conditioned']
    hist = C.read_jsonl(runs['conditioned'][0] / 'telemetry_history.jsonl')
    assert th['evicting_snapshots'] == sum(h['evicted_in_window'] > 0 for h in hist) == len(th['evicting_occupancies'])
    # a counter that the logged decisions cannot reproduce is reported as an error
    p = runs['fixed'][0] / 'kvtier_stats/scheduler-1.json'; orig = p.read_text()
    x = json.loads(orig); x['counters']['skipped_requests'] += 1; p.write_text(json.dumps(x))
    try:
        with pytest.raises(ValueError):
            admission_sensitivity.main(['--run', str(runs['fixed'][0]), '--out', str(tmp_path / 't.json')])
    finally:
        p.write_text(orig)


def test_declined_reuse_classification(world, tmp_path):
    d, trace, runs = world
    res = declined_reuse.main(['--trace', str(trace), '--run', str(runs['fixed'][0]), '--run', str(runs['conditioned'][0]),
                               '--run', str(runs['offload'][0]), '--run', str(runs['recompute'][0]), '--compare-gib', '0.25', '1',
                               '--processes', '2', '--out', str(tmp_path / 'd.json'), '--md', str(tmp_path / 'd.md')])
    assert set(res['admission']) == {'fixed', 'conditioned'} and set(res['offload']) == {'offload'}
    for name in ('fixed', 'conditioned'):
        r = res['admission'][name]; truth = runs[name][1]['truth']
        assert r['counter_check_exact'] is True
        assert r['declined_requests'] == len(truth['declined'])
        s = r['by_order']['sched_lookup']
        dcl = s['declined']
        assert dcl['never'] + dcl['ge_K'] + dcl['lt_K'] == dcl['chunks'] == r['declined_chunks']
        assert s['lru_replay']['exact_hit_frac'] == 1.0        # the recorded stream with its declines reproduces the host hits
    off = res['offload']['offload']['by_order']['sched_lookup']
    assert off['lru_replay']['exact_hit_frac'] == 1.0 and 0 <= off['sd_prefix_hit_exact_frac'] <= 1
    assert res['trace']['calls'] == sum(len(t['calls']) for t in C.load_trace(trace))
    assert (tmp_path / 'd.md').read_text().startswith('# Reuse distance of declined chunks')


def test_pressure_estimate_shares(world, tmp_path):
    d, trace, runs = world
    res = pressure_estimate.main(['--run', str(runs['offload'][0]), '--run', str(runs['conditioned'][0]), '--host-gib', '0.125', '0.25', '1',
                                  '--out', str(tmp_path / 'p.json')])
    for name in ('offload', 'conditioned'):
        r = res['runs'][name]
        shares = [r['estimate_above'][g]['share_of_requests'] for g in ('0.125', '0.25', '1')]
        assert shares == sorted(shares, reverse=True) and all(0 <= x <= 1 for x in shares)
        t = r['telemetry']
        assert t['pressure_share_of_requests'] <= min(t['full_evicting_share_of_requests'], t['estimate_above_own_share_of_requests'])
        assert len(r['telemetry_ranks']) == 2
