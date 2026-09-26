"""CPU tests of the per-run metrics, the run summary and the run comparison on synthetic run directories."""
import json

import pytest

from efficientagent.analysis import compare_runs, run_metrics, run_summary
from synthetic import make_run, make_trace


@pytest.fixture
def runs(tmp_path):
    trace = make_trace(tmp_path / 'trace', n_tasks=6, n_steps=4, seed=2)
    out = {}
    for name, admission, gib in (('recompute', 'none', 0), ('offload', 'none', 0.25), ('fixed', 'fixed', 0.25), ('conditioned', 'conditioned', 0.25)):
        out[name] = (tmp_path / 'runs' / name, make_run(tmp_path / 'runs' / name, trace, admission=admission, host_gib=gib))
    return tmp_path / 'runs', out


def test_run_metrics_match_simulated_truth(runs):
    _, rs = runs
    for name, (d, sim) in rs.items():
        m = run_metrics.write(d)
        t = sim['truth']
        assert m['prefill_computed_tokens'] == t['prefill_computed']
        assert m['retrieved_per_rank'] == t['retrieved_per_rank'] and m['stored_per_rank'] == t['stored_per_rank']
        assert m['requests'] == sum(1 for _ in open(d / 'replay/requests.jsonl'))
        assert json.loads((d / 'run_metrics.json').read_text())['run'] == name


def test_run_summary(runs, tmp_path):
    root, rs = runs
    rows = run_summary.summarize([root])
    assert [r['policy'] for r in rows] == ['recompute', 'offload', 'fixed', 'conditioned']
    by = {r['run']: r for r in rows}
    for name in ('fixed', 'conditioned'):
        c = rs[name][1]['truth']['counters']
        assert by[name]['decisions'] == c['decisions'] and by[name]['skipped_requests'] == c['skipped_requests']
    assert by['offload']['decisions'] is None
    run_summary.main([str(root), '--out', str(tmp_path / 's.json'), '--markdown', str(tmp_path / 's.md')])
    assert (tmp_path / 's.md').read_text().count('\n') == 2 + len(rows)


def test_compare_runs_pairs_every_request(runs, tmp_path):
    _, rs = runs
    c = compare_runs.compare(rs['offload'][0], rs['conditioned'][0])
    n = sum(1 for _ in open(rs['offload'][0] / 'replay/requests.jsonl'))
    assert c['paired']['paired_requests'] == n
    assert c['metrics']['prefill_computed_tokens']['right'] == rs['conditioned'][1]['truth']['prefill_computed']
    assert '| metric |' in compare_runs.markdown(c)
