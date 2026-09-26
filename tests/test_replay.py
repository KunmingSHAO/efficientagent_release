"""CPU tests of the replay path: trace builder, closed-loop client against a mock server, forced-output processor."""
import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from efficientagent.analysis.common import load_trace
from efficientagent.replay import build_trace, client, synthetic_trace


class MockServer(BaseHTTPRequestHandler):
    """Echoes the forced output and reports the prompt length, like the real server with the logits processor."""
    seen = []

    def do_GET(self):
        data = json.dumps(dict(data=[dict(id='mock-model')])).encode()
        self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        forced = [int(x) for x in body.get('vllm_xargs', {}).get('ea_forced_output', '').split(',') if x]
        MockServer.seen.append(body)
        r = dict(choices=[dict(token_ids=forced[:body['max_tokens']], finish_reason='length')],
                 usage=dict(prompt_tokens=len(body['prompt']), completion_tokens=len(forced)))
        data = json.dumps(r).encode(); self.send_response(200); self.send_header('Content-Length', str(len(data))); self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a): pass


@pytest.fixture
def server():
    MockServer.seen = []
    srv = ThreadingHTTPServer(('127.0.0.1', 0), MockServer); th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    yield 'http://127.0.0.1:%d' % srv.server_address[1]
    srv.shutdown()


def write_agent_run(root, tasks=3, steps=4):
    """Agent-log directory in the trace builder's input format, with one failed call per task."""
    for k in range(tasks):
        inst = f'repo__issue-{k}'; base = 10**12 * (k + 1)
        (root / 'tasks').mkdir(parents=True, exist_ok=True)
        att = root / 'attempts' / inst / 'attempt_01' / 'telemetry'; att.mkdir(parents=True)
        calls = []; t = base + 2 * 10**9; prompt = [1, 2, 3, 100 + k]
        for s in range(steps):
            out = [7, 8, 9][: 1 + s % 3]
            calls.append(dict(trial_id=f'trial{k}', request_sequence=s + 1, error=None, token_identity_verified=True, prompt_token_ids=list(prompt),
                              generated_token_ids=[out], start_wall_ns=t, end_wall_ns=t + 10**9, client_elapsed_ns=10**9))
            t += 3 * 10**9; prompt = prompt + out + [50 + s]
        calls.insert(2, dict(trial_id=f'trial{k}', request_sequence=99, error='timeout', prompt_token_ids=[], generated_token_ids=[[]],
                             start_wall_ns=t, end_wall_ns=t, client_elapsed_ns=0))
        (att / 'llm_requests.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in calls))
        (root / 'tasks' / f'{inst}.json').write_text(json.dumps(dict(attempt_index=1, trial_id=f'trial{k}',
                                                                     task_timing=dict(assigned_ns=base, finished_ns=t + 5 * 10**9))))


def test_build_trace_roundtrip(tmp_path):
    write_agent_run(tmp_path / 'agent')
    s = build_trace.build(tmp_path / 'agent', tmp_path / 'trace')
    assert (s['tasks'], s['steps'], s['skipped_calls']) == (3, 12, 3)
    tasks = load_trace(tmp_path / 'trace')
    for t in tasks:
        c = t['calls']
        assert c[0]['gap'] == pytest.approx(2.0) and all(x['gap'] == pytest.approx(2.0) for x in c[1:])
        assert all(c[i + 1]['shared'] == c[i]['plen'] for i in range(len(c) - 1))   # append-only history
    head, steps = client.load_task(sorted((tmp_path / 'trace').glob('*.jsonl.gz'))[0])
    assert head['attempt'] == 'attempts/repo__issue-0/attempt_01' and head['tail_gap_s'] == pytest.approx(7.0)
    manifest = json.loads((tmp_path / 'trace/trace_manifest.json').read_text())
    assert not any(str(tmp_path) in json.dumps(v) for v in manifest.values())


def test_client_preserves_dependencies_and_forces_outputs(tmp_path, server):
    synthetic_trace.generate(tmp_path / 'trace', tasks=3, steps=(2, 3), system_tokens=64, task_tokens=32,
                             observation_tokens=(8, 16), output_tokens=(2, 5), gap_s=(0.05, 0.06), seed=1)
    s = client.run(tmp_path / 'trace', tmp_path / 'out', server, workers=2)
    assert s['model'] == 'mock-model' and s['all_outputs_forced_exactly'] and s['prompt_len_mismatch'] == 0
    assert s['requests'] == s['forced_match'] == sum(len(t['calls']) for t in load_trace(tmp_path / 'trace'))
    rows = [json.loads(line) for line in open(tmp_path / 'out/requests.jsonl')]
    tasks = {t['inst']: t for t in load_trace(tmp_path / 'trace')}
    for inst, t in tasks.items():
        mine = sorted((r for r in rows if r['instance_id'] == inst), key=lambda r: r['seq'])
        for a, b in zip(mine, mine[1:]):
            assert b['start'] - a['end'] >= 0.045                                    # previous response + recorded gap
    bodies = {(b['kv_transfer_params']['efficientagent.task_id'], b['kv_transfer_params']['efficientagent.request_sequence']): b
              for b in MockServer.seen}
    for inst, t in tasks.items():
        for c in t['calls']:
            b = bodies[(inst, c['seq'])]
            assert b['prompt'] == c['prompt'].tolist() and b['ignore_eos'] and b['max_tokens'] == c['olen']


def test_synthetic_trace_shares_prefixes(tmp_path):
    synthetic_trace.generate(tmp_path, tasks=2, steps=(3, 3), system_tokens=16, task_tokens=8, observation_tokens=(4, 4),
                             output_tokens=(2, 2), seed=0)
    tasks = load_trace(tmp_path)
    assert tasks[0]['calls'][0]['prompt'][:16].tolist() == tasks[1]['calls'][0]['prompt'][:16].tolist()
    assert [c['shared'] for c in tasks[0]['calls']] == [0, 24, 30]


def test_forced_output_processor():
    torch = pytest.importorskip('torch')
    pytest.importorskip('vllm.v1.sample.logits_processor')
    from vllm import SamplingParams
    from vllm.v1.sample.logits_processor import BatchUpdate, MoveDirectionality
    from efficientagent.replay.forced_output import ForcedOutputLogitsProcessor
    p = ForcedOutputLogitsProcessor(None, torch.device('cpu'), False)
    assert not p.is_argmax_invariant()
    out0, out1 = [], []
    sp = SamplingParams(max_tokens=3, extra_args={'ea_forced_output': '5,7,9'}); plain = SamplingParams(max_tokens=3)
    p.update_state(BatchUpdate(batch_size=2, removed=[], added=[(0, sp, [1, 2], out0), (1, plain, [3], out1)], moved=[]))
    base = torch.randn(2, 20); base[1, 11] = 100.
    for want in (5, 7, 9):
        logits = p.apply(base.clone())
        assert int(logits[0].argmax()) == want and int(logits[1].argmax()) == 11 and int(torch.isinf(logits[0]).sum()) == 19
        out0.append(want); out1.append(11)
    assert torch.equal(p.apply(base.clone()), base)                                  # exhausted: untouched
    out2 = []
    p.update_state(BatchUpdate(batch_size=2, removed=[0], added=[(0, SamplingParams(extra_args={'ea_forced_output': '3'}), [1], out2)],
                               moved=[(0, 1, MoveDirectionality.SWAP)]))
    logits = p.apply(base.clone())
    assert int(logits[1].argmax()) == 3 and int(logits[0].argmax()) == int(base[0].argmax())
    with pytest.raises(ValueError):
        ForcedOutputLogitsProcessor.validate_params(SamplingParams(extra_args={'ea_forced_output': 'x,1'}))
