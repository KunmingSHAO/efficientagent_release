"""CPU test of the full launcher flow against a stand-in server process (no vLLM, no GPU)."""
import json
import socket
import stat
import sys
import textwrap

from efficientagent.replay import launch, synthetic_trace

MOCK = textwrap.dedent('''\
    #!{python}
    """Stand-in for the vLLM OpenAI server: /health, /metrics, /v1/models, /v1/completions with forced outputs."""
    import json, sys, threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    port = int(sys.argv[sys.argv.index('--port') + 1]); lock = threading.Lock(); c = dict(req=0, prompt=0, gen=0)
    print('INFO GPU KV cache size: 12,345 tokens', flush=True)

    class H(BaseHTTPRequestHandler):
        def reply(self, data, ctype='application/json'):
            self.send_response(200); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(data)))
            self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path == '/health': return self.reply(b'')
            if self.path == '/v1/models': return self.reply(json.dumps(dict(data=[dict(id='stand-in')])).encode())
            with lock:
                text = ''.join(f'{{k}}{{{{engine="0"}}}} {{v}}\\n' for k, v in (
                    ('vllm:request_success_total', c['req']), ('vllm:prompt_tokens_total', c['prompt']),
                    ('vllm:generation_tokens_total', c['gen']), ('vllm:request_prefill_kv_computed_tokens_sum', c['prompt'] // 2),
                    ('vllm:num_preemptions_total', 0), ('vllm:request_queue_time_seconds_sum', 0.5 * c['req']),
                    ('vllm:request_queue_time_seconds_count', c['req']), ('vllm:num_requests_running', 1)))
            self.reply(text.encode(), 'text/plain')
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            forced = [int(x) for x in body.get('vllm_xargs', {{}}).get('ea_forced_output', '').split(',') if x][:body['max_tokens']]
            with lock: c['req'] += 1; c['prompt'] += len(body['prompt']); c['gen'] += len(forced)
            self.reply(json.dumps(dict(choices=[dict(token_ids=forced, finish_reason='length')],
                                       usage=dict(prompt_tokens=len(body['prompt']), completion_tokens=len(forced)))).encode())
        def log_message(self, *a): pass

    ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
    ''')


def test_launch_execute_with_stand_in_server(tmp_path):
    exe = tmp_path / 'stand_in_server'
    exe.write_text(MOCK.format(python=sys.executable)); exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    synthetic_trace.generate(tmp_path / 'trace', tasks=3, steps=(2, 3), system_tokens=64, task_tokens=32, observation_tokens=(8, 16),
                             output_tokens=(2, 4), gap_s=(0.01, 0.02), seed=3)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    out = tmp_path / 'run'
    rc = launch.main(['--admission', 'conditioned', '--host-gib', '5', '--model', 'stand-in', '--kv-bytes-per-token', '24576',
                      '--python', str(exe), '--port', str(port), '--trace', str(tmp_path / 'trace'), '--out', str(out),
                      '--workers', '2', '--greedy-check', '1', '--startup-timeout', '60', '--execute'])
    run = json.loads((out / 'run.json').read_text())
    assert rc == 0 and run['status'] == 'passed', run.get('errors')
    assert run['gpu_kv_tokens'] == 12345 and run['spec']['admission'] == 'conditioned'
    assert run['replay']['forced_match'] == run['replay']['requests'] > 0
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['env']['EA_KVTIER_ADMISSION'] == 'conditioned' and manifest['server_command'][0] == str(exe)
    m = json.loads((out / 'run_metrics.json').read_text())
    assert m['prompt_tokens'] == sum(json.loads(line)['prompt_len'] for line in open(out / 'replay/requests.jsonl'))
    assert m['prefill_computed_tokens'] == m['prompt_tokens'] // 2 and m['mean_queue_s'] == 0.5
    assert json.loads((out / 'greedy_check.json').read_text())
