"""End-to-end replay on a real server (GPU). Skipped unless run with --run-gpu and EA_TEST_MODEL is a local model dir.

Example::

    EA_TEST_MODEL=/models/Qwen2.5-0.5B-Instruct pytest --run-gpu tests/test_gpu_replay.py
"""
import json
import os
import socket
from pathlib import Path

import pytest

from efficientagent.replay import launch, synthetic_trace


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); return s.getsockname()[1]


@pytest.mark.gpu
@pytest.mark.parametrize('admission,host_gib', [('none', 0), ('none', 1), ('fixed', 1), ('conditioned', 1)])
def test_replay_end_to_end(tmp_path, admission, host_gib):
    model = os.environ.get('EA_TEST_MODEL')
    if not model or not Path(model, 'config.json').exists():
        pytest.skip('set EA_TEST_MODEL to a local model directory')
    synthetic_trace.generate(tmp_path / 'trace', tasks=4, steps=(3, 4), system_tokens=1024, task_tokens=1024,
                             observation_tokens=(200, 600), output_tokens=(8, 32), gap_s=(0.01, 0.05), token_range=(1000, 20000))
    out = tmp_path / 'run'
    rc = launch.main(['--admission', admission, '--host-gib', str(host_gib), '--model', model, '--trace', str(tmp_path / 'trace'),
                      '--out', str(out), '--workers', '4', '--max-model-len', '16384', '--port', str(free_port()),
                      '--gpu-memory-utilization', os.environ.get('EA_TEST_GPU_MEMORY', '0.5'), '--execute'])
    run = json.loads((out / 'run.json').read_text())
    assert rc == 0 and run['status'] == 'passed', run.get('errors')
    assert run['replay']['forced_mismatch'] == 0 and run['replay']['errors'] == 0
    metrics = json.loads((out / 'run_metrics.json').read_text())
    assert metrics['prefill_computed_tokens'] is not None
    if admission in ('fixed', 'conditioned'):
        assert run['kvtier_counters'].get('scheduler.decisions', 0) > 0
