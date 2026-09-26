"""Shared pytest configuration: GPU marker handling and a vLLM-free import path for the connector."""
import importlib
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pytest_addoption(parser):
    parser.addoption('--run-gpu', action='store_true', default=False, help='run tests marked gpu (needs CUDA, vLLM, LMCache)')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-gpu') or os.environ.get('EA_RUN_GPU_TESTS') == '1':
        return
    skip = pytest.mark.skip(reason='GPU test: pass --run-gpu (or set EA_RUN_GPU_TESTS=1) on a machine with CUDA, vLLM and LMCache')
    for item in items:
        if 'gpu' in item.keywords:
            item.add_marker(skip)


def _install_vllm_stub(monkeypatch):
    """Minimal stand-in for the one vLLM class the connector subclasses, used when vLLM is not installed."""
    names = ['vllm', 'vllm.distributed', 'vllm.distributed.kv_transfer', 'vllm.distributed.kv_transfer.kv_connector',
             'vllm.distributed.kv_transfer.kv_connector.v1', 'vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector']
    for n in names:
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))

    class LMCacheConnectorV1:
        def __init__(self, *a, **k): pass

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            return 0, False

    sys.modules[names[-1]].LMCacheConnectorV1 = LMCacheConnectorV1
    sys.modules.pop('efficientagent.kvtier.connector', None)


@pytest.fixture
def connector(monkeypatch, tmp_path):
    """The connector module with the native lookup stubbed (no server, no GPU) and a private telemetry file."""
    try:
        importlib.import_module('vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector')
    except ImportError:
        _install_vllm_stub(monkeypatch)
    mod = importlib.import_module('efficientagent.kvtier.connector')
    monkeypatch.setattr(mod.LMCacheConnectorV1, 'get_num_new_matched_tokens', lambda self, r, n: (0, False))
    # Plain assignment (not restored): a telemetry thread started by a test keeps a valid path after the test ends.
    mod.TELEMETRY_FILE = str(tmp_path / 'telemetry.json')
    monkeypatch.setenv('EA_KVTIER_STATS_DIR', str(tmp_path / 'stats'))
    return mod
