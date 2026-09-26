"""CPU tests of the admission option mapping and the server command built by the run launcher (no server start)."""
import json

import pytest

from efficientagent.kvtier.policies import AdmissionConfig, TierPlan, kv_bytes_per_token, kv_bytes_per_token_from_config
from efficientagent.replay import launch


def test_kv_bytes_per_token():
    # 48 layers, 4 KV heads, head dim 128, BF16: 96 KiB per token, 24 KiB per rank at TP4 and TP8 (heads replicated).
    assert [kv_bytes_per_token(48, 4, 128, tp) for tp in (1, 2, 4, 8)] == [98304, 49152, 24576, 24576]
    assert kv_bytes_per_token(64, 8, 128, 8) == 32768


def test_kv_bytes_per_token_from_config(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps(dict(num_hidden_layers=48, num_key_value_heads=4, num_attention_heads=32,
                                                          head_dim=128, hidden_size=2048)))
    assert kv_bytes_per_token_from_config(tmp_path, tp=8) == 24576


@pytest.mark.parametrize('admission,host,use_connector', [('fixed', 0, False), ('conditioned', 0, False), ('none', 0, True)])
def test_admission_needs_host_tier(admission, host, use_connector):
    with pytest.raises(ValueError):
        TierPlan(admission, host, use_connector=use_connector, params=AdmissionConfig(bytes_per_token=1))


def test_conditioned_needs_bytes_per_token():
    with pytest.raises(ValueError):
        TierPlan('conditioned', 5)


def test_plans():
    native = TierPlan('none', 5, tp=8)
    args = native.server_args()
    assert args[args.index('--kv-offloading-size') + 1] == '40.0' and native.connector_env() == {}
    assert json.loads(args[args.index('--kv-transfer-config') + 1])['kv_connector'] == 'LMCacheConnectorV1'
    assert native.lmcache_env()['LMCACHE_MAX_LOCAL_CPU_SIZE'] == '5.0'
    for admission, use_connector in (('none', True), ('fixed', False), ('conditioned', False)):
        p = TierPlan(admission, 10, tp=8, use_connector=use_connector, params=AdmissionConfig(bytes_per_token=24576))
        a = p.server_args()
        assert '--kv-offloading-size' not in a
        cfg = json.loads(a[a.index('--kv-transfer-config') + 1])
        assert (cfg['kv_connector'], cfg['kv_connector_module_path']) == ('EfficientAgentConnector', 'efficientagent.kvtier.connector')
        assert cfg['kv_connector_extra_config']['lmcache.max_local_cpu_size'] == 10.0
        env = p.connector_env(stats_dir='S', telemetry_file='T')
        assert env['EA_KVTIER_ADMISSION'] == admission and env['EA_KVTIER_WRITE_THRESHOLD'] == '8'
        assert ('EA_KVTIER_TELEMETRY_FILE' in env) == (admission == 'conditioned')
    recompute = TierPlan('none', 0)
    assert recompute.server_args() == [] and recompute.lmcache_env() == {'VLLM_PLUGINS': ''}


def plan_for(tmp_path, *extra):
    a = launch.parse_args(['--model', 'org/model', '--trace', str(tmp_path / 'trace'), '--out', str(tmp_path / 'run'),
                           '--tp', '8', '--kv-bytes-per-token', '24576', *extra])
    plan = launch.build_plan(a)
    cmd, env = launch.server_command(a, plan, tmp_path / 'run')
    return a, plan, cmd, env


def test_server_commands_differ_only_in_admission(tmp_path):
    cmds = {}
    for adm in ('fixed', 'conditioned'):
        _, _, cmd, env = plan_for(tmp_path, '--admission', adm, '--host-gib', '5', '--max-num-seqs', '16')
        cmds[adm] = (cmd, env)
        assert cmd[cmd.index('--logits-processors') + 1] == launch.LOGITS_PROCESSOR
        assert env['PYTHONHASHSEED'] == '0' and str(launch.PACKAGE_ROOT) in env['PYTHONPATH']
        assert env['EA_LMCACHE_OBSERVER_DIR'].endswith('lmcache_observer')
    (cf, ef), (cc, ec) = cmds['fixed'], cmds['conditioned']
    assert cf == cc
    diff = {k for k in set(ef) | set(ec) if ef.get(k) != ec.get(k)}
    assert diff == {'EA_KVTIER_ADMISSION', 'EA_KVTIER_TELEMETRY_FILE'}


def test_recompute_and_native_commands(tmp_path):
    _, plan, cmd, env = plan_for(tmp_path, '--admission', 'none', '--host-gib', '0')
    assert not plan.host_tier and '--kv-transfer-config' not in cmd and env['VLLM_PLUGINS'] == ''
    assert not any(k.startswith(('EA_KVTIER', 'LMCACHE')) for k in env)
    _, plan, cmd, env = plan_for(tmp_path, '--admission', 'none', '--host-gib', '20')
    assert plan.native and cmd[cmd.index('--kv-offloading-size') + 1] == '160.0' and env['LMCACHE_MAX_LOCAL_CPU_SIZE'] == '20.0'


def test_dry_run_prints_plan(tmp_path, capsys):
    rc = launch.main(['--admission', 'conditioned', '--host-gib', '5', '--model', 'org/model', '--kv-bytes-per-token', '24576',
                      '--trace', str(tmp_path), '--out', str(tmp_path / 'run')])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out['spec']['admission'] == 'conditioned' and out['spec']['write_threshold'] == 8
    assert out['env']['EA_KVTIER_BYTES_PER_TOKEN'] == '24576' and not (tmp_path / 'run').exists()


def test_counter_totals_skip_non_counter_files(tmp_path):
    from efficientagent.analysis.common import run_counters
    d = tmp_path / 'kvtier_stats'; d.mkdir()
    (d / 'scheduler-1.json').write_text(json.dumps(dict(role='scheduler', counters=dict(decisions=5, skipped_requests=2))))
    (d / 'worker-2.json').write_text(json.dumps(dict(role='worker', counters=dict(cpu_evicted_chunks=7))))
    (d / 'worker-3.json').write_text(json.dumps(dict(role='worker', counters=dict(cpu_evicted_chunks=1))))
    (d / 'telemetry_rank0.json').write_text(json.dumps(dict(t=1.0, evicted_in_window=3, registered=10, capacity=10)))
    assert run_counters(tmp_path) == {'scheduler.decisions': 5, 'scheduler.skipped_requests': 2, 'worker.cpu_evicted_chunks': 8}
