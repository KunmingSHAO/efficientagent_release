"""Read-only sampler of the LMCache CPU tier, loaded through PYTHONPATH by the run launcher.

Active only when ``EA_LMCACHE_OBSERVER_DIR`` is set. An import hook wraps
``lmcache.v1.storage_backend.local_cpu_backend`` after it loads: every ``LocalCPUBackend`` starts a daemon thread
that appends one row every ``EA_LMCACHE_OBSERVER_INTERVAL_S`` seconds (default 10) to
``<dir>/cpu-<pid>.jsonl``: resident keys, pinned objects (counted under the backend's ``cpu_lock``), cumulative
capacity evictions (``batched_remove`` with ``force=False``), the configured capacity and the process RSS. The
sampler only reads backend state; it does not change caching behavior.
"""
import os
import sys

if os.environ.get('EA_LMCACHE_OBSERVER_DIR'):
    import importlib.abc
    import importlib.util
    import json
    import threading
    import time

    TARGET = 'lmcache.v1.storage_backend.local_cpu_backend'
    INTERVAL_S = float(os.environ.get('EA_LMCACHE_OBSERVER_INTERVAL_S', '10'))

    def _patch(mod):
        cls = mod.LocalCPUBackend; init = cls.__init__; brem = cls.batched_remove
        state = dict(evicted=0, evict_calls=0)

        def batched_remove(self, keys, force=True):
            n = brem(self, keys, force)
            if not force: state['evicted'] += len(keys); state['evict_calls'] += 1
            return n

        def sampler(self):
            path = os.path.join(os.environ['EA_LMCACHE_OBSERVER_DIR'], f'cpu-{os.getpid()}.jsonl')
            while True:
                time.sleep(INTERVAL_S)
                try:
                    keys = len(self.hot_cache); pinned = 0
                    with self.cpu_lock:
                        for o in list(self.hot_cache.values()):
                            if getattr(o, 'is_pinned', False): pinned += 1
                    rss = 0
                    for line in open('/proc/self/status'):
                        if line.startswith('VmRSS'): rss = int(line.split()[1]) * 1024
                    row = dict(t=time.time(), keys=keys, pinned=pinned, evicted=state['evicted'], evict_calls=state['evict_calls'],
                               capacity_gb=float(getattr(self.config, 'max_local_cpu_size', 0) or 0), rss_bytes=rss)
                    with open(path, 'a') as f: f.write(json.dumps(row) + '\n')
                except Exception as e:
                    try:
                        with open(path, 'a') as f: f.write(json.dumps(dict(t=time.time(), error=repr(e))) + '\n')
                    except Exception: pass

        def __init__(self, *a, **k):
            init(self, *a, **k)
            os.makedirs(os.environ['EA_LMCACHE_OBSERVER_DIR'], exist_ok=True)
            threading.Thread(target=sampler, args=(self,), daemon=True, name='ea-lmcache-observer').start()

        cls.__init__ = __init__; cls.batched_remove = batched_remove

    class _Loader(importlib.abc.Loader):
        def __init__(self, inner): self.inner = inner
        def create_module(self, spec): return self.inner.create_module(spec)
        def exec_module(self, module): self.inner.exec_module(module); _patch(module)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != TARGET: return None
            sys.meta_path.remove(self)
            try: spec = importlib.util.find_spec(name)
            finally: sys.meta_path.insert(0, self)
            if spec and spec.loader: spec.loader = _Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, _Finder())
