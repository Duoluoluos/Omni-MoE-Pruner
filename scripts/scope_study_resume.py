"""Durable per-question checkpoints for the new scope study; no model policy."""
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value), 'Nonfinite checkpoint value'
    elif isinstance(value, dict):
        for v in value.values(): finite(v)
    elif isinstance(value, list):
        for v in value: finite(v)


def completed_rows(out, protocol, records, gates_sha=None):
    """Only atomically committed whole questions count; partials never count."""
    rows = {}
    expected = set(protocol['evaluation_indices'])
    for path in sorted((out/'predictions').glob('*.json')):
        if path.name.endswith('.partial.json'):
            continue
        assert re.fullmatch(r'\d{4}\.json', path.name), path
        row = json.loads(path.read_text())
        idx = row['sample_index']
        assert idx in expected and idx == int(path.stem) and idx not in rows
        assert row.get('complete') is True and row['video'] == records[idx]['video']
        assert set(row['arms']) == set(protocol['arms'])
        assert row['counts']['audio'] > 0 and row['counts']['video'] > 0
        assert row['timed'] == (idx >= 5)
        assert row['execution_session'] and row['input_sha256']
        if gates_sha is not None:
            assert row['gates_sha256'] == gates_sha, 'Gate drift at resume'
        for a in row['arms'].values():
            assert a['generation_s'] > 0 and set(a['option_logp']) == set('ABCD')
        if idx in protocol['diagnostic_indices']:
            assert row['native']['work_ledger']
            assert all('js_to_native' in a for a in row['arms'].values())
            assert set(row['probes']) == set(protocol['probes'])
            assert len(row['native_decomposition']) == (48 if protocol['model']=='qwen' else 28)*3
            assert all(a.get('work_ledger') for a in row['arms'].values())
        finite(row)
        rows[idx] = row
    return rows


def rebuild_ledger(out, rows, session):
    """Reconcile a killed append using durable question files as authority."""
    raw = out/'raw_results.jsonl'
    contents = ''.join(json.dumps(rows[i], allow_nan=False)+'\n' for i in sorted(rows))
    previous = raw.read_text() if raw.exists() else ''
    if previous == contents:
        return
    if raw.exists():
        shutil.copy2(raw, session/'raw_before_reconciliation.jsonl')
    tmp = raw.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        f.write(contents); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, raw)


@contextlib.contextmanager
def run_lock(out, recover=False):
    lock = out/'.run_lock'
    if lock.exists():
        owner = json.loads((lock/'owner.json').read_text())
        if owner['host'] == socket.gethostname():
            try: os.kill(owner['pid'], 0)
            except ProcessLookupError: pass
            else: raise RuntimeError(f'Run is still owned by process {owner}')
        if not recover:
            raise RuntimeError('Stale lock needs --recover-lock after confirming its old worker has stopped')
        archive = out/'sessions'/f'recovered_lock_{time.time_ns()}'
        archive.parent.mkdir(exist_ok=True)
        os.rename(lock, archive)
    lock.mkdir()
    write(lock/'owner.json', dict(host=socket.gethostname(), pid=os.getpid(), time=time.time()))
    try:
        yield
    finally:
        (lock/'owner.json').unlink(missing_ok=True)
        lock.rmdir()
