"""SQLite stores immutable array bundles and incremental run records without small files.

SQLite transactions are the completion marker; OS locks serialize work on one key.
Connections are short lived and safe with spawned workers. Array payloads use NPZ,
never pickle. Large individual payloads are bounded by one motion/asset.
"""
from contextlib import contextmanager
import fcntl
import hashlib
from io import BytesIO
from functools import lru_cache
from zipfile import BadZipFile
import json
from pathlib import Path
import sqlite3

import numpy as np


class Store:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS artifacts (key TEXT PRIMARY KEY, metadata TEXT NOT NULL, payload BLOB NOT NULL, checksum TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS results (name TEXT PRIMARY KEY, ordinal INTEGER NOT NULL, result TEXT NOT NULL)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=600)
        db.execute('PRAGMA busy_timeout=600000')
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def lock(self, key):
        # Fixed lock stripes keep file count bounded even for millions of keys.
        stripe = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 32
        with self.path.with_suffix(f'.lock-{stripe:02d}').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def get(self, key):
        with self.connect() as db:
            row = db.execute('SELECT metadata,payload,checksum FROM artifacts WHERE key=?', (key,)).fetchone()
        if row is None:
            return None
        metadata, payload, checksum = row
        if hashlib.sha256(payload).hexdigest() != checksum:
            return None
        try:
            with np.load(BytesIO(payload), allow_pickle=False) as data:
                arrays = {name: data[name].copy() for name in data.files}
            return json.loads(metadata), arrays
        except (ValueError, OSError, EOFError, BadZipFile):
            return None

    def put(self, key, metadata, **arrays):
        stream = BytesIO()
        np.savez_compressed(stream, **arrays)
        payload = stream.getvalue()
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?)',
                       (key, json.dumps(metadata, allow_nan=False), payload, hashlib.sha256(payload).hexdigest()))
        return dict(store=str(self.path), key=key)

    def record_many(self, rows):
        if rows:
            with self.connect() as db:
                db.executemany('INSERT OR REPLACE INTO results VALUES (?,?,?)',
                    [(name, ordinal, json.dumps(result, allow_nan=False)) for name,ordinal,result in rows])

    def results(self):
        with self.connect() as db:
            for (value,) in db.execute('SELECT result FROM results ORDER BY ordinal,name'):
                yield json.loads(value)


def load_arrays(reference):
    if reference.get('kind') == 'source_g1':
        from data_pipeline.sources.base import SourceMotion
        from data_pipeline.sources.omg import read_motion
        if digest(reference['path']) != reference['checksum']:
            raise ValueError('Source motion changed since processing')
        motion = SourceMotion('source_g1', Path(reference['path']), representation='g1_qpos36', metadata=reference['metadata'])
        return {'qpos_36': read_motion(motion)}
    if not Path(reference['store']).is_file():
        raise ValueError('Missing artifact store')
    item = _open_store(reference['store']).get(reference['key'])
    if item is None:
        raise ValueError(f'Missing or corrupt artifact {reference["key"]}')
    return item[1]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


@lru_cache(maxsize=4096)
def _digest(path, size, mtime):
    from data_pipeline.io import sha256_file
    return sha256_file(path)


def digest(path):
    p = Path(path); st = p.stat()
    return _digest(str(p.resolve()), st.st_size, st.st_mtime_ns)


@lru_cache(maxsize=8)
def _open_store(path):
    return Store(path)
