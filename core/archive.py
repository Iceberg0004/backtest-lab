"""v9: results that outlive the 24-hour job folder.
- Every finished search / scan is also kept in a small local SQLite file (compressed). It survives the /tmp clean-up,
  but NOT a full rebuild of the free cloud server, so:
- The user can download all saved results as one .btlab file and upload it later to bring Riwayat back.
  No account, token or outside service is needed.
Uploaded files are only accepted if they contain plain data (numbers, tables, text): loading uses a strict allow-list,
never the normal unpickler, so a tampered file cannot run code on the server."""
from __future__ import annotations
import io, json, os, pickle, sqlite3, threading, time, zipfile, zlib

MAX_MB = float(os.environ.get("BT_ARCHIVE_MAX_MB", "150"))
MAX_ITEMS = int(os.environ.get("BT_ARCHIVE_MAX_ITEMS", "200"))
MAX_UPLOAD_MB = 60
MAX_ITEM_RAW_MB = 80          # one result unpacked (the largest seen so far is ~2 MB)
KINDS = ("search", "scan")
MAGIC = "backtest-lab-results"
_LOCK = threading.Lock()


def _path():
    p = os.environ.get("BT_ARCHIVE")
    if p:
        return p
    for d in (os.path.join(os.path.expanduser("~"), ".backtest_lab"), "/tmp/backtest_lab_archive"):
        try:
            os.makedirs(d, exist_ok=True)
            t = os.path.join(d, ".w")
            open(t, "w").close(); os.remove(t)
            return os.path.join(d, "archive.sqlite")
        except OSError:
            continue
    return ":memory:"


def _db():
    c = sqlite3.connect(_path(), timeout=10)
    c.execute("CREATE TABLE IF NOT EXISTS res (id TEXT PRIMARY KEY, kind TEXT, label TEXT, finished REAL, meta TEXT, "
              "blob BLOB, size INTEGER)")
    return c


# ------------------------------------------------------------------ safe loading
_ALLOW = {
    ("builtins", "slice"), ("builtins", "set"), ("builtins", "frozenset"), ("builtins", "complex"), ("builtins", "range"),
    ("builtins", "bytearray"), ("collections", "OrderedDict"), ("collections", "defaultdict"),
    ("datetime", "datetime"), ("datetime", "date"), ("datetime", "time"), ("datetime", "timedelta"), ("datetime", "timezone"),
    ("zoneinfo", "ZoneInfo"),
    ("numpy", "dtype"), ("numpy", "ndarray"), ("numpy._core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "scalar"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"), ("numpy._core.numeric", "_frombuffer"),
    ("numpy.core.numeric", "_frombuffer"),
    ("pandas", "DataFrame"), ("pandas", "Series"), ("pandas", "Index"), ("pandas", "DatetimeIndex"), ("pandas", "RangeIndex"),
    ("pandas", "MultiIndex"), ("pandas", "Timestamp"), ("pandas", "Timedelta"), ("pandas", "DatetimeTZDtype"),
    ("pandas", "StringDtype"), ("pandas", "CategoricalDtype"), ("pandas", "Categorical"),
    ("pandas._libs.arrays", "__pyx_unpickle_NDArrayBacked"), ("pandas._libs.internals", "_unpickle_block"),
    ("pandas._libs.tslibs.timestamps", "_unpickle_timestamp"), ("pandas._libs.tslibs.timedeltas", "_timedelta_unpickle"),
    ("pandas.arrays", "ArrowStringArray"), ("pandas.arrays", "DatetimeArray"), ("pandas.arrays", "StringArray"),
    ("pandas.core.indexes.base", "_new_Index"), ("pandas.core.indexes.datetimes", "_new_DatetimeIndex"),
    ("pandas.core.indexes.range", "RangeIndex"), ("pandas.core.indexes.multi", "MultiIndex"),
    ("pandas.core.internals.managers", "BlockManager"), ("pandas.core.internals.managers", "SingleBlockManager"),
    ("pandas.core.frame", "DataFrame"), ("pandas.core.series", "Series"),
    ("pyarrow.lib", "_restore_array"), ("pyarrow.lib", "py_buffer"), ("pyarrow.lib", "type_for_alias"),
}
_GETATTR_OK = {("zoneinfo.ZoneInfo", "_unpickle")}


def _safe_getattr(obj, name):
    import zoneinfo
    key = (f"{getattr(obj, '__module__', '')}.{getattr(obj, '__name__', '')}", name)
    if obj is not zoneinfo.ZoneInfo or key not in _GETATTR_OK:
        raise pickle.UnpicklingError(f"tidak diizinkan: getattr {key}")
    return getattr(obj, name)


class _Safe(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("builtins", "getattr"):
            return _safe_getattr
        if (module, name) in _ALLOW:
            return super().find_class(module, name)
        if module == "pandas._libs.tslibs.offsets":          # date frequencies (Hour, Day, ...): plain data classes
            from pandas._libs.tslibs.offsets import BaseOffset
            obj = super().find_class(module, name)
            if isinstance(obj, type) and issubclass(obj, BaseOffset):
                return obj
        raise pickle.UnpicklingError(f"tidak diizinkan: {module}.{name}")


def safe_loads(raw: bytes):
    return _Safe(io.BytesIO(raw)).load()


def _unz(blob: bytes) -> bytes:
    d = zlib.decompressobj()
    raw = d.decompress(blob, int(MAX_ITEM_RAW_MB * 1e6))
    if d.unconsumed_tail:
        raise ValueError("isi file terlalu besar")
    return raw


# ------------------------------------------------------------------ store / list / load
def put(meta: dict, raw: bytes):
    """raw = pickle bytes of a job result (already written by core.jobs)."""
    if meta.get("kind") not in KINDS:
        return
    blob = zlib.compress(raw, 6)
    with _LOCK:
        c = _db()
        try:
            c.execute("INSERT OR REPLACE INTO res VALUES (?,?,?,?,?,?,?)",
                      (meta["id"], meta.get("kind"), meta.get("label") or "", float(meta.get("finished") or time.time()),
                       json.dumps(meta, default=str), blob, len(blob)))
            _prune(c)
            c.commit()
        finally:
            c.close()


def _prune(c):
    rows = c.execute("SELECT id, size FROM res ORDER BY finished DESC").fetchall()
    tot = 0
    for k, (jid, sz) in enumerate(rows):
        tot += sz or 0
        if k >= MAX_ITEMS or tot > MAX_MB * 1e6:
            c.execute("DELETE FROM res WHERE id=?", (jid,))


def listing(limit=200):
    with _LOCK:
        c = _db()
        try:
            rows = c.execute("SELECT meta, size FROM res ORDER BY finished DESC LIMIT ?", (int(limit),)).fetchall()
        finally:
            c.close()
    out = []
    for m, sz in rows:
        try:
            d = json.loads(m); d["archived"] = True; d["size"] = sz
            out.append(d)
        except Exception:
            pass
    return out


def has(jid) -> bool:
    try:
        with _LOCK:
            c = _db()
            try:
                return c.execute("SELECT 1 FROM res WHERE id=?", (jid,)).fetchone() is not None
            finally:
                c.close()
    except Exception:
        return False


def get(jid):
    """(meta, raw pickle bytes) or None."""
    with _LOCK:
        c = _db()
        try:
            r = c.execute("SELECT meta, blob FROM res WHERE id=?", (jid,)).fetchone()
        finally:
            c.close()
    if not r:
        return None
    return json.loads(r[0]), _unz(r[1])


def delete(jid):
    with _LOCK:
        c = _db()
        try:
            c.execute("DELETE FROM res WHERE id=?", (jid,)); c.commit()
        finally:
            c.close()


def count():
    with _LOCK:
        c = _db()
        try:
            return c.execute("SELECT COUNT(*), COALESCE(SUM(size),0) FROM res").fetchone()
        finally:
            c.close()


# ------------------------------------------------------------------ file export / import
def export_bytes() -> bytes:
    """All saved results as one zip (.btlab): manifest.json + one compressed blob per result."""
    with _LOCK:
        c = _db()
        try:
            rows = c.execute("SELECT id, meta, blob FROM res ORDER BY finished DESC").fetchall()
        finally:
            c.close()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("manifest.json", json.dumps(dict(format=MAGIC, version=1, exported=time.time(),
                                                    items=[json.loads(m) for _, m, _ in rows]), default=str))
        for jid, _, blob in rows:
            z.writestr(f"r/{jid}.z", blob)
    return buf.getvalue()


def import_bytes(data: bytes):
    """-> (added, skipped, errors). Each result is checked with the strict loader before it is stored."""
    if len(data) > MAX_UPLOAD_MB * 1e6:
        raise ValueError(f"file lebih dari {MAX_UPLOAD_MB} MB")
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        man = json.loads(z.read("manifest.json"))
    except Exception:
        raise ValueError("bukan file riwayat Backtest Lab (.btlab)")
    if man.get("format") != MAGIC:
        raise ValueError("bukan file riwayat Backtest Lab (.btlab)")
    added = skipped = 0
    errs = []
    for m in man.get("items", [])[:MAX_ITEMS]:
        jid = str(m.get("id", ""))
        if not jid or not all(ch.isalnum() or ch == "_" for ch in jid) or m.get("kind") not in KINDS:
            errs.append(jid or "?"); continue
        if has(jid):
            skipped += 1; continue
        try:
            info = z.getinfo(f"r/{jid}.z")
            if info.file_size > MAX_ITEM_RAW_MB * 1e6:
                raise ValueError("terlalu besar")
            raw = _unz(z.read(info))
            safe_loads(raw)                      # reject anything that is not plain data
            put(m, raw)
            added += 1
        except Exception:
            errs.append(jid)
    return added, skipped, errs
