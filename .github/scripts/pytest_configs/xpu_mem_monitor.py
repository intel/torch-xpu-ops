"""Background poller for ``xpu-smi`` GPU memory utilization.

Two roles share this module:

* **Writer** (xdist controller, or a non-xdist run): owns a daemon thread
  that periodically runs ``xpu-smi dump ... -n 1`` and atomically rewrites
  a small snapshot file (~100 B). Publishes its tile count via the
  ``XPU_SMI_TILES`` env var so workers don't need to subprocess.
* **Reader** (xdist workers): only call :func:`attach` then
  :func:`get_max_mem_util`. Pure file read + regex; never spawns
  subprocesses, never imports the installer module.

Public API
----------
start(interval=3, timeout=60) -> str | None
    Start the writer. Returns the snapshot path, or ``None`` if
    ``xpu-smi`` is unavailable. Idempotent. The writer always samples
    every visible device/tile; per-card filtering is reader-side via
    :func:`get_max_mem_util`.
stop() -> None
    Stop the writer (no-op for readers).
attach(path) -> None
    Bind this process to an existing snapshot file.
get_max_mem_util(cards=None) -> tuple[int, float] | None
    ``(card_id, util)`` with the highest GPU memory utilization across
    ``cards`` (``None`` = every device/tile in the snapshot). Synthetic
    card id matches ``card = dev * tiles + tile`` (or ``dev`` in
    non-tile mode).
log_path() -> str | None
    Current snapshot file path, or ``None``.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import threading
from contextlib import suppress
from pathlib import Path
from collections.abc import Iterable, Iterator

# Matches an xpu-smi CSV row, with or without a tile column:
#   "23:59:21.296,    0, 80"        (device-only)
#   "23:59:21.296,    0, 1, 80"     (device+tile)
_ROW_RE = re.compile(
    r"^\s*\d{1,2}:\d{2}:\d{2}\.\d+\s*,"
    r"\s*(\d+)\s*,"
    r"(?:\s*(\d+)\s*,)?"
    r"\s*([0-9.]+)\s*$"
)
_TILE_RE = re.compile(r"Tile\s+(\d+)")
_TILES_ENV = "XPU_SMI_TILES"

# Module state (per process) ---------------------------------------------------

_log_path: Path | None = None
_poller_thread: threading.Thread | None = None
_poller_stop: threading.Event | None = None
_interval: float = 3.0
_timeout: int = 60

_tiles_per_device: int | None = None
_tiles_lock = threading.Lock()


def _log(msg: str) -> None:
    print(f"[xpu-mem] {msg}", file=sys.stderr, flush=True)


# --- tile probing -------------------------------------------------------------

def _probe_tiles_subprocess() -> int:
    """Probe tiles-per-device via ``xpu-smi stats -d 0``. Writer-only."""
    try:
        r = subprocess.run(
            ["xpu-smi", "stats", "-d", "0"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return 1
    if r.returncode != 0:
        return 1
    return max(len({int(m.group(1)) for m in _TILE_RE.finditer(r.stdout)}), 1)


def _get_tiles_per_device() -> int:
    """Cached tiles-per-device, read from ``XPU_SMI_TILES``.

    Reader-safe: never spawns a subprocess. The writer publishes the
    real count in :func:`start` (see :func:`_resolve_tiles_writer`); if
    the env var is missing or malformed we conservatively default to 1
    so that the reader still works against a single-tile snapshot.
    """
    global _tiles_per_device
    if _tiles_per_device is not None:
        return _tiles_per_device
    with _tiles_lock:
        if _tiles_per_device is not None:
            return _tiles_per_device
        env_val = os.environ.get(_TILES_ENV, "")
        _tiles_per_device = max(int(env_val), 1) if env_val.isdigit() else 1
        return _tiles_per_device


def _resolve_tiles_writer() -> int:
    """Writer-side: probe via subprocess and cache. Called from :func:`start`."""
    global _tiles_per_device
    with _tiles_lock:
        if _tiles_per_device is None:
            _tiles_per_device = _probe_tiles_subprocess()
        return _tiles_per_device


# --- snapshot writer ----------------------------------------------------------

def _run_xpu_smi(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["xpu-smi", *args],
            stderr=subprocess.DEVNULL, text=True, timeout=_timeout,
        )
    except (subprocess.SubprocessError, OSError) as e:
        _log(f"xpu-smi {' '.join(args)} failed: {e}")
        return None


def _take_snapshot() -> str | None:
    tiles = _get_tiles_per_device()
    if tiles > 1:
        tile_arg = ",".join(str(t) for t in range(tiles))
        return _run_xpu_smi(["dump", "-t", tile_arg, "-m", "5", "-n", "1"])
    return _run_xpu_smi(["dump", "-m", "5", "-n", "1"])


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp_str = tempfile.mkstemp(prefix=".xpu_smi_", suffix=".tmp",
                                   dir=str(path.parent))
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except OSError as e:
        _log(f"snapshot write failed: {e}")
        with suppress(OSError):
            tmp.unlink(missing_ok=True)


def _poller_loop(stop_event: threading.Event, path: Path) -> None:
    while not stop_event.is_set():
        try:
            snap = _take_snapshot()
        except Exception as e:  # noqa: BLE001 - never let the thread die
            _log(f"poller iteration failed: {e!r}")
            snap = None
        if snap:
            _atomic_write(path, snap)
        if stop_event.wait(_interval):
            return


# --- public API ---------------------------------------------------------------

def start(interval: float = 3, timeout: int = 60) -> str | None:
    """Start the writer thread. Returns the snapshot path, or ``None``."""
    global _poller_thread, _poller_stop, _log_path, _interval, _timeout

    if _poller_thread is not None:
        return str(_log_path) if _log_path else None

    # Lazy import: only the writer process pulls in the installer module.
    try:
        from xpu_smi_install import ensure_xpu_smi
    except ImportError as e:
        _log(f"xpu_smi_install import failed: {e}")
        return None
    if not ensure_xpu_smi():
        _log("xpu-smi unavailable; memory monitoring disabled")
        return None

    _interval = max(float(interval), 0.1)
    _timeout = max(int(timeout), 1)

    tiles = _resolve_tiles_writer()
    os.environ[_TILES_ENV] = str(tiles)

    try:
        fd, path_str = tempfile.mkstemp(prefix="xpu_smi_", suffix=".log")
        os.close(fd)
    except OSError as e:
        _log(f"failed to create snapshot file: {e}")
        return None

    path = Path(path_str)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_poller_loop, args=(stop_event, path),
        name="xpu-smi-poller", daemon=True,
    )
    thread.start()
    _poller_stop = stop_event
    _poller_thread = thread
    _log_path = path
    _log(f"started poller every {_interval}s, tiles={tiles}, log={path}")
    return str(path)


def stop() -> None:
    """Stop the writer (if any) and remove its snapshot file."""
    global _poller_thread, _poller_stop, _log_path

    stop_event, _poller_stop = _poller_stop, None
    if stop_event is not None:
        stop_event.set()

    thread, _poller_thread = _poller_thread, None
    if thread is not None:
        thread.join(timeout=_timeout + 1)

    path, _log_path = _log_path, None
    if path is not None:
        with suppress(OSError):
            path.unlink(missing_ok=True)


def attach(path: str | os.PathLike[str] | None) -> None:
    """Reader-side: bind this process to an existing snapshot file."""
    global _log_path
    _log_path = Path(path) if path else None


def log_path() -> str | None:
    return str(_log_path) if _log_path else None


# --- snapshot reader ----------------------------------------------------------

def _read_snapshot() -> str:
    if _log_path is None:
        return ""
    try:
        return _log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _iter_rows(text: str, tiles: int) -> Iterator[tuple[int, float]]:
    """Yield ``(card_id, util)`` for each xpu-smi CSV row in ``text``."""
    for line in text.splitlines():
        if (m := _ROW_RE.match(line)) is None:
            continue
        row_dev, row_tile, val = m.group(1), m.group(2), m.group(3)
        try:
            dev, util = int(row_dev), float(val)
        except ValueError:
            continue
        if tiles > 1:
            if row_tile is None:
                continue
            with suppress(ValueError):
                yield dev * tiles + int(row_tile), util
        elif row_tile is None:
            yield dev, util


def _coerce_card_set(cards: Iterable[int] | None) -> set[int] | None:
    if cards is None:
        return None
    out: set[int] = set()
    for c in cards:
        with suppress(TypeError, ValueError):
            out.add(int(c))
    return out


def get_max_mem_util(
    cards: Iterable[int] | None = None,
) -> tuple[int, float] | None:
    """Return ``(card_id, util)`` with the highest utilization, or ``None``."""
    text = _read_snapshot()
    if not text:
        return None
    target = _coerce_card_set(cards)
    if cards is not None and not target:
        return None

    rows = (
        (card, util)
        for card, util in _iter_rows(text, _get_tiles_per_device())
        if target is None or card in target
    )
    return max(rows, key=lambda cu: cu[1], default=None)
