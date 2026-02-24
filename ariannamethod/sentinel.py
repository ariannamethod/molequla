"""
sentinel.py — DNA watcher for molequla ecology.

Watches the dna/ directory for filesystem changes.
Translates events into AML field effects via sentinel.aml.
Uses Blood-compiled C hash when available, falls back to Python.

Usage:
    from ariannamethod import Sentinel

    s = Sentinel("dna/", aml_path="dna/sentinel.aml")
    changes = s.scan()
    # changes = [{"path": "dna/incoming/food.txt", "event": "new", "size": 1234}, ...]

Part of molequla. The sentinel that watches the living field.
"""

import ctypes
import hashlib
import os
import time
from pathlib import Path


# Event types
EVENT_NEW = "new"
EVENT_MODIFIED = "modified"
EVENT_DELETED = "deleted"


class WatchedFile:
    """Snapshot of a watched file."""
    __slots__ = ("path", "mtime", "size", "hash")

    def __init__(self, path, mtime, size, file_hash=None):
        self.path = path
        self.mtime = mtime
        self.size = size
        self.hash = file_hash


class FileChange:
    """A detected change in the DNA directory."""
    __slots__ = ("path", "event", "size", "element")

    def __init__(self, path, event, size=0, element=None):
        self.path = path
        self.event = event
        self.size = size
        self.element = element  # which element's territory (from path)

    def as_dict(self):
        return {
            "path": self.path,
            "event": self.event,
            "size": self.size,
            "element": self.element,
        }


def _detect_element(path):
    """Detect element from path: dna/output/fire/... → 'fire'."""
    parts = Path(path).parts
    for i, p in enumerate(parts):
        if p == "output" and i + 1 < len(parts):
            elem = parts[i + 1]
            if elem in ("earth", "air", "water", "fire"):
                return elem
    return None


def _detect_zone(path):
    """Detect zone from path: incoming, shared, output."""
    parts = Path(path).parts
    for p in parts:
        if p in ("incoming", "shared", "output"):
            return p
    return "unknown"


class Sentinel:
    """
    DNA watcher — the sentinel operator.

    Watches dna/ directory tree for changes.
    Reports new, modified, and deleted files.
    Triggers AML field reactions when libaml is available.
    """

    def __init__(self, dna_path, aml_path=None, lib=None):
        """
        Args:
            dna_path: root of dna/ directory
            aml_path: path to sentinel.aml (optional, for field reactions)
            lib: loaded libaml ctypes handle (optional, for Blood + AML execution)
        """
        self.dna_path = str(dna_path)
        self.aml_path = aml_path
        self.lib = lib
        self._watched = {}  # path → WatchedFile
        self._changes = []  # latest scan results
        self._scan_count = 0
        self._total_new = 0
        self._total_modified = 0
        self._total_deleted = 0
        self._last_scan_time = 0.0
        self._blood_hash = None  # Blood-compiled hash function

        # Load sentinel.aml if lib available
        if self.lib is not None and self.aml_path:
            self._load_aml()

        # Try to get Blood-compiled hash function
        if self.lib is not None:
            self._init_blood_hash()

        # Initial scan — populate watched state without reporting changes
        self._bootstrap()

    def _load_aml(self):
        """Load sentinel.aml via am_exec_file."""
        if not self.aml_path or not os.path.exists(self.aml_path):
            return
        try:
            # Bind am_exec_file if not yet bound
            if not hasattr(self.lib, '_sentinel_exec_bound'):
                self.lib.am_exec_file.restype = ctypes.c_int
                self.lib.am_exec_file.argtypes = [ctypes.c_char_p]
                self.lib._sentinel_exec_bound = True
            rc = self.lib.am_exec_file(self.aml_path.encode())
            if rc != 0:
                err = self.lib.am_get_error()
                if err:
                    print(f"[sentinel] aml load warning: {err}")
        except Exception as e:
            print(f"[sentinel] aml load failed: {e}")

    def _init_blood_hash(self):
        """Try to get Blood-compiled sentinel_hash_file function."""
        try:
            if not hasattr(self.lib, '_sentinel_blood_bound'):
                self.lib.am_blood_count.restype = ctypes.c_int
                self.lib.am_blood_count.argtypes = []
                self.lib.am_blood_sym.restype = ctypes.c_void_p
                self.lib.am_blood_sym.argtypes = [ctypes.c_int, ctypes.c_char_p]
                self.lib._sentinel_blood_bound = True

            n = self.lib.am_blood_count()
            for i in range(n):
                ptr = self.lib.am_blood_sym(i, b"sentinel_hash_file")
                if ptr:
                    # Cast to function: uint64 sentinel_hash_file(const char*)
                    HASH_FUNC = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_char_p)
                    self._blood_hash = HASH_FUNC(ptr)
                    break
        except Exception:
            pass  # Blood not available, use Python fallback

    def _hash_file(self, path):
        """Hash a file. Uses Blood C function if available, else Python FNV-1a."""
        if self._blood_hash is not None:
            try:
                return self._blood_hash(path.encode())
            except Exception:
                pass

        # Python fallback: FNV-1a (matches Blood implementation)
        h = 14695981039346656037
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    for b in chunk:
                        h ^= b
                        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        except (OSError, IOError):
            return 0
        return h

    def _bootstrap(self):
        """Initial scan — learn the current state without reporting changes."""
        for dirpath, dirnames, filenames in os.walk(self.dna_path):
            # Skip hidden dirs
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
            for fname in filenames:
                if fname.startswith('.') or fname.endswith('.aml'):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    st = os.stat(fpath)
                    self._watched[fpath] = WatchedFile(
                        path=fpath,
                        mtime=st.st_mtime,
                        size=st.st_size,
                    )
                except OSError:
                    pass

    def scan(self):
        """
        Scan dna/ directory for changes since last scan.

        Returns list of FileChange objects.
        """
        self._changes = []
        now = time.time()
        seen = set()

        for dirpath, dirnames, filenames in os.walk(self.dna_path):
            dirnames[:] = [d for d in dirnames if not d.startswith('.')]
            for fname in filenames:
                if fname.startswith('.'):
                    continue
                # Skip .aml files from scanning (they're code, not food)
                if fname.endswith('.aml'):
                    continue
                fpath = os.path.join(dirpath, fname)
                seen.add(fpath)

                try:
                    st = os.stat(fpath)
                except OSError:
                    continue

                if fpath in self._watched:
                    # Known file — check for modification
                    w = self._watched[fpath]
                    if w.mtime != st.st_mtime or w.size != st.st_size:
                        w.mtime = st.st_mtime
                        w.size = st.st_size
                        change = FileChange(
                            path=fpath,
                            event=EVENT_MODIFIED,
                            size=st.st_size,
                            element=_detect_element(fpath),
                        )
                        self._changes.append(change)
                        self._total_modified += 1
                else:
                    # New file
                    self._watched[fpath] = WatchedFile(
                        path=fpath,
                        mtime=st.st_mtime,
                        size=st.st_size,
                    )
                    change = FileChange(
                        path=fpath,
                        event=EVENT_NEW,
                        size=st.st_size,
                        element=_detect_element(fpath),
                    )
                    self._changes.append(change)
                    self._total_new += 1

        # Check for deleted files
        deleted = set(self._watched.keys()) - seen
        for fpath in deleted:
            change = FileChange(
                path=fpath,
                event=EVENT_DELETED,
                size=0,
                element=_detect_element(fpath),
            )
            self._changes.append(change)
            self._total_deleted += 1
            del self._watched[fpath]

        self._scan_count += 1
        self._last_scan_time = now

        # Trigger AML field reactions
        if self._changes and self.lib is not None:
            self._react(self._changes)

        return self._changes

    def _react(self, changes):
        """Trigger AML field reactions based on changes."""
        if self.lib is None:
            return

        n_new = sum(1 for c in changes if c.event == EVENT_NEW)
        n_mod = sum(1 for c in changes if c.event == EVENT_MODIFIED)
        n_del = sum(1 for c in changes if c.event == EVENT_DELETED)
        total = len(changes)

        try:
            # Overload protection
            if total > 20:
                self.lib.am_exec(b"on_overload()")
                return

            # New DNA — hunger stimulus
            if n_new > 0:
                intensity = min(1.0, n_new * 0.2)
                self.lib.am_exec(f"on_new_dna({intensity:.2f})".encode())

            # Check zones
            for c in changes:
                zone = _detect_zone(c.path)
                if zone == "shared" and c.event in (EVENT_NEW, EVENT_MODIFIED):
                    strength = min(1.0, c.size / 50000.0)
                    self.lib.am_exec(f"on_shared({strength:.2f})".encode())
                elif zone == "output" and c.event == EVENT_NEW:
                    self.lib.am_exec(b"on_output(0)")

        except Exception as e:
            print(f"[sentinel] react error: {e}")

    def changes(self):
        """Return latest changes from last scan."""
        return self._changes

    def watched_count(self):
        """Number of files being tracked."""
        return len(self._watched)

    def stats(self):
        """Sentinel statistics."""
        return {
            "scans": self._scan_count,
            "watched": len(self._watched),
            "total_new": self._total_new,
            "total_modified": self._total_modified,
            "total_deleted": self._total_deleted,
            "last_scan": self._last_scan_time,
            "blood_hash": self._blood_hash is not None,
        }

    def status_line(self):
        """One-line status for mycelium display."""
        s = self.stats()
        blood = "C" if s["blood_hash"] else "py"
        n_changes = len(self._changes)
        change_str = f" ({n_changes} changes)" if n_changes > 0 else ""
        return (f"[sentinel] watching={s['watched']} scans={s['scans']} "
                f"new={s['total_new']} mod={s['total_modified']} "
                f"del={s['total_deleted']} hash={blood}{change_str}")

    def report(self):
        """Detailed report of latest changes."""
        if not self._changes:
            return "[sentinel] no changes detected."

        lines = [f"[sentinel] {len(self._changes)} changes:"]
        for c in self._changes:
            zone = _detect_zone(c.path)
            elem = f" [{c.element}]" if c.element else ""
            fname = os.path.basename(c.path)
            if c.event == EVENT_NEW:
                lines.append(f"  + {zone}/{fname}{elem} ({c.size} bytes)")
            elif c.event == EVENT_MODIFIED:
                lines.append(f"  ~ {zone}/{fname}{elem} ({c.size} bytes)")
            elif c.event == EVENT_DELETED:
                lines.append(f"  - {zone}/{fname}{elem}")

        return "\n".join(lines)
