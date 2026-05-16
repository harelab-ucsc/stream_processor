"""
Threading race condition tests for the DB writer pattern in stream_processor.

Tests are self-contained — no ROS2 required. They verify:
  1. The old pattern (N concurrent sqlite3.connect() calls) reliably produces
     "database is locked" errors under contention.
  2. The new pattern (queue + single writer thread) produces zero errors and
     writes every row even under heavy concurrent producer load.
  3. Pending rows queued just before shutdown are all flushed before exit.
  4. Queue depth accurately reflects backlog when the writer is slower than
     producers.
"""

import os
import queue
import sqlite3
import tempfile
import threading
import time


# ---------------------------------------------------------------------------
# Shared DB helpers
# ---------------------------------------------------------------------------

DDL = (
    "CREATE TABLE IF NOT EXISTS frames "
    "(id INTEGER PRIMARY KEY AUTOINCREMENT, val TEXT UNIQUE)"
)
INSERT = "INSERT OR IGNORE INTO frames (val) VALUES (?);"


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.execute(DDL)
    conn.commit()
    conn.close()


def _row_count(path):
    conn = sqlite3.connect(path)
    n = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    conn.close()
    return n


# ---------------------------------------------------------------------------
# Minimal replica of the _db_writer pattern from stream_processor.py
# ---------------------------------------------------------------------------

class SingleWriterDB:
    """Queue-backed SQLite writer — one connection, one thread, WAL mode."""

    def __init__(self, db_path):
        self._db_queue = queue.Queue()
        self.errors = []

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(DDL)
        conn.commit()
        conn.close()

        self._thread = threading.Thread(
            target=self._writer, args=(db_path,), daemon=True
        )
        self._thread.start()

    def _writer(self, db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        while True:
            item = self._db_queue.get()
            if item is None:
                self._db_queue.task_done()
                break
            try:
                cursor.execute(INSERT, (item,))
                conn.commit()
            except Exception as e:
                self.errors.append(str(e))
            finally:
                self._db_queue.task_done()
        conn.close()

    def put(self, val):
        self._db_queue.put(val)

    def qsize(self):
        return self._db_queue.qsize()

    def shutdown(self):
        self._db_queue.put(None)
        self._db_queue.join()


# ---------------------------------------------------------------------------
# 1. Old pattern — concurrent connections race on the write lock
# ---------------------------------------------------------------------------

def test_old_pattern_produces_lock_errors():
    """
    N threads vs. a held exclusive lock must produce 'database is locked'.

    A long-running writer holds the write lock while N competing threads each
    open their own sqlite3.connect() with a short timeout. This mirrors the
    8-worker-thread scenario from the flight log.
    """
    N = 8
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "race.db")
        _make_db(db_path)

        errors = []
        lock_held = threading.Event()
        release_lock = threading.Event()

        # Background thread acquires an exclusive write lock and holds it.
        def lock_holder():
            conn = sqlite3.connect(db_path)
            conn.execute("BEGIN EXCLUSIVE")
            lock_held.set()
            release_lock.wait()
            conn.rollback()
            conn.close()

        holder = threading.Thread(target=lock_holder, daemon=True)
        holder.start()
        lock_held.wait()  # ensure the lock is held before competitors start

        barrier = threading.Barrier(N)

        def competitor(i):
            barrier.wait()
            try:
                conn = sqlite3.connect(db_path, timeout=0.05)
                conn.execute(INSERT, (f"row_{i}",))
                conn.commit()
                conn.close()
            except sqlite3.OperationalError as e:
                errors.append(str(e))

        threads = [threading.Thread(target=competitor, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        release_lock.set()
        holder.join()

        assert len(errors) > 0, (
            "Expected 'database is locked' errors when competing against a "
            "held exclusive lock, but none occurred."
        )


# ---------------------------------------------------------------------------
# 2. New pattern — zero errors, all rows present
# ---------------------------------------------------------------------------

def test_new_pattern_no_lock_errors():
    """Single writer thread: zero database-is-locked errors under N producers."""
    N = 24
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "safe.db")
        db = SingleWriterDB(db_path)

        barrier = threading.Barrier(N)

        def producer(i):
            barrier.wait()
            db.put(f"row_{i}")

        threads = [threading.Thread(target=producer, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        db.shutdown()

        assert db.errors == [], f"Unexpected DB errors: {db.errors}"
        assert _row_count(db_path) == N, (
            f"Expected {N} rows, got {_row_count(db_path)}"
        )


# ---------------------------------------------------------------------------
# 3. Heavy contention — every row from 50 producers must land in the DB
# ---------------------------------------------------------------------------

def test_all_rows_written_under_heavy_contention():
    """50 simultaneous producers — no row may be silently dropped."""
    N = 50
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "heavy.db")
        db = SingleWriterDB(db_path)

        barrier = threading.Barrier(N)

        def producer(i):
            barrier.wait()
            db.put(f"frame_{i:04d}")

        threads = [threading.Thread(target=producer, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        db.shutdown()

        assert db.errors == []
        assert _row_count(db_path) == N, (
            f"Expected {N} rows after heavy contention, got {_row_count(db_path)}"
        )


# ---------------------------------------------------------------------------
# 4. Shutdown drains all pending items before the writer exits
# ---------------------------------------------------------------------------

def test_shutdown_drains_pending_items():
    """Items queued just before the shutdown sentinel must all be written."""
    N = 30
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "drain.db")
        db = SingleWriterDB(db_path)

        for i in range(N):
            db.put(f"pending_{i}")

        db.shutdown()  # sentinel is enqueued *after* all N items

        assert _row_count(db_path) == N, (
            f"Expected {N} rows after drain, got {_row_count(db_path)}"
        )


# ---------------------------------------------------------------------------
# 5. Queue depth reflects real backlog when writer is slower than producers
# ---------------------------------------------------------------------------

def test_queue_depth_reflects_backlog():
    """Queue depth must reach > 1 when producers outpace the writer."""
    N = 40
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "backlog.db")

        slow_q = queue.Queue()
        depths = []

        def slow_writer():
            conn = sqlite3.connect(db_path)
            conn.execute(DDL)
            conn.commit()
            while True:
                item = slow_q.get()
                if item is None:
                    slow_q.task_done()
                    break
                time.sleep(0.015)  # deliberate slow I/O
                try:
                    conn.execute(INSERT, (item,))
                    conn.commit()
                except Exception:
                    pass
                finally:
                    slow_q.task_done()
            conn.close()

        t = threading.Thread(target=slow_writer, daemon=True)
        t.start()

        for i in range(N):
            slow_q.put(f"item_{i}")

        for _ in range(6):
            depths.append(slow_q.qsize())
            time.sleep(0.025)

        slow_q.put(None)
        slow_q.join()

        assert any(d > 1 for d in depths), (
            f"Queue never built a backlog — depths sampled: {depths}"
        )
