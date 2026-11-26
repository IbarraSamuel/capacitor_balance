"""SQLite persistence for sessions, tabs, capacitor stock/bank, and solve runs."""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

NUM_PHASES = 3
MAX_DIM = 24
MAX_TABS_PER_SESSION = 10
DEFAULT_MAX_CONCURRENT_USERS = 50
DEFAULT_SESSION_TTL_SECONDS = 90.0


class TooManyConnectionsError(Exception):
    """Raised when the concurrent active session limit is reached."""


class TabLimitError(Exception):
    """Raised when a session exceeds MAX_TABS_PER_SESSION tabs."""


class NotFoundError(Exception):
    """Raised when a requested tab or capacitor does not exist."""


def _serialize_tab(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    tab_id = str(row["tab_id"])
    cap_rows = conn.execute(
        "SELECT id, row_index, serial, capacitance, deviation, slot_position "
        "FROM tab_capacitors WHERE tab_id = ? ORDER BY row_index ASC",
        (tab_id,),
    ).fetchall()
    capacitors = [
        {
            "id": int(c["id"]),
            "row_index": int(c["row_index"]),
            "serial": str(c["serial"]),
            "capacitance": float(c["capacitance"]),
            "deviation": int(c["deviation"]),
            "slot_position": (
                int(c["slot_position"]) if c["slot_position"] is not None else None
            ),
        }
        for c in cap_rows
    ]
    return {
        "tab_id": tab_id,
        "title": str(row["title"]),
        "voltage_kv": int(row["voltage_kv"]),
        "series": int(row["series"]),
        "parallel": int(row["parallel"]),
        "source_filename": str(row["source_filename"]),
        "solver_used": str(row["solver_used"]),
        "last_solve_ms": (
            float(row["last_solve_ms"]) if row["last_solve_ms"] is not None else None
        ),
        "last_fitness": (
            float(row["last_fitness"]) if row["last_fitness"] is not None else None
        ),
        "capacitors": capacitors,
    }


def _require_tab(
    conn: sqlite3.Connection, session_id: str, tab_id: str
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM bank_tabs WHERE tab_id = ? AND session_id = ?",
        (tab_id, session_id),
    ).fetchone()
    if row is None:
        msg = f"Tab '{tab_id}' not found"
        raise NotFoundError(msg)
    return cast("sqlite3.Row", row)


class ServerDB:
    """Thread-safe SQLite store for all server state."""

    def __init__(
        self,
        db_path: Path,
        *,
        max_concurrent_users: int = DEFAULT_MAX_CONCURRENT_USERS,
        session_ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
    ) -> None:
        """Initialize the SQLite database and create tables if needed."""
        self.db_path = db_path
        self.max_concurrent_users = max_concurrent_users
        self.session_ttl_seconds = session_ttl_seconds
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextlib.contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection]:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS active_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    client_ip TEXT NOT NULL DEFAULT '',
                    user_agent TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS bank_tabs (
                    tab_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    voltage_kv INTEGER NOT NULL DEFAULT 115,
                    series INTEGER NOT NULL DEFAULT 6,
                    parallel INTEGER NOT NULL DEFAULT 9,
                    source_filename TEXT NOT NULL DEFAULT '',
                    solver_used TEXT NOT NULL DEFAULT '',
                    last_solve_ms REAL,
                    last_fitness REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tabs_session
                    ON bank_tabs(session_id, created_at);

                CREATE TABLE IF NOT EXISTS tab_capacitors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tab_id TEXT NOT NULL REFERENCES bank_tabs(tab_id)
                        ON DELETE CASCADE,
                    row_index INTEGER NOT NULL,
                    serial TEXT NOT NULL,
                    capacitance REAL NOT NULL,
                    deviation INTEGER NOT NULL,
                    slot_position INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_caps_tab
                    ON tab_capacitors(tab_id, slot_position, row_index);

                CREATE TABLE IF NOT EXISTS solve_runs (
                    run_id TEXT PRIMARY KEY,
                    tab_id TEXT NOT NULL REFERENCES bank_tabs(tab_id)
                        ON DELETE CASCADE,
                    session_id TEXT NOT NULL,
                    solver_id TEXT NOT NULL,
                    series INTEGER NOT NULL,
                    parallel INTEGER NOT NULL,
                    fitness REAL NOT NULL,
                    solve_ms REAL NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )

    def acquire_or_touch_session(
        self,
        session_id: str | None,
        *,
        client_ip: str = "",
        user_agent: str = "",
    ) -> tuple[str, int]:
        """Validate/touch a session or allocate a new one under the concurrency cap.

        Returns:
            Tuple of (session_id, active_user_count).

        Raises:
            TooManyConnectionsError: If active sessions already equal the limit.
        """
        now = time.time()
        cutoff = now - self.session_ttl_seconds
        with self._transaction() as conn:
            conn.execute(
                "DELETE FROM active_sessions WHERE last_seen_at < ?", (cutoff,)
            )
            if session_id is not None and bool(session_id):
                existing = conn.execute(
                    "SELECT session_id FROM active_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if existing is not None:
                    conn.execute(
                        "UPDATE active_sessions SET last_seen_at = ? "
                        "WHERE session_id = ?",
                        (now, session_id),
                    )
                    count = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM active_sessions"
                        ).fetchone()[0]
                    )
                    return session_id, count

            count = int(
                conn.execute("SELECT COUNT(*) FROM active_sessions").fetchone()[0]
            )
            if count >= self.max_concurrent_users:
                msg = (
                    f"Too many connections: maximum of "
                    f"{self.max_concurrent_users} concurrent users reached."
                )
                raise TooManyConnectionsError(msg)

            new_id = (
                session_id
                if session_id is not None and bool(session_id)
                else uuid.uuid4().hex
            )
            conn.execute(
                "INSERT INTO active_sessions "
                "(session_id, created_at, last_seen_at, client_ip, user_agent) "
                "VALUES (?, ?, ?, ?, ?)",
                (new_id, now, now, client_ip, user_agent[:200]),
            )
            return new_id, count + 1

    def release_session(self, session_id: str) -> None:
        """Remove a session from active_sessions on disconnect."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM active_sessions WHERE session_id = ?", (session_id,)
            )

    def active_session_count(self) -> int:
        """Return the number of non-expired active sessions.

        Returns:
            Current active concurrent user count.
        """
        cutoff = time.time() - self.session_ttl_seconds
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM active_sessions WHERE last_seen_at < ?", (cutoff,)
            )
            return int(
                conn.execute("SELECT COUNT(*) FROM active_sessions").fetchone()[0]
            )

    def list_tabs(self, session_id: str) -> list[dict[str, Any]]:
        """Return all tabs for a session, creating a default tab if none exist.

        Returns:
            Ordered list of serialized bank tabs.
        """
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM bank_tabs WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
            if not rows:
                now = time.time()
                tab_id = uuid.uuid4().hex[:12]
                conn.execute(
                    "INSERT INTO bank_tabs "
                    "(tab_id, session_id, title, voltage_kv, series, parallel, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (tab_id, session_id, "Bank 1", 115, 6, 9, now, now),
                )
                rows = conn.execute(
                    "SELECT * FROM bank_tabs WHERE session_id = ? "
                    "ORDER BY created_at ASC",
                    (session_id,),
                ).fetchall()
            return [_serialize_tab(conn, r) for r in rows]

    def get_tab(self, session_id: str, tab_id: str) -> dict[str, Any]:
        """Fetch a single tab owned by the session.

        Returns:
            Serialized tab dictionary.
        """
        with self._lock, self._connect() as conn:
            row = _require_tab(conn, session_id, tab_id)
            return _serialize_tab(conn, row)

    def create_tab(
        self,
        session_id: str,
        *,
        title: str | None = None,
        voltage_kv: int = 115,
        series: int = 6,
        parallel: int = 9,
    ) -> dict[str, Any]:
        """Create a new capacitor bank tab up to MAX_TABS_PER_SESSION.

        Returns:
            Newly created serialized tab dictionary.

        Raises:
            ValueError: If series or parallel are out of bounds.
            TabLimitError: If session already has MAX_TABS_PER_SESSION tabs.
        """
        if series < 1 or parallel < 1 or series > MAX_DIM or parallel > MAX_DIM:
            msg = f"Series and parallel must be between 1 and {MAX_DIM}"
            raise ValueError(msg)
        with self._transaction() as conn:
            count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM bank_tabs WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
            )
            if count >= MAX_TABS_PER_SESSION:
                msg = f"Maximum of {MAX_TABS_PER_SESSION} tabs reached"
                raise TabLimitError(msg)
            now = time.time()
            tab_id = uuid.uuid4().hex[:12]
            tab_title = (title or f"Bank {count + 1}").strip()[:40] or "Bank"
            conn.execute(
                "INSERT INTO bank_tabs "
                "(tab_id, session_id, title, voltage_kv, series, parallel, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (tab_id, session_id, tab_title, voltage_kv, series, parallel, now, now),
            )
            row = conn.execute(
                "SELECT * FROM bank_tabs WHERE tab_id = ?", (tab_id,)
            ).fetchone()
            return _serialize_tab(conn, row)

    def update_tab_config(  # ruff: ignore[too-many-arguments]
        self,
        session_id: str,
        tab_id: str,
        *,
        title: str | None = None,
        voltage_kv: int | None = None,
        series: int | None = None,
        parallel: int | None = None,
    ) -> dict[str, Any]:
        """Update tab metadata and dimensions, returning out-of-range caps to stock.

        Returns:
            Updated serialized tab dictionary.

        Raises:
            ValueError: If series or parallel are out of bounds.
        """
        with self._transaction() as conn:
            row = _require_tab(conn, session_id, tab_id)
            new_title = (
                title.strip()[:40]
                if title is not None and title.strip()
                else str(row["title"])
            )
            new_v = voltage_kv if voltage_kv is not None else int(row["voltage_kv"])
            new_s = series if series is not None else int(row["series"])
            new_p = parallel if parallel is not None else int(row["parallel"])
            if new_s < 1 or new_p < 1 or new_s > MAX_DIM or new_p > MAX_DIM:
                msg = f"Series and parallel must be between 1 and {MAX_DIM}"
                raise ValueError(msg)

            max_slots = NUM_PHASES * new_s * new_p
            conn.execute(
                "UPDATE bank_tabs SET title = ?, voltage_kv = ?, series = ?, "
                "parallel = ?, updated_at = ? WHERE tab_id = ?",
                (new_title, new_v, new_s, new_p, time.time(), tab_id),
            )
            conn.execute(
                "UPDATE tab_capacitors SET slot_position = NULL "
                "WHERE tab_id = ? AND slot_position >= ?",
                (tab_id, max_slots),
            )
            updated = conn.execute(
                "SELECT * FROM bank_tabs WHERE tab_id = ?", (tab_id,)
            ).fetchone()
            return _serialize_tab(conn, updated)

    def delete_tab(self, session_id: str, tab_id: str) -> list[dict[str, Any]]:
        """Delete a tab and return the remaining tabs for the session.

        Returns:
            Remaining serialized tabs (creating a fresh default tab if empty).

        Raises:
            NotFoundError: If the tab does not exist for this session.
        """
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM bank_tabs WHERE tab_id = ? AND session_id = ?",
                (tab_id, session_id),
            )
            if cur.rowcount == 0:
                msg = f"Tab '{tab_id}' not found"
                raise NotFoundError(msg)
        return self.list_tabs(session_id)

    def set_tab_inventory(
        self,
        session_id: str,
        tab_id: str,
        *,
        filename: str,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Store uploaded capacitors with empty bank (all in stock).

        Returns:
            Updated serialized tab dictionary.
        """
        with self._transaction() as conn:
            _require_tab(conn, session_id, tab_id)
            conn.execute("DELETE FROM tab_capacitors WHERE tab_id = ?", (tab_id,))
            conn.executemany(
                "INSERT INTO tab_capacitors "
                "(tab_id, row_index, serial, capacitance, deviation, slot_position) "
                "VALUES (?, ?, ?, ?, ?, NULL)",
                [
                    (
                        tab_id,
                        idx,
                        str(rec["serial"]),
                        float(rec["capacitance"]),
                        int(rec["deviation"]),
                    )
                    for idx, rec in enumerate(records)
                ],
            )
            conn.execute(
                "UPDATE bank_tabs SET source_filename = ?, solver_used = '', "
                "last_solve_ms = NULL, last_fitness = NULL, updated_at = ? "
                "WHERE tab_id = ?",
                (filename[:120], time.time(), tab_id),
            )
            updated = conn.execute(
                "SELECT * FROM bank_tabs WHERE tab_id = ?", (tab_id,)
            ).fetchone()
            return _serialize_tab(conn, updated)

    def apply_solution(  # ruff: ignore[too-many-arguments]
        self,
        session_id: str,
        tab_id: str,
        *,
        solver_id: str,
        ordered_row_indices: list[int],
        fitness: float,
        solve_ms: float,
    ) -> dict[str, Any]:
        """Persist solver slot assignments and log the solve run in SQLite.

        Returns:
            Updated serialized tab dictionary.
        """
        with self._transaction() as conn:
            row = _require_tab(conn, session_id, tab_id)
            conn.execute(
                "UPDATE tab_capacitors SET slot_position = NULL WHERE tab_id = ?",
                (tab_id,),
            )
            conn.executemany(
                "UPDATE tab_capacitors SET slot_position = ? "
                "WHERE tab_id = ? AND row_index = ?",
                [
                    (slot_pos, tab_id, row_idx)
                    for slot_pos, row_idx in enumerate(ordered_row_indices)
                ],
            )
            now = time.time()
            conn.execute(
                "UPDATE bank_tabs SET solver_used = ?, last_solve_ms = ?, "
                "last_fitness = ?, updated_at = ? WHERE tab_id = ?",
                (solver_id, solve_ms, fitness, now, tab_id),
            )
            conn.execute(
                "INSERT INTO solve_runs "
                "(run_id, tab_id, session_id, solver_id, series, parallel, "
                "fitness, solve_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    uuid.uuid4().hex[:12],
                    tab_id,
                    session_id,
                    solver_id,
                    int(row["series"]),
                    int(row["parallel"]),
                    fitness,
                    solve_ms,
                    now,
                ),
            )
            updated = conn.execute(
                "SELECT * FROM bank_tabs WHERE tab_id = ?", (tab_id,)
            ).fetchone()
            return _serialize_tab(conn, updated)

    def place_capacitor(
        self,
        session_id: str,
        tab_id: str,
        *,
        cap_id: int,
        slot_position: int,
    ) -> dict[str, Any]:
        """Place a capacitor into a bank slot (returning any previous occupant).

        Returns:
            Updated serialized tab dictionary.

        Raises:
            ValueError: If slot_position is out of bounds.
            NotFoundError: If cap_id does not exist in the tab.
        """
        with self._transaction() as conn:
            tab = _require_tab(conn, session_id, tab_id)
            max_slots = NUM_PHASES * int(tab["series"]) * int(tab["parallel"])
            if slot_position < 0 or slot_position >= max_slots:
                msg = f"slot_position must be in [0, {max_slots - 1}]"
                raise ValueError(msg)

            cap = conn.execute(
                "SELECT id FROM tab_capacitors WHERE id = ? AND tab_id = ?",
                (cap_id, tab_id),
            ).fetchone()
            if cap is None:
                msg = f"Capacitor {cap_id} not found in tab"
                raise NotFoundError(msg)

            conn.execute(
                "UPDATE tab_capacitors SET slot_position = NULL "
                "WHERE tab_id = ? AND slot_position = ?",
                (tab_id, slot_position),
            )
            conn.execute(
                "UPDATE tab_capacitors SET slot_position = ? WHERE id = ?",
                (slot_position, cap_id),
            )
            conn.execute(
                "UPDATE bank_tabs SET updated_at = ? WHERE tab_id = ?",
                (time.time(), tab_id),
            )
            updated = conn.execute(
                "SELECT * FROM bank_tabs WHERE tab_id = ?", (tab_id,)
            ).fetchone()
            return _serialize_tab(conn, updated)

    def remove_capacitor(
        self,
        session_id: str,
        tab_id: str,
        *,
        cap_id: int | None = None,
        slot_position: int | None = None,
    ) -> dict[str, Any]:
        """Move a capacitor from the bank back to stock.

        Returns:
            Updated serialized tab dictionary.

        Raises:
            ValueError: If neither cap_id nor slot_position is provided.
        """
        with self._transaction() as conn:
            _require_tab(conn, session_id, tab_id)
            if cap_id is not None:
                conn.execute(
                    "UPDATE tab_capacitors SET slot_position = NULL "
                    "WHERE id = ? AND tab_id = ?",
                    (cap_id, tab_id),
                )
            elif slot_position is not None:
                conn.execute(
                    "UPDATE tab_capacitors SET slot_position = NULL "
                    "WHERE slot_position = ? AND tab_id = ?",
                    (slot_position, tab_id),
                )
            else:
                msg = "Provide either cap_id or slot_position"
                raise ValueError(msg)

            conn.execute(
                "UPDATE bank_tabs SET updated_at = ? WHERE tab_id = ?",
                (time.time(), tab_id),
            )
            updated = conn.execute(
                "SELECT * FROM bank_tabs WHERE tab_id = ?", (tab_id,)
            ).fetchone()
            return _serialize_tab(conn, updated)

    def swap_capacitors(
        self,
        session_id: str,
        tab_id: str,
        *,
        cap_id_a: int,
        cap_id_b: int,
    ) -> dict[str, Any]:
        """Swap slot positions of two capacitors (Bank<->Bank or Bank<->Stock).

        Returns:
            Updated serialized tab dictionary.

        Raises:
            NotFoundError: If either capacitor does not exist in the tab.
        """
        with self._transaction() as conn:
            _require_tab(conn, session_id, tab_id)
            a = conn.execute(
                "SELECT id, slot_position FROM tab_capacitors "
                "WHERE id = ? AND tab_id = ?",
                (cap_id_a, tab_id),
            ).fetchone()
            b = conn.execute(
                "SELECT id, slot_position FROM tab_capacitors "
                "WHERE id = ? AND tab_id = ?",
                (cap_id_b, tab_id),
            ).fetchone()
            if a is None or b is None:
                msg = "One or both capacitors not found in tab"
                raise NotFoundError(msg)

            conn.execute(
                "UPDATE tab_capacitors SET slot_position = ? WHERE id = ?",
                (b["slot_position"], cap_id_a),
            )
            conn.execute(
                "UPDATE tab_capacitors SET slot_position = ? WHERE id = ?",
                (a["slot_position"], cap_id_b),
            )
            conn.execute(
                "UPDATE bank_tabs SET updated_at = ? WHERE tab_id = ?",
                (time.time(), tab_id),
            )
            updated = conn.execute(
                "SELECT * FROM bank_tabs WHERE tab_id = ?", (tab_id,)
            ).fetchone()
            return _serialize_tab(conn, updated)

    def clear_bank(self, session_id: str, tab_id: str) -> dict[str, Any]:
        """Return all placed capacitors in a tab back to stock.

        Returns:
            Updated serialized tab dictionary.
        """
        with self._transaction() as conn:
            _require_tab(conn, session_id, tab_id)
            conn.execute(
                "UPDATE tab_capacitors SET slot_position = NULL WHERE tab_id = ?",
                (tab_id,),
            )
            conn.execute(
                "UPDATE bank_tabs SET solver_used = '', last_solve_ms = NULL, "
                "last_fitness = NULL, updated_at = ? WHERE tab_id = ?",
                (time.time(), tab_id),
            )
            updated = conn.execute(
                "SELECT * FROM bank_tabs WHERE tab_id = ?", (tab_id,)
            ).fetchone()
            return _serialize_tab(conn, updated)
