"""Lightweight Python HTTP server for the Capacitor Bank Balancer web app."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast, override
from urllib.parse import urlparse

from server.db import (
    DEFAULT_MAX_CONCURRENT_USERS,
    DEFAULT_SESSION_TTL_SECONDS,
    MAX_TABS_PER_SESSION,
    NotFoundError,
    ServerDB,
    TabLimitError,
    TooManyConnectionsError,
)
from server.parser import (
    REQUIRED_COLUMNS,
    ValidationError,
    build_template_csv,
    build_template_xlsx,
    parse_and_validate_upload,
)
from server.solvers import SOLVER_CATALOG, load_sample_records, run_solver

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_FILE = REPO_ROOT / "visualizer" / "index.html"
DEFAULT_DB_PATH = REPO_ROOT / "server" / "app_state.db"
MAX_BODY_BYTES = 15 * 1024 * 1024  # 15 MB

_TAB_ROUTE_RE = re.compile(r"^/api/tabs/([A-Za-z0-9_-]+)(?:/([a-z]+))?$")

_TOO_MANY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Too Many Connections — Capacitor Bank Balancer</title>
  <style>
    body {
      margin: 0; min-height: 100vh; display: grid; place-items: center;
      background: #080d14; color: #e6eef8;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .card {
      max-width: 480px; background: #0f1724; border: 1px solid #ef4444;
      border-radius: 14px; padding: 28px; text-align: center;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.55);
    }
    .code {
      font-family: ui-monospace, monospace; font-size: 13px; color: #fca5a5;
      background: rgba(239, 68, 68, 0.15); display: inline-block;
      padding: 4px 10px; border-radius: 999px; margin-bottom: 12px;
    }
    h1 { margin: 0 0 10px; font-size: 22px; }
    p { color: #94a3b8; line-height: 1.5; margin: 0 0 18px; }
    button {
      background: #1e293b; border: 1px solid #38bdf8; color: #fff;
      padding: 9px 18px; border-radius: 8px; cursor: pointer; font-weight: 600;
    }
    button:hover { background: #26354d; }
  </style>
</head>
<body>
  <div class="card">
    <div class="code">HTTP 503 &bull; CONNECTION LIMIT REACHED</div>
    <h1>Too Many Connections</h1>
    <p>
      This Capacitor Bank Balancer server is currently at its maximum capacity of
      <strong>50 concurrent users</strong>. Please wait a moment and try again.
    </p>
    <button onclick="location.reload()">Retry Connection</button>
  </div>
</body>
</html>
"""


class BalancerHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer carrying shared SQLite state."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        db: ServerDB,
        ui_path: Path = UI_FILE,
    ) -> None:
        """Initialize the HTTP server with SQLite DB and UI file path."""
        self.db = db
        self.ui_path = ui_path
        super().__init__(server_address, BalancerRequestHandler)


class BalancerRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the static UI and JSON REST API."""

    @property
    def _balancer_server(self) -> BalancerHTTPServer:
        return cast("BalancerHTTPServer", self.server)

    @override
    def log_message(self, format: str, *args: object) -> None:
        """Suppress HTTP access logs when CAP_BAL_QUIET=1."""
        if os.environ.get("CAP_BAL_QUIET") == "1":
            return
        super().log_message(format, *args)

    def _extract_session_id(self) -> str | None:
        header_sid = self.headers.get("X-Session-Id")
        if header_sid is not None and header_sid.strip():
            return header_sid.strip()
        cookie_hdr = self.headers.get("Cookie")
        if cookie_hdr is not None and cookie_hdr.strip():
            cookie = SimpleCookie()
            cookie.load(cookie_hdr)
            morsel = cookie.get("cb_session")
            if morsel is not None:
                val = str(morsel.value).strip()
                if val:
                    return val
        return None

    def _send_bytes(
        self,
        status: int,
        payload: bytes,
        content_type: str,
        *,
        session_id: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        if session_id is not None and bool(session_id):
            self.send_header(
                "Set-Cookie",
                f"cb_session={session_id}; Path=/; HttpOnly; SameSite=Lax",
            )
            self.send_header("X-Session-Id", session_id)
        if extra_headers is not None:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(
        self,
        status: int,
        data: dict[str, Any],
        *,
        session_id: str | None = None,
    ) -> None:
        body = json.dumps(data).encode("utf-8")
        self._send_bytes(
            status,
            body,
            "application/json; charset=utf-8",
            session_id=session_id,
        )

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            msg = "Request payload exceeds 15 MB limit."
            raise ValueError(msg)
        raw = self.rfile.read(length)
        if not raw:
            return {}
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            msg = "JSON request body must be an object."
            raise TypeError(msg)
        return parsed

    def _acquire_session_or_reject(self, *, is_html: bool) -> tuple[str, int] | None:
        db = self._balancer_server.db
        try:
            return db.acquire_or_touch_session(
                self._extract_session_id(),
                client_ip=str(self.client_address[0]),
                user_agent=self.headers.get("User-Agent", ""),
            )
        except TooManyConnectionsError as exc:
            if is_html:
                self._send_bytes(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    _TOO_MANY_HTML.encode("utf-8"),
                    "text/html; charset=utf-8",
                )
            else:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {
                        "error": str(exc),
                        "code": "TOO_MANY_CONNECTIONS",
                        "active_users": db.max_concurrent_users,
                        "max_users": db.max_concurrent_users,
                    },
                )
            return None

    def _dispatch_api_get(
        self, path: str, session_id: str, active_users: int
    ) -> None:
        db = self._balancer_server.db
        if path == "/api/state":
            self._send_json(
                HTTPStatus.OK,
                {
                    "session_id": session_id,
                    "active_users": active_users,
                    "max_users": db.max_concurrent_users,
                    "max_tabs": MAX_TABS_PER_SESSION,
                    "required_columns": list(REQUIRED_COLUMNS),
                    "solvers": SOLVER_CATALOG,
                    "tabs": db.list_tabs(session_id),
                },
                session_id=session_id,
            )
        elif path == "/api/template.csv":
            self._send_bytes(
                HTTPStatus.OK,
                build_template_csv(load_sample_records()),
                "text/csv; charset=utf-8",
                session_id=session_id,
                extra_headers={
                    "Content-Disposition": (
                        'attachment; filename="capacitor_template.csv"'
                    )
                },
            )
        elif path == "/api/template.xlsx":
            self._send_bytes(
                HTTPStatus.OK,
                build_template_xlsx(load_sample_records()),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                session_id=session_id,
                extra_headers={
                    "Content-Disposition": (
                        'attachment; filename="capacitor_template.xlsx"'
                    )
                },
            )
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {path}"})

    def do_GET(self) -> None:
        """Handle GET requests for the HTML page, health check, state, and templates."""
        path = urlparse(self.path).path
        if path == "/api/health":
            db = self._balancer_server.db
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "active_users": db.active_session_count(),
                    "max_users": db.max_concurrent_users,
                },
            )
            return

        if path in {"/", "/index.html"}:
            sess = self._acquire_session_or_reject(is_html=True)
            if sess is not None:
                session_id, _ = sess
                self._send_bytes(
                    HTTPStatus.OK,
                    self._balancer_server.ui_path.read_bytes(),
                    "text/html; charset=utf-8",
                    session_id=session_id,
                )
            return

        sess = self._acquire_session_or_reject(is_html=False)
        if sess is not None:
            session_id, active_users = sess
            self._dispatch_api_get(path, session_id, active_users)

    def do_POST(self) -> None:
        """Handle POST requests for sessions, tab creation, uploads, solves, & edits."""
        path = urlparse(self.path).path
        db = self._balancer_server.db

        if path == "/api/session/disconnect":
            sid = self._extract_session_id()
            if sid is not None and bool(sid):
                db.release_session(sid)
            self._send_json(HTTPStatus.OK, {"released": True})
            return

        sess = self._acquire_session_or_reject(is_html=False)
        if sess is None:
            return
        session_id, active_users = sess

        if path == "/api/session/heartbeat":
            self._send_json(
                HTTPStatus.OK,
                {
                    "session_id": session_id,
                    "active_users": active_users,
                    "max_users": db.max_concurrent_users,
                },
                session_id=session_id,
            )
            return

        try:
            body = self._read_json_body()
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        if path == "/api/tabs":
            self._create_tab(session_id, body)
            return

        match = _TAB_ROUTE_RE.match(path)
        if not match or not match.group(2):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})
            return

        self._handle_tab_action(session_id, match.group(1), match.group(2), body)

    def _create_tab(self, session_id: str, body: dict[str, Any]) -> None:
        try:
            tab = self._balancer_server.db.create_tab(
                session_id,
                title=body.get("title"),
                voltage_kv=int(body.get("voltage_kv", 115)),
                series=int(body.get("series", 6)),
                parallel=int(body.get("parallel", 9)),
            )
        except (TabLimitError, ValueError, TypeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.CREATED, {"tab": tab}, session_id=session_id)

    def _exec_tab_action(  # ruff: ignore[too-many-return-statements]
        self,
        session_id: str,
        tab_id: str,
        action: str,
        body: dict[str, Any],
    ) -> tuple[int, dict[str, Any]]:
        db = self._balancer_server.db
        if action == "upload":
            filename = str(body.get("filename", "")).strip()
            b64 = str(body.get("content_base64", ""))
            if not filename or not b64:
                msg = "Both 'filename' and 'content_base64' are required."
                raise ValueError(msg)
            records = parse_and_validate_upload(filename, base64.b64decode(b64))
            tab = db.set_tab_inventory(
                session_id, tab_id, filename=filename, records=records
            )
            return HTTPStatus.OK, {"tab": tab, "uploaded_count": len(records)}

        if action == "sample":
            records = load_sample_records()
            tab = db.set_tab_inventory(
                session_id, tab_id, filename="sample_capacitors.csv", records=records
            )
            return HTTPStatus.OK, {"tab": tab, "uploaded_count": len(records)}

        if action == "solve":
            solver_id = str(body.get("solver_id", "mojo_residual"))
            seed = int(body.get("seed", 42))
            tab = db.get_tab(session_id, tab_id)
            ordered_indices, fitness, solve_ms = run_solver(
                solver_id,
                tab["capacitors"],
                series=int(tab["series"]),
                parallel=int(tab["parallel"]),
                seed=seed,
            )
            updated = db.apply_solution(
                session_id,
                tab_id,
                solver_id=solver_id,
                ordered_row_indices=ordered_indices,
                fitness=fitness,
                solve_ms=solve_ms,
            )
            return HTTPStatus.OK, {
                "tab": updated,
                "fitness": fitness,
                "solve_ms": solve_ms,
                "solver_id": solver_id,
            }

        if action == "place":
            tab = db.place_capacitor(
                session_id,
                tab_id,
                cap_id=int(body["cap_id"]),
                slot_position=int(body["slot_position"]),
            )
            return HTTPStatus.OK, {"tab": tab}

        if action == "remove":
            cap_id = int(body["cap_id"]) if body.get("cap_id") is not None else None
            slot_pos = (
                int(body["slot_position"])
                if body.get("slot_position") is not None
                else None
            )
            tab = db.remove_capacitor(
                session_id, tab_id, cap_id=cap_id, slot_position=slot_pos
            )
            return HTTPStatus.OK, {"tab": tab}

        if action == "swap":
            tab = db.swap_capacitors(
                session_id,
                tab_id,
                cap_id_a=int(body["cap_id_a"]),
                cap_id_b=int(body["cap_id_b"]),
            )
            return HTTPStatus.OK, {"tab": tab}

        if action == "clear":
            return HTTPStatus.OK, {"tab": db.clear_bank(session_id, tab_id)}

        return HTTPStatus.NOT_FOUND, {"error": f"Unknown tab action: {action}"}

    def _handle_tab_action(
        self,
        session_id: str,
        tab_id: str,
        action: str,
        body: dict[str, Any],
    ) -> None:
        try:
            status, payload = self._exec_tab_action(session_id, tab_id, action, body)
        except ValidationError as exc:
            self._send_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": exc.message, "details": exc.details},
            )
        except NotFoundError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except (KeyError, ValueError, TypeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except (FileNotFoundError, RuntimeError) as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
        else:
            self._send_json(status, payload, session_id=session_id)

    def do_PATCH(self) -> None:
        """Handle PATCH requests to update tab title, voltage, series, or parallel."""
        sess = self._acquire_session_or_reject(is_html=False)
        if sess is None:
            return
        session_id, _ = sess

        match = _TAB_ROUTE_RE.match(urlparse(self.path).path)
        if not match or match.group(2) is not None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Invalid tab route"})
            return

        try:
            body = self._read_json_body()
            tab = self._balancer_server.db.update_tab_config(
                session_id,
                match.group(1),
                title=body.get("title"),
                voltage_kv=(
                    int(body["voltage_kv"])
                    if body.get("voltage_kv") is not None
                    else None
                ),
                series=int(body["series"]) if body.get("series") is not None else None,
                parallel=(
                    int(body["parallel"]) if body.get("parallel") is not None else None
                ),
            )
        except NotFoundError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        else:
            self._send_json(HTTPStatus.OK, {"tab": tab}, session_id=session_id)

    def do_DELETE(self) -> None:
        """Handle DELETE requests to remove a capacitor bank tab."""
        sess = self._acquire_session_or_reject(is_html=False)
        if sess is None:
            return
        session_id, _ = sess

        match = _TAB_ROUTE_RE.match(urlparse(self.path).path)
        if not match or match.group(2) is not None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Invalid tab route"})
            return

        try:
            tabs = self._balancer_server.db.delete_tab(session_id, match.group(1))
        except NotFoundError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        else:
            self._send_json(HTTPStatus.OK, {"tabs": tabs}, session_id=session_id)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    *,
    db_path: Path = DEFAULT_DB_PATH,
    max_users: int = DEFAULT_MAX_CONCURRENT_USERS,
    session_ttl: float = DEFAULT_SESSION_TTL_SECONDS,
) -> BalancerHTTPServer:
    db = ServerDB(
        db_path,
        max_concurrent_users=max_users,
        session_ttl_seconds=session_ttl,
    )
    return BalancerHTTPServer((host, port), db)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lightweight HTTP server for the HV Capacitor Bank Balancer."
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Bind port (default: 8080)"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite state DB path (default: server/app_state.db)",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_USERS,
        help="Maximum concurrent user sessions (default: 50)",
    )
    args = parser.parse_args()

    srv = create_server(
        args.host,
        args.port,
        db_path=args.db,
        max_users=args.max_users,
    )
    print(  # ruff: ignore[print]
        f"Capacitor Bank Balancer server listening on "
        f"http://{args.host}:{args.port} (max_users={args.max_users}, db={args.db})"
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")  # ruff: ignore[print]
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
