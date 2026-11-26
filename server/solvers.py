"""Unified solver dispatch (sorted fastest -> slowest) using the Mojo .so extension."""

from __future__ import annotations

import asyncio
import importlib.util
import shutil
import sqlite3
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import TYPE_CHECKING, Any

from capacitor_balance import functional_program as py_fp

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
MOJO_DIR = REPO_ROOT / "mojo_solver"
MOJO_EXT_SRC = MOJO_DIR / "capacitor_mojo.mojo"
MOJO_CORE_SRC = MOJO_DIR / "core.mojo"
MOJO_EXT_SO = MOJO_DIR / "capacitor_mojo.so"
SAMPLE_DB_PATH = REPO_ROOT / "src" / "capacitor_balance" / "capacitors.db"
DEFAULT_SOLVE_TIMEOUT_SECONDS = 60.0

_MOJO_EXT_LOCK = threading.Lock()
_MOJO_EXT_CACHE: dict[str, ModuleType | None] = {"mod": None}

SOLVER_CATALOG: list[dict[str, Any]] = [
    {
        "id": "mojo_residual",
        "name": "Mojo Residual Balancer (.so)",
        "speed_rank": 1,
        "speed_bars": 5,
        "speed_label": "Instant (~0.04 ms)",
        "speed_tier": "instant",
        "description": (
            "In-process Mojo .so C-extension: algebraic max-min residual matching + "
            "binary-searched pool swaps + phase A/B/C reciprocal balancing."
        ),
    },
    {
        "id": "mojo_fast",
        "name": "Mojo Local Search 2-Opt (.so)",
        "speed_rank": 2,
        "speed_bars": 4,
        "speed_label": "Ultra Fast (~2 ms)",
        "speed_tier": "ultra",
        "description": (
            "In-process Mojo .so C-extension: greedy LPT initialization + "
            "exhaustive incremental 2-opt inter-group swaps and pool replacements."
        ),
    },
    {
        "id": "mojo_ga",
        "name": "Mojo Genetic Algorithm (.so)",
        "speed_rank": 3,
        "speed_bars": 3,
        "speed_label": "Fast (~0.3 s)",
        "speed_tier": "fast",
        "description": (
            "In-process Mojo .so C-extension: native compiled SIMD genetic "
            "algorithm (1,000 pop x 500 iters)."
        ),
    },
    {
        "id": "python_polars_ga",
        "name": "Python + Polars GA",
        "speed_rank": 4,
        "speed_bars": 1,
        "speed_label": "Slow (~18 s)",
        "speed_tier": "slow",
        "description": (
            "Vectorized async Polars LazyFrame genetic algorithm "
            "(1,000 pop x 500 iters)."
        ),
    },
]

VALID_SOLVER_IDS = {s["id"] for s in SOLVER_CATALOG}


def _find_mojo_executable() -> str:
    venv_mojo = Path(sys.executable).parent / "mojo"
    if venv_mojo.exists():
        return str(venv_mojo)
    found = shutil.which("mojo")
    if found is not None:
        return found
    msg = (
        "Could not locate `mojo` compiler. Run `uv sync` to install the "
        "Mojo nightly dependency."
    )
    raise FileNotFoundError(msg)


def _needs_rebuild() -> bool:
    if not MOJO_EXT_SO.exists():
        return True
    so_mtime = MOJO_EXT_SO.stat().st_mtime
    return any(
        src.exists() and src.stat().st_mtime > so_mtime
        for src in (MOJO_EXT_SRC, MOJO_CORE_SRC)
    )


def load_mojo_extension(
    *,
    force_rebuild: bool = False,
    timeout_seconds: float = DEFAULT_SOLVE_TIMEOUT_SECONDS,
) -> ModuleType:
    """Build (if missing or stale) and import ``mojo_solver/capacitor_mojo.so``.

    Returns:
        The loaded ``capacitor_mojo`` Python extension module.

    Raises:
        TimeoutError: If compiling the shared library exceeds timeout_seconds.
        RuntimeError: If building or loading the shared library fails.
    """
    with _MOJO_EXT_LOCK:
        cached = _MOJO_EXT_CACHE["mod"]
        if cached is not None and not force_rebuild and not _needs_rebuild():
            return cached

        if force_rebuild or _needs_rebuild():
            mojo_bin = _find_mojo_executable()
            try:
                proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
                    [
                        mojo_bin,
                        "build",
                        "-I",
                        str(MOJO_DIR),
                        "--emit",
                        "shared-lib",
                        "-O",
                        "3",
                        "--Werror",
                        "-o",
                        str(MOJO_EXT_SO),
                        str(MOJO_EXT_SRC),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                msg = (
                    f"Building {MOJO_EXT_SO.name} timed out after "
                    f"{timeout_seconds:g}s."
                )
                raise TimeoutError(msg) from exc
            if proc.returncode != 0:
                err = proc.stderr.strip() or proc.stdout.strip() or "mojo build failed"
                msg = f"Failed to build {MOJO_EXT_SO.name}: {err}"
                raise RuntimeError(msg)

        spec = importlib.util.spec_from_file_location("capacitor_mojo", MOJO_EXT_SO)
        if spec is None or spec.loader is None:
            msg = f"Unable to create import spec for {MOJO_EXT_SO}"
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MOJO_EXT_CACHE["mod"] = module
        return module


def load_sample_records() -> list[dict[str, Any]]:
    """Load unique valid capacitors from the repository sample SQLite DB.

    Returns:
        List of capacitor records with unique serials.
    """
    if not SAMPLE_DB_PATH.exists():
        return []
    with sqlite3.connect(SAMPLE_DB_PATH) as conn:
        rows = conn.execute(
            'SELECT "N° DE SERIE", "CAPACITANCIA ", "CAP. DESVIACION " '
            'FROM capacitors WHERE "CAPACITANCIA " IS NOT NULL'
        ).fetchall()
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    for serial, cap, dev in rows:
        s_str = str(serial).strip()
        if not s_str:
            continue
        if s_str in seen:
            s_str = f"{s_str}-B"
        seen.add(s_str)
        records.append(
            {
                "serial": s_str,
                "capacitance": float(cap),
                "deviation": int(dev),
            }
        )
    return records


def _run_python_polars_ga(  # ruff: ignore[too-many-arguments]
    capacitors: list[dict[str, Any]],
    series: int,
    parallel: int,
    *,
    iterations: int = 500,
    generation_length: int = 1000,
    timeout_seconds: float = DEFAULT_SOLVE_TIMEOUT_SECONDS,
) -> tuple[list[int], float, float]:
    genoma_len = 3 * series * parallel
    best_len = max(1, int(generation_length * 10 / 100))
    mut_len = int(generation_length * 20 / 100)
    shuff_len = int(generation_length * 30 / 100)
    cross_len = int(generation_length * 10 / 100)
    rand_len = generation_length - (best_len + mut_len + shuff_len + cross_len)

    options: list[py_fp.Capacitor] = [
        py_fp.Capacitor(
            idx=int(c["row_index"]),
            serial=str(c["serial"]),
            cap=float(c["capacitance"]),
        )
        for c in capacitors
    ]

    t0 = time.perf_counter()

    async def _evolve() -> tuple[list[int], float]:
        generation = py_fp.random_generation(options, generation_length, genoma_len)
        for _ in range(iterations):
            if time.perf_counter() - t0 > timeout_seconds:
                msg = (
                    f"Solver 'python_polars_ga' timed out after "
                    f"{timeout_seconds:g} seconds. Try a faster Mojo solver."
                )
                raise TimeoutError(msg)
            sorted_gen, _ = await py_fp.sort_generation(
                generation, series=series, parallel=parallel
            )
            top_genomas = sorted_gen[:best_len]
            mutated, shuffled, randomized, crossovered = await asyncio.gather(
                asyncio.to_thread(py_fp.mutated_genomas, options, top_genomas, mut_len),
                asyncio.to_thread(py_fp.shuffled_genomas, top_genomas, shuff_len),
                asyncio.to_thread(
                    py_fp.random_genomas, options, genoma_len, rand_len
                ),
                asyncio.to_thread(
                    py_fp.crossovered_genomas, top_genomas, cross_len, generation
                ),
            )
            generation = top_genomas + mutated + shuffled + randomized + crossovered

        final_sorted, best_fit = await py_fp.sort_generation(
            generation, series=series, parallel=parallel
        )
        return [int(c["idx"]) for c in final_sorted[0]], float(best_fit)

    ordered_indices, best_fit = asyncio.run(
        asyncio.wait_for(_evolve(), timeout=timeout_seconds)
    )
    solve_ms = (time.perf_counter() - t0) * 1000.0
    return ordered_indices, best_fit, solve_ms


def run_solver(  # ruff: ignore[too-many-arguments]
    solver_id: str,
    capacitors: list[dict[str, Any]],
    *,
    series: int,
    parallel: int,
    seed: int = 42,
    timeout_seconds: float = DEFAULT_SOLVE_TIMEOUT_SECONDS,
) -> tuple[list[int], float, float]:
    """Execute the selected solver on uploaded capacitors with a timeout.

    Returns:
        Tuple of (ordered_row_indices, fitness, solve_ms).

    Raises:
        ValueError: If capacitor count is insufficient or solver_id is unknown.
        TimeoutError: If the solver does not finish within timeout_seconds.
    """
    needed = 3 * series * parallel
    if len(capacitors) < needed:
        msg = (
            f"Bank requires 3 x {series} x {parallel} = {needed} capacitors, "
            f"but only {len(capacitors)} uploaded."
        )
        raise ValueError(msg)

    if solver_id == "python_polars_ga":
        return _run_python_polars_ga(
            capacitors, series, parallel, timeout_seconds=timeout_seconds
        )

    if solver_id not in VALID_SOLVER_IDS:
        msg = f"Unknown solver_id: '{solver_id}'"
        raise ValueError(msg)

    caps_float = [float(c["capacitance"]) for c in capacitors]
    ext = load_mojo_extension(timeout_seconds=timeout_seconds)

    def _invoke_mojo() -> tuple[list[int], float, float]:
        if solver_id == "mojo_residual":
            raw_indices, raw_fit, raw_ms = ext.solve_residual(
                caps_float, series, parallel
            )
        elif solver_id == "mojo_fast":
            raw_indices, raw_fit, raw_ms = ext.solve_fast(
                caps_float, series, parallel, seed
            )
        else:
            raw_indices, raw_fit, raw_ms = ext.solve_ga(
                caps_float, series, parallel, 1000, 500, seed
            )
        return [int(i) for i in raw_indices], float(raw_fit), float(raw_ms)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_invoke_mojo)
    try:
        local_indices, fitness, solve_ms = future.result(timeout=timeout_seconds)
    except FuturesTimeoutError as exc:
        executor.shutdown(wait=False, cancel_futures=True)
        msg = (
            f"Solver '{solver_id}' timed out after {timeout_seconds:g} seconds."
        )
        raise TimeoutError(msg) from exc
    else:
        executor.shutdown(wait=False)

    ordered_row_indices = [
        int(capacitors[int(local_idx)]["row_index"]) for local_idx in local_indices
    ]
    return ordered_row_indices, float(fitness), float(solve_ms)
