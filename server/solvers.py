"""Unified solver dispatch (sorted fastest -> slowest) for the web server."""

from __future__ import annotations

import asyncio
import csv
import sqlite3
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import tempfile
import time
from pathlib import Path
from typing import Any

from capacitor_balance import functional_program as py_fp

REPO_ROOT = Path(__file__).resolve().parent.parent
MOJO_RESIDUAL_BIN = REPO_ROOT / "mojo_solver" / "residual_solver"
MOJO_SOLVER_BIN = REPO_ROOT / "mojo_solver" / "solver"
SAMPLE_DB_PATH = REPO_ROOT / "src" / "capacitor_balance" / "capacitors.db"

SOLVER_CATALOG: list[dict[str, Any]] = [
    {
        "id": "mojo_residual",
        "name": "Mojo Residual Balancer",
        "speed_rank": 1,
        "speed_bars": 5,
        "speed_label": "Instant (~0.04 ms)",
        "speed_tier": "instant",
        "description": (
            "Algebraic max-min residual matching + binary-searched pool swaps + "
            "phase A/B/C reciprocal balancing."
        ),
    },
    {
        "id": "mojo_fast",
        "name": "Mojo Local Search (2-Opt)",
        "speed_rank": 2,
        "speed_bars": 4,
        "speed_label": "Ultra Fast (~1.5 ms)",
        "speed_tier": "ultra",
        "description": (
            "Greedy LPT initialization + exhaustive incremental 2-opt inter-group "
            "swaps and pool replacements."
        ),
    },
    {
        "id": "mojo_ga",
        "name": "Mojo Genetic Algorithm",
        "speed_rank": 3,
        "speed_bars": 3,
        "speed_label": "Fast (~0.3 s)",
        "speed_tier": "fast",
        "description": (
            "Native compiled Mojo SIMD genetic algorithm (1,000 pop x 500 iters)."
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


def _write_temp_sqlite_db(db_path: Path, capacitors: list[dict[str, Any]]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            'CREATE TABLE "capacitors" ('
            '"N°" INTEGER, '
            '"N° DE SERIE" TEXT, '
            '"CAPACITANCIA " REAL, '
            '"CAP. DESVIACION " INTEGER)'
        )
        conn.executemany(
            "INSERT INTO capacitors VALUES (?, ?, ?, ?)",
            [
                (
                    i + 1,
                    str(cap["serial"]),
                    float(cap["capacitance"]),
                    int(cap["deviation"]),
                )
                for i, cap in enumerate(capacitors)
            ],
        )
        conn.commit()


def _run_mojo_binary(
    binary: Path,
    args: list[str],
    capacitors: list[dict[str, Any]],
) -> tuple[list[int], float, float]:
    if not binary.exists():
        msg = f"Mojo binary not found at {binary}. Build it first with `mojo build`."
        raise FileNotFoundError(msg)

    with tempfile.TemporaryDirectory(prefix="cap_bal_") as tmp_dir:
        tmp_db = Path(tmp_dir) / "input.db"
        tmp_csv = Path(tmp_dir) / "output.csv"
        _write_temp_sqlite_db(tmp_db, capacitors)

        t0 = time.perf_counter()
        proc = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            [str(binary), "--db", str(tmp_db), "--out", str(tmp_csv), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip() or "Mojo solver failed"
            raise RuntimeError(err)

        ordered_row_indices: list[int] = []
        with tmp_csv.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                local_idx = int(row["db_idx"])
                ordered_row_indices.append(int(capacitors[local_idx]["row_index"]))

        tokens = dict(
            tok.split("=", 1)
            for tok in proc.stdout.strip().splitlines()[-1].split()
            if "=" in tok
        )
        fitness = float(tokens.get("best_fit", "0.0"))
        if "solve_us" in tokens:
            solve_ms = float(tokens["solve_us"]) / 1000.0
        elif "solve_s" in tokens:
            solve_ms = float(tokens["solve_s"]) * 1000.0
        else:
            solve_ms = elapsed_ms

        return ordered_row_indices, fitness, solve_ms


def _run_python_polars_ga(
    capacitors: list[dict[str, Any]],
    series: int,
    parallel: int,
    *,
    iterations: int = 500,
    generation_length: int = 1000,
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

    async def _evolve() -> tuple[list[int], float]:
        generation = py_fp.random_generation(options, generation_length, genoma_len)
        for _ in range(iterations):
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

    t0 = time.perf_counter()
    ordered_indices, best_fit = asyncio.run(_evolve())
    solve_ms = (time.perf_counter() - t0) * 1000.0
    return ordered_indices, best_fit, solve_ms


def run_solver(
    solver_id: str,
    capacitors: list[dict[str, Any]],
    *,
    series: int,
    parallel: int,
    seed: int = 42,
) -> tuple[list[int], float, float]:
    """Execute the selected solver on uploaded capacitors.

    Returns:
        Tuple of (ordered_row_indices, fitness, solve_ms).

    Raises:
        ValueError: If capacitor count is insufficient or solver_id is unknown.
    """
    needed = 3 * series * parallel
    if len(capacitors) < needed:
        msg = (
            f"Bank requires 3 x {series} x {parallel} = {needed} capacitors, "
            f"but only {len(capacitors)} uploaded."
        )
        raise ValueError(msg)

    if solver_id == "mojo_residual":
        return _run_mojo_binary(
            MOJO_RESIDUAL_BIN,
            ["--series", str(series), "--parallel", str(parallel)],
            capacitors,
        )
    if solver_id == "mojo_fast":
        return _run_mojo_binary(
            MOJO_SOLVER_BIN,
            [
                "--mode",
                "fast",
                "--series",
                str(series),
                "--parallel",
                str(parallel),
                "--seed",
                str(seed),
            ],
            capacitors,
        )
    if solver_id == "mojo_ga":
        return _run_mojo_binary(
            MOJO_SOLVER_BIN,
            [
                "--mode",
                "ga",
                "--series",
                str(series),
                "--parallel",
                str(parallel),
                "--seed",
                str(seed),
            ],
            capacitors,
        )
    if solver_id == "python_polars_ga":
        return _run_python_polars_ga(capacitors, series, parallel)

    msg = f"Unknown solver_id: '{solver_id}'"
    raise ValueError(msg)
