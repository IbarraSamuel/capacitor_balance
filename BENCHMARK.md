# Capacitor Balance — Python vs. Mojo Benchmark

Comparison of execution time and solution accuracy between the Python/Polars genetic algorithm ([functional_program.py](src/capacitor_balance/functional_program.py)) and the native Mojo implementation ([solver.mojo](mojo_solver/solver.mojo)).

## Problem Definition

Given `capacitors.db` (324 rows total; 321 valid non-`NULL` capacitors with capacitances in `[1.010, 1.037]`), select an ordered sequence of **162 distinct capacitors** arranged as **3 phases × 6 series groups × 9 parallel capacitors** to minimize:

$$\text{fitness} = 20 \cdot \sigma_{\text{phase}} + \sigma_{\text{parallel}}$$

where:
- Each of the 18 parallel groups has capacitance $P_g = \sum_{k=0}^{8} C_{9g+k}$ and sample standard deviation $\sigma_{\text{parallel}}$ (`ddof=1`).
- Each of the 3 phases has series capacitance $S_p = \left(\sum_{s=0}^{5} \frac{1}{P_{6p+s}}\right)^{-1}$ and sample standard deviation $\sigma_{\text{phase}}$ (`ddof=1`).

## Test Environment

| Component | Details |
|---|---|
| **Hardware** | Apple M1 (8 cores: 4 performance + 4 efficiency) |
| **OS** | macOS (ARM64) |
| **Python** | CPython 3.15.2 (`uv`), Polars 2.2.0 |
| **Mojo** | `Mojo 1.3.0.dev2026112505` (nightly wheel via `uv`), compiled with `mojo build -O 3 --Werror` |
| **Dataset** | `src/capacitor_balance/capacitors.db` (321 valid capacitors) |

## Implementations Compared

1. **Python + Polars GA** ([src/capacitor_balance/functional_program.py](src/capacitor_balance/functional_program.py)) — unmodified baseline. Vectorized `pl.LazyFrame.collect_async()` batch evaluation + concurrent `asyncio.gather` offspring generation (`1,000` population × `500` iterations).
2. **Mojo GA (`--mode ga`)** ([mojo_solver/solver.mojo](mojo_solver/solver.mojo)) — faithful port of the exact same `1,000` population × `500` iterations GA (10% elite, 20% mutation, 30% shuffle, 10% crossover, 30% random) using flat double-buffered `List[Int32]` storage, SIMD-4 `Float64` reductions, native `libsqlite3` FFI, and $O(1)$ generation-stamp membership checks.
3. **Mojo Fast (`--mode fast`)** ([mojo_solver/solver.mojo](mojo_solver/solver.mojo)) — greedy balanced LPT (Longest-Processing-Time) initialization followed by incremental $O(1)$ 2-opt inter-group swap and unused-pool replacement local search, using exact two-pass SIMD-4 variance so fitness remains numerically exact down to `0.0`.
4. **Mojo Residual (`residual_solver`)** ([mojo_solver/residual_solver.mojo](mojo_solver/residual_solver.mojo)) — tightest-window selection + snake/LPT packing + targeted $O(1)$ max-min group cancellation & $O(\log M)$ binary-searched pool replacement + phase A/B/C reciprocal balancing (zero `sqrt` calls in the inner balancing loop).

---

## Benchmark Results

### Timing Comparison

| Solver | Trials | Process Wall Time (mean) | Process Wall Time (median / min / max) | Pure Solve Time (mean / median) | Speedup vs. Python (Wall) | Speedup vs. Python (Solve) |
|---|---:|---:|---:|---:|---:|---:|
| **Python + Polars GA** | 3 | **18.43 s** | 18.27 s / 17.44 s / 19.57 s | ~18.4 s | 1.0× | 1.0× |
| **Mojo GA (`--mode ga`)** | 10 | **0.334 s** | 0.316 s / 0.305 s / 0.437 s | **0.304 s** / 0.297 s | **55.2×** | **60.6×** |
| **Mojo Fast (`--mode fast`)** | 20 | **17.7 ms** | 17.8 ms / 16.8 ms / 18.2 ms | **1.46 ms** / 1.45 ms | **1,043×** | **12,666×** |
| **Mojo Residual (`residual_solver`)** | 20 *(+1000 warm)* | **14.6 ms** | 14.1 ms / 13.5 ms / 24.1 ms | **36.2 µs** cold / **19.7 µs** warm | **1,267×** | **509,000×** *(cold)* / **934,000×** *(warm)* |

> **Note:** Process wall time for Mojo CLI includes OS process spawn (`subprocess.run`), dynamic loading of `libsqlite3.dylib`, reading the SQLite database, solving, and writing CSV output. On `residual_solver`, 99.7% of CLI wall time is OS/SQLite I/O; the solve itself finishes in **10 moves** (**36.2 µs** cold, **19.7 µs** warm across 1,000 runs, **18.0 µs** min).

### In-Process Python `.so` Extension (`mojo_solver/capacitor_mojo.so`)

Calling Mojo directly from Python via the compiled shared library (`--emit shared-lib`) eliminates subprocess startup, temporary SQLite I/O, and CSV serialization:

| Python Call (`import capacitor_mojo`) | End-to-End Python Time | Mojo Inner Solve Time | Speedup vs. Mojo CLI Subprocess | Fitness |
|---|---:|---:|---:|---:|
| `capacitor_mojo.solve_residual(caps, 6, 9)` | **0.155 ms** *(155 µs)* | **0.044–0.051 ms** | **94× faster** *(vs. 14.6 ms)* | `5.44 × 10⁻¹⁵` |
| `capacitor_mojo.solve_fast(caps, 6, 9, 42)` | **2.76 ms** | **2.59–2.72 ms** | **6.4× faster** *(vs. 17.7 ms)* | **`0.0`** |
| `capacitor_mojo.solve_ga(caps, 6, 9, 1000, 500, 42)` | **282.3 ms** | **281.2–282.2 ms** | **1.18× faster** *(vs. 334 ms)* | `7.67 × 10⁻⁴` |

### Accuracy & Cross-Verification

Every Mojo solution (10 GA runs, 20 Fast runs, and 20 Residual runs) was loaded back into Python and independently verified against `src/capacitor_balance/capacitors.db` and the unmodified `functional_program.evaluate_generation()` function:

| Solver | Best Fitness (mean) | Best Fitness (min / max) | Parallel Group Spread ($\max P_g - \min P_g$) | Phase Spread ($\max S_p - \min S_p$) | Max Discrepancy vs. Python Evaluator |
|---|---:|---:|---:|---:|---:|
| **Python + Polars GA** | `6.82 × 10⁻⁴` | `5.94 × 10⁻⁴` / `8.58 × 10⁻⁴` | — | — | reference |
| **Mojo GA (`--mode ga`)** | `7.30 × 10⁻⁴` | `3.83 × 10⁻⁴` / `1.15 × 10⁻³` | — | — | `7.29 × 10⁻¹⁵` |
| **Mojo Fast (`--mode fast`)** | **`0.0`** | **`0.0` / `0.0`** *(20/20 runs)* | **`0.000`** *(all 18 groups = `9.209`)* | **`0.000`** *(all 3 phases = `1.534833333333`)* | **`0.0`** |
| **Mojo Residual (`residual_solver`)** | **`5.44 × 10⁻¹⁵`** *(1 ULP)* | **`5.44 × 10⁻¹⁵`** *(20/20 runs)* | **`0.000`** *(all 18 groups = `9.202`)* | **`0.000`** *(all 3 phases = `1.533666666667`)* | **`3.77 × 10⁻¹⁵`** *(Python eval: `1.67 × 10⁻¹⁵`)* |

#### Verified Properties of Mojo Fast Solution
- **162 / 162** unique database rows (`db_idx`), zero `NULL` capacitances, exact capacitance & serial match against `capacitors.db`.
- All 18 parallel groups achieve an identical total capacitance of **`9.209000000000`**.
- All 3 series phases achieve an identical total capacitance of **`1.534833333333`**.
- Independently scored by Python's unmodified `evaluate_generation([mojo_genoma])` as **`0.0000000000000000e+00`** (global mathematical optimum).

---

## Why Mojo Is Faster (and More Accurate)

1. **Zero per-iteration allocation & data-frame overhead (55–61× on identical GA):**
   - Python constructs a 162,000-row Polars `LazyFrame` from Python lists of `dict`s on every iteration (500 times), plus Python-side list/dict copying for mutation, shuffle, and crossover.
   - Mojo stores the entire population in two flat contiguous `List[Int32]` buffers (`1000 × 162`), evaluates all 18 group sums and variances in registers using `SIMD[.float64, 4]`, and tracks uniqueness in $O(1)$ with a reusable stamp array.
2. **Algorithmic leverage enabled by native speed (1,043× wall / 12,666× solve + exact `0.0` fitness):**
   - A genetic algorithm explores randomly and plateaus around `~6 × 10⁻⁴` after 500 generations because simultaneous random swaps rarely cancel residual group imbalances.
   - Because each incremental fitness evaluation in compiled Mojo takes ~25 ns (evaluating only the 18 group sums with SIMD-4 two-pass variance), greedy LPT initialization + exhaustive 2-opt inter-group swaps and pool replacements reaches exact `0.0` balance in **3 sweeps (1.46 ms)**.
3. **Zero Python startup overhead:**
   - Reading `capacitors.db` directly through Mojo's `OwnedDLHandle("libsqlite3.dylib")` avoids initializing CPython or Polars altogether.

## How to Reproduce

```bash
# 0. Sync dependencies (includes Mojo nightly, Polars, XlsxWriter)
uv sync

# 1. Build the in-process Mojo Python extension (.so)
uv run mojo build -I mojo_solver --emit shared-lib -O 3 --Werror \
  -o mojo_solver/capacitor_mojo.so mojo_solver/capacitor_mojo.mojo

# 2. Build standalone Mojo CLI solvers (optional)
uv run mojo build -O 3 --Werror -o mojo_solver/solver mojo_solver/solver.mojo
uv run mojo build -O 3 --Werror -o mojo_solver/residual_solver mojo_solver/residual_solver.mojo

# 3. Run Mojo CLI (GA parity, fast local search, and microsecond residual balancer)
./mojo_solver/solver --mode ga --seed 42
./mojo_solver/solver --mode fast --seed 42
./mojo_solver/residual_solver --bench 1000

# 4. Run Python baseline
PYTHONPATH=src uv run python src/capacitor_balance/functional_program.py
```
