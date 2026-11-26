# Capacitor Balance

A 3-phase (**A, B, C**) series-parallel capacitor bank balancer.

Given measured capacitances in `src/capacitor_balance/capacitors.db`, this project selects and arranges **`3 × series × parallel` distinct capacitors** across **3 phases (A, B, C)** (default: **3 phases × 6 series groups × 9 parallel capacitors = 162 units**) so that all parallel groups share capacitance evenly and all three phases achieve near-zero phase-to-phase unbalance.

Both a **Python + Polars** solver and **native Mojo** solvers (usable both as CLI binaries and as an in-process `.so` Python extension module) are included.

> [!WARNING]
> **Mojo Nightly Dependency:** This project depends on **Mojo nightly** (`mojo` from `https://whl.modular.com/nightly/simple/`). Because Mojo nightly is under active development, language syntax, stdlib APIs, and Python extension bindings (`PythonModuleBuilder`) may break between nightly updates. If a newer nightly breaks compilation, pin `mojo` in `pyproject.toml` to the tested nightly version (`1.3.0.dev2026112505`).

## Project Layout

```text
capacitor_balance/
├── src/capacitor_balance/
│   ├── capacitors.db           # Input SQLite database of measured capacitors
│   └── functional_program.py   # Async Python + vectorized Polars GA solver
├── mojo_solver/
│   ├── core.mojo               # Shared Mojo solver algorithms (no main)
│   ├── capacitor_mojo.mojo     # Python C-extension module -> capacitor_mojo.so
│   ├── solver.mojo             # Standalone CLI solver (--mode fast | --mode ga)
│   └── residual_solver.mojo    # Standalone CLI sub-millisecond residual solver
├── server/
│   ├── app.py                  # Lightweight stdlib HTTP server (50-user cap)
│   ├── db.py                   # SQLite persistence (sessions, tabs, stock/bank, runs)
│   ├── parser.py               # Strict CSV & XLSX upload validator + templates
│   └── solvers.py              # Speed-sorted solver catalog calling capacitor_mojo.so
├── visualizer/
│   └── index.html              # Interactive multi-tab 3-phase HV bank web UI
├── DESIGN.md                   # Architecture & algorithm design details
├── BENCHMARK.md                # Timing & accuracy comparison (Python vs. Mojo)
└── pyproject.toml              # Python & Mojo nightly dependencies, ruff & ty config
```

## What the Solver Optimizes

For an ordered selection of 162 capacitors:

1. **Parallel reduction** (18 groups of 9 capacitors): $P_g = \sum_{k=0}^{8} C_{9g+k}$, with sample standard deviation $\sigma_{\text{parallel}}$ (`ddof=1`).
2. **Series reduction** (3 phases A, B, C of 6 groups each): $S_p = \left(\sum_{s=0}^{5} \frac{1}{P_{6p+s}}\right)^{-1}$, with sample standard deviation $\sigma_{\text{phase}}$ (`ddof=1`).
3. **Fitness to minimize**: $F = 20 \cdot \sigma_{\text{phase}} + \sigma_{\text{parallel}}$.

A fitness of `0.0` means every parallel group has the exact same total capacitance and all three phases (A, B, C) have identical equivalent capacitance.

---

## Using the Python Solver

### Prerequisites

- Python `>= 3.13`
- [`uv`](https://docs.astral.sh/uv/)

### Install & Run

```bash
# Sync dependencies and install package editably (mojo nightly, polars, xlsxwriter)
uv sync

# Run the genetic algorithm (default: 3 phases × 6 series × 9 parallel = 162 capacitors)
uv run python -m capacitor_balance.functional_program

# Or choose a custom 3-phase bank size (e.g. 3 × 4 × 5 = 60 capacitors)
uv run python -m capacitor_balance.functional_program \
  --series 4 --parallel 5 --iterations 200 --generation-length 500
```

| Flag | Default | Description |
|---|---|---|
| `--series <int>` | `6` | Series-connected parallel groups per phase (total groups = `3 × series`) |
| `--parallel <int>` | `9` | Capacitors in parallel per group (total capacitors = `3 × series × parallel`) |
| `--iterations <int>` | `500` | Number of GA iterations |
| `--generation-length <int>` | `1000` | Population size per generation |
| `--db <path>` | `src/capacitor_balance/capacitors.db` | Input SQLite database |
| `--out <path>` | `src/capacitor_balance/parallel_results.xlsx` | Output Excel file |

The script prints the iteration number and best fitness score per generation, then writes the selected capacitors to the output Excel file (sheet `"Capacitors Balanced"`).

### Static Checks (`ruff` & `ty`)

```bash
uv run ruff check src/capacitor_balance/functional_program.py
uv run ty check src/capacitor_balance/functional_program.py
```

---

## Using the Mojo Solver (`.so` Extension & Standalone CLI)

### Prerequisites

- Installed automatically via `uv sync` (`mojo` nightly is listed in `pyproject.toml`)
- System `libsqlite3` (built-in on macOS; `libsqlite3.so.0` on Linux) for standalone CLI binaries

### Build the Python Extension (`.so`)

```bash
# Build native Python extension module (auto-built by the web server if missing)
uv run mojo build -I mojo_solver --emit shared-lib -O 3 --Werror \
  -o mojo_solver/capacitor_mojo.so mojo_solver/capacitor_mojo.mojo

# Call directly from Python (zero subprocess overhead)
PYTHONPATH=mojo_solver uv run python -c "
import capacitor_mojo
caps = [1.02 + (i % 7) * 0.001 for i in range(200)]
indices, fitness, solve_ms = capacitor_mojo.solve_residual(caps, 6, 9)
print(f'placed={len(indices)} fitness={fitness:.3e} solve_ms={solve_ms:.3f}')
"
```

### Build Standalone CLI Binaries (Optional)

```bash
# GA + 2-opt local search CLI
uv run mojo build -O 3 --Werror -o mojo_solver/solver mojo_solver/solver.mojo

# Alternative microsecond residual-balancing CLI
uv run mojo build -O 3 --Werror -o mojo_solver/residual_solver mojo_solver/residual_solver.mojo
```

### Run

```bash
# Alternative residual-balancing solver (~20–36 µs solve, exact balance across phases A, B, C)
./mojo_solver/residual_solver
./mojo_solver/residual_solver --series 4 --parallel 5
./mojo_solver/residual_solver --bench 1000

# Fast local-search solver (achieves exact 0.0 fitness in ~1.5 ms)
./mojo_solver/solver --mode fast
./mojo_solver/solver --mode fast --series 10 --parallel 10

# Genetic algorithm mode (exact 1,000 × 500 algorithm parity with Python)
./mojo_solver/solver --mode ga --series 6 --parallel 9 --seed 42
```

### CLI Options

| Flag | Applies To | Default | Description |
|---|---|---|---|
| `--series <int>` | `solver`, `residual_solver` | `6` | Series-connected parallel groups per phase (3 phases fixed → `3 × series` groups) |
| `--parallel <int>` | `solver`, `residual_solver` | `9` | Capacitors in parallel per group (`3 × series × parallel` total capacitors) |
| `--mode <fast\|ga>` | `solver` | `fast` | `fast`: greedy LPT + 2-opt swap & pool replacement; `ga`: genetic algorithm |
| `--iterations <int>` | `solver` (`ga`) | `500` | Number of GA iterations |
| `--generation-length <int>` | `solver` (`ga`) | `1000` | Population size per generation |
| `--bench <int>` | `residual_solver` | `1` | Repeat pure solve N times and print mean/min microseconds |
| `--db <path>` | `solver`, `residual_solver` | `src/capacitor_balance/capacitors.db` | Path to the SQLite database containing the `capacitors` table |
| `--out <path>` | `solver`, `residual_solver` | `mojo_solver/*_results.csv` | Destination CSV path for the ordered arrangement |
| `--seed <int>` | `solver` | `42` | Random seed for reproducible runs |
| `--verbose` | `solver` | off | Print per-iteration / per-restart fitness progress |

### Output Format (`mojo_results.csv`)

Each row assigns one physical capacitor to its exact slot in the bank:

```csv
position,phase,series_group,parallel_slot,db_idx,serial,capacitance,deviation
0,0,0,0,104,502143701-10-0089,1.023,4
```

- `phase`: `0` = Phase A, `1` = Phase B, `2` = Phase C
- `series_group`: `0..5` (6 series-connected parallel groups per phase)
- `parallel_slot`: `0..8` (9 parallel capacitors per group)

---

## Using the Web Visualizer & Lightweight Server

A zero-framework Python HTTP server (`server/app.py`, stdlib `ThreadingHTTPServer` + SQLite WAL persistence) serves the interactive web UI (`visualizer/index.html`) for 115 kV, 230 kV, or custom 3-phase (A, B, C) capacitor banks.

### Start the Server

```bash
# Sync dependencies (includes Mojo nightly, Polars, XlsxWriter)
uv sync

# Launch the web server (auto-builds mojo_solver/capacitor_mojo.so on first solve)
uv run capacitor-balance-server --host 127.0.0.1 --port 8080
# or: uv run python -m server.app --host 127.0.0.1 --port 8080
```

### Web App Features

- **Strict CSV / Excel (`.xlsx`) Upload & Validation**: Requires columns `serial` (unique non-empty string, ≤ 64 chars), `capacitance` (positive finite float), and `deviation` (integer). Downloadable `.csv` and `.xlsx` templates are available directly from the Upload dialog.
- **Empty Bank + Visual Stock on Upload**: Newly uploaded inventories start completely in the visual **Available Capacitor Stock** panel with an empty bank.
- **Speed-Ranked Solvers + Loading Indicator**: Choose from all four solvers sorted fastest → slowest with visual speed indicators (`Mojo Residual Balancer` → `Mojo Local Search (2-Opt)` → `Mojo Genetic Algorithm` → `Python + Polars GA`).
- **Interactive Bank Editing**: Drag-and-drop or click to **swap** capacitors, **remove** placed units back to Stock, or **add** units from Stock into any slot, with live recalculation of phase equivalents and unbalance.
- **Multi-Tab Workspaces (Max 10 Tabs)**: Balance up to 10 independent capacitor banks in separate tabs without losing configurations.
- **Concurrency Cap (Max 50 Users) & SQLite State**: All sessions, tabs, capacitor placements, and solve logs are persisted in `server/app_state.db`. Connections beyond 50 concurrent active users are rejected with `HTTP 503` ("Too Many Connections").

---

## Performance & Documentation

| Implementation | Mean Wall Time | Mean Solve Time | Mean Best Fitness |
|---|---:|---:|---:|
| **Python + Polars GA** | 18.43 s | ~18.4 s | `6.82 × 10⁻⁴` |
| **Mojo GA (`--mode ga`)** | 0.334 s *(55×)* | 0.304 s *(61×)* | `7.30 × 10⁻⁴` |
| **Mojo Fast (`--mode fast`)** | 17.7 ms *(1,043×)* | 1.46 ms *(12,666×)* | **`0.0`** *(exact balance)* |
| **Mojo Residual (`residual_solver`)** | 14.6 ms *(1,267×)* | **36.2 µs** cold / **19.7 µs** warm | **`1.67 × 10⁻¹⁵`** *(exact balance, 1 ULP)* |

- See [DESIGN.md](DESIGN.md) for how each implementation and optimization algorithm works.
- See [BENCHMARK.md](BENCHMARK.md) for full multi-trial benchmark results and cross-verification details.
