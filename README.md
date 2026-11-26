# Capacitor Balance

A 3-phase (**A, B, C**) series-parallel capacitor bank balancer.

Given measured capacitances in `src/capacitor_balance/capacitors.db`, this project selects and arranges **162 distinct capacitors** into **3 phases × 6 series groups × 9 parallel capacitors** so that all 18 parallel groups share capacitance evenly and all three phases (A, B, and C) achieve near-zero phase-to-phase unbalance.

Both a **Python + Polars** solver and a **native Mojo** solver are included.

## Project Layout

```text
capacitor_balance/
├── src/capacitor_balance/
│   ├── capacitors.db           # Input SQLite database of measured capacitors
│   ├── functional_program.py   # Async Python + vectorized Polars GA solver
│   └── parallel_results.xlsx   # Excel output written by the Python solver
├── mojo_solver/
│   ├── solver.mojo             # Native Mojo solver (--mode fast | --mode ga)
│   ├── residual_solver.mojo    # Alternative sub-millisecond residual-balancing solver
│   ├── mojo_results.csv        # CSV output written by solver.mojo
│   └── residual_results.csv    # CSV output written by residual_solver.mojo
├── DESIGN.md                   # Architecture & algorithm design details
├── BENCHMARK.md                # Timing & accuracy comparison (Python vs. Mojo)
└── pyproject.toml              # Python dependencies, ruff & ty configuration
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
# Sync dependencies (polars, xlsxwriter)
uv sync

# Run the genetic algorithm (1,000 population × 500 iterations)
PYTHONPATH=src uv run python src/capacitor_balance/functional_program.py
```

The script prints the iteration number and best fitness score per generation, then writes the selected 162 capacitors to `src/capacitor_balance/parallel_results.xlsx` (sheet `"Capacitors Balanced"`).

### Static Checks (`ruff` & `ty`)

```bash
uv run ruff check src/capacitor_balance/functional_program.py
uv run ty check src/capacitor_balance/functional_program.py
```

---

## Using the Mojo Solver

### Prerequisites

- [Mojo](https://www.modular.com/mojo) (`>= 1.3.0.dev`, tested with `Mojo 1.3.0.dev2026112405`)
- System `libsqlite3` (built-in on macOS; `libsqlite3.so.0` on Linux)

### Build

```bash
# GA + 2-opt local search solver
mojo build -O 3 --Werror -o mojo_solver/solver mojo_solver/solver.mojo

# Alternative microsecond residual-balancing solver (fastest)
mojo build -O 3 --Werror -o mojo_solver/residual_solver mojo_solver/residual_solver.mojo
```

### Run

```bash
# Alternative residual-balancing solver (~20–36 µs solve, exact balance across phases A, B, C)
./mojo_solver/residual_solver
./mojo_solver/residual_solver --bench 1000

# Fast local-search solver (achieves exact 0.0 fitness in ~1.5 ms)
./mojo_solver/solver --mode fast

# Genetic algorithm mode (exact 1,000 × 500 algorithm parity with Python)
./mojo_solver/solver --mode ga --seed 42
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--mode <fast\|ga>` | `fast` | `fast`: greedy LPT + incremental 2-opt swap & pool replacement; `ga`: 1,000 × 500 genetic algorithm |
| `--db <path>` | `src/capacitor_balance/capacitors.db` | Path to the SQLite database containing the `capacitors` table |
| `--out <path>` | `mojo_solver/mojo_results.csv` | Destination CSV path for the ordered 162-capacitor arrangement |
| `--seed <int>` | `42` | Random seed for reproducible runs |
| `--verbose` | off | Print per-iteration / per-restart fitness progress |

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

## Performance & Documentation

| Implementation | Mean Wall Time | Mean Solve Time | Mean Best Fitness |
|---|---:|---:|---:|
| **Python + Polars GA** | 18.43 s | ~18.4 s | `6.82 × 10⁻⁴` |
| **Mojo GA (`--mode ga`)** | 0.334 s *(55×)* | 0.304 s *(61×)* | `7.30 × 10⁻⁴` |
| **Mojo Fast (`--mode fast`)** | 17.7 ms *(1,043×)* | 1.46 ms *(12,666×)* | **`0.0`** *(exact balance)* |
| **Mojo Residual (`residual_solver`)** | 14.6 ms *(1,267×)* | **36.2 µs** cold / **19.7 µs** warm | **`1.67 × 10⁻¹⁵`** *(exact balance, 1 ULP)* |

- See [DESIGN.md](DESIGN.md) for how each implementation and optimization algorithm works.
- See [BENCHMARK.md](BENCHMARK.md) for full multi-trial benchmark results and cross-verification details.
