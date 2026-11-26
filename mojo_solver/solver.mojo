"""High-performance Mojo solver for Capacitor Balance.

Implements two solver modes on the same 3-phase x 6-series x 9-parallel
(162-capacitor) balancing problem as the Python/Polars version:

1. `ga`: Exact genetic algorithm parity (1000 population x 500 iterations,
   10% elite / 20% mutation / 30% shuffle / 10% crossover / 30% random) using
   flat contiguous arrays, SIMD reductions, and O(1) stamp-array membership.
2. `fast` (default): Greedy balanced initialization + incremental O(1) 2-opt
   inter-group swap & unused-pool replacement local search with multi-start
   perturbations.
"""

from std.ffi import OwnedDLHandle, c_int
from std.math import sqrt
from std.sys import CompilationTarget, argv
from std.time import perf_counter

comptime NUM_PHASES = 3
comptime GROUPS_PER_PHASE = 6
comptime CAPS_PER_GROUP = 9
comptime NUM_GROUPS = NUM_PHASES * GROUPS_PER_PHASE  # 18
comptime GENOMA_LENGTH = NUM_GROUPS * CAPS_PER_GROUP  # 162

# Genetic algorithm parameters (matching functional_program.py)
comptime GENERATION_LENGTH = 1000
comptime ITERATIONS = 500
comptime BEST_GENOMAS_LENGTH = 100
comptime MUTATION_LENGTH = 200
comptime SHUFFLE_LENGTH = 300
comptime CROSSOVER_LENGTH = 100
comptime RANDOM_LENGTH = 300


struct Rng(TrivialRegisterPassable):
    """Fast SplitMix64-seeded XorShift64* PRNG (zero global state)."""

    var state: UInt64

    def __init__(out self, seed: UInt64):
        var z = seed + 0x9E3779B97F4A7C15
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        self.state = z ^ (z >> 31)
        if self.state == 0:
            self.state = 0x2545F4914F6CDD1D

    @inline(.always)
    def next_u64(mut self) -> UInt64:
        var x = self.state
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.state = x
        return x * 0x2545F4914F6CDD1D

    @inline(.always)
    def next_int(mut self, bound: Int) -> Int:
        return Int(self.next_u64() % UInt64(bound))

    @inline(.always)
    def next_bit(mut self) -> Bool:
        return (self.next_u64() & 1) == 1


@fieldwise_init
struct CapacitorData(Movable):
    """Flat struct-of-arrays storage for all valid capacitors from SQLite."""

    var db_indices: List[Int]
    var serials: List[String]
    var caps: List[Float64]
    var deviations: List[Int]


@fieldwise_init
struct SolveResult(Movable):
    """Result returned by a solver run."""

    var best_genoma: List[Int32]
    var best_fitness: Float64
    var solve_seconds: Float64
    var iterations_run: Int


def load_capacitors(db_path: String) raises -> CapacitorData:
    """Load valid (non-NULL capacitance) capacitors directly via libsqlite3."""
    comptime SQLITE_LIB = (
        "libsqlite3.dylib" if CompilationTarget.is_macos() else "libsqlite3.so.0"
    )
    comptime SQLITE_OK: c_int = 0
    comptime SQLITE_ROW: c_int = 100
    comptime SQLITE_NULL: c_int = 5

    var lib = OwnedDLHandle(SQLITE_LIB)
    var db_handle: Int = 0
    var db_path_c = db_path.copy()
    var rc = lib.call["sqlite3_open", c_int](
        db_path_c.as_c_string_span().ptr(),
        Pointer(to=db_handle),
    )
    if rc != SQLITE_OK:
        raise Error("sqlite3_open failed with code " + String(rc))

    var sql = String(
        'SELECT "N°", "N° DE SERIE", "CAPACITANCIA ", "CAP. DESVIACION "'
        " FROM capacitors"
    )
    var stmt_handle: Int = 0
    rc = lib.call["sqlite3_prepare_v2", c_int](
        db_handle,
        sql.as_c_string_span().ptr(),
        c_int(-1),
        Pointer(to=stmt_handle),
        Int(0),
    )
    if rc != SQLITE_OK:
        _ = lib.call["sqlite3_close", c_int](db_handle)
        raise Error("sqlite3_prepare_v2 failed with code " + String(rc))

    var db_indices = List[Int]()
    var serials = List[String]()
    var caps = List[Float64]()
    var deviations = List[Int]()

    var row_idx = 0
    while lib.call["sqlite3_step", c_int](stmt_handle) == SQLITE_ROW:
        var col_type = lib.call["sqlite3_column_type", c_int](
            stmt_handle, c_int(2)
        )
        if col_type != SQLITE_NULL:
            var text_ptr = lib.call[
                "sqlite3_column_text",
                ImmPointer[UInt8, ImmUntrackedOrigin],
            ](stmt_handle, c_int(1))
            var text_len = Int(
                lib.call["sqlite3_column_bytes", c_int](stmt_handle, c_int(1))
            )
            var serial = String(
                StringSpan(
                    unsafe_from_utf8=Span(unsafe_ptr=text_ptr, length=text_len)
                )
            )
            var cap = lib.call["sqlite3_column_double", Float64](
                stmt_handle, c_int(2)
            )
            var dev = Int(
                lib.call["sqlite3_column_int", c_int](stmt_handle, c_int(3))
            )
            db_indices.append(row_idx)
            serials.append(serial^)
            caps.append(cap)
            deviations.append(dev)
        row_idx += 1

    _ = lib.call["sqlite3_finalize", c_int](stmt_handle)
    _ = lib.call["sqlite3_close", c_int](db_handle)

    return CapacitorData(
        db_indices=db_indices^,
        serials=serials^,
        caps=caps^,
        deviations=deviations^,
    )


@inline(.always)
def fitness_from_group_sums(
    group_sums: Array[Float64, NUM_GROUPS],
) -> Float64:
    """Compute exact ddof=1 fitness from the 18 parallel group sums.

    Uses two-pass variance so results remain accurate down to ~1e-16 without
    catastrophic cancellation.
    """
    # Parallel group mean & sample std (ddof=1 over 18 groups) using SIMD-4
    var sum_vec = SIMD[.float64, 4](0.0)
    comptime for g in range(0, 16, 4):
        sum_vec += SIMD[.float64, 4](
            group_sums[g],
            group_sums[g + 1],
            group_sums[g + 2],
            group_sums[g + 3],
        )
    var par_mean = (
        sum_vec.reduce_add() + group_sums[16] + group_sums[17]
    ) / Float64(NUM_GROUPS)

    var mean_vec = SIMD[.float64, 4](par_mean)
    var sq_vec = SIMD[.float64, 4](0.0)
    comptime for g in range(0, 16, 4):
        var diff = (
            SIMD[.float64, 4](
                group_sums[g],
                group_sums[g + 1],
                group_sums[g + 2],
                group_sums[g + 3],
            )
            - mean_vec
        )
        sq_vec += diff * diff
    var d16 = group_sums[16] - par_mean
    var d17 = group_sums[17] - par_mean
    var par_var = (sq_vec.reduce_add() + d16 * d16 + d17 * d17) / Float64(
        NUM_GROUPS - 1
    )
    var par_std = sqrt(par_var) if par_var > 0.0 else 0.0

    # Series reduction for each of the 3 phases (6 groups in series)
    var phase_caps = Array[Float64, NUM_PHASES](fill=0.0)
    comptime for p in range(NUM_PHASES):
        var base = p * GROUPS_PER_PHASE
        var v4 = SIMD[.float64, 4](
            group_sums[base],
            group_sums[base + 1],
            group_sums[base + 2],
            group_sums[base + 3],
        )
        var inv_sum = (
            (SIMD[.float64, 4](1.0) / v4).reduce_add()
            + 1.0 / group_sums[base + 4]
            + 1.0 / group_sums[base + 5]
        )
        phase_caps[p] = 1.0 / inv_sum

    # Phase sample std (ddof=1 over 3 phases)
    var phase_mean = (phase_caps[0] + phase_caps[1] + phase_caps[2]) / 3.0
    var dp0 = phase_caps[0] - phase_mean
    var dp1 = phase_caps[1] - phase_mean
    var dp2 = phase_caps[2] - phase_mean
    var phase_var = (dp0 * dp0 + dp1 * dp1 + dp2 * dp2) / 2.0
    var phase_std = sqrt(phase_var) if phase_var > 0.0 else 0.0

    return 20.0 * phase_std + par_std


@inline(.always)
def compute_fitness(
    genoma: Span[Int32, _],
    caps: Span[Float64, _],
) -> Float64:
    """Evaluate a 162-element genoma directly from capacitance array."""
    var group_sums = Array[Float64, NUM_GROUPS](fill=0.0)
    comptime for g in range(NUM_GROUPS):
        var base = g * CAPS_PER_GROUP
        var s = (
            caps[Int(genoma[base])]
            + caps[Int(genoma[base + 1])]
            + caps[Int(genoma[base + 2])]
            + caps[Int(genoma[base + 3])]
            + caps[Int(genoma[base + 4])]
            + caps[Int(genoma[base + 5])]
            + caps[Int(genoma[base + 6])]
            + caps[Int(genoma[base + 7])]
            + caps[Int(genoma[base + 8])]
        )
        group_sums[g] = s
    return fitness_from_group_sums(group_sums)


def fill_random_genoma(
    mut dst: List[Int32],
    dst_offset: Int,
    mut pool: List[Int32],
    mut rng: Rng,
):
    """Partial Fisher-Yates sample of GENOMA_LENGTH distinct indices."""
    var n = len(pool)
    for i in range(GENOMA_LENGTH):
        var j = i + rng.next_int(n - i)
        var tmp = pool[i]
        pool[i] = pool[j]
        pool[j] = tmp
        dst[dst_offset + i] = pool[i]


def run_ga(
    caps: List[Float64],
    iterations: Int,
    seed: UInt64,
    verbose: Bool = False,
) -> SolveResult:
    """Faithful Mojo port of functional_program.py's Genetic Algorithm."""
    comptime assert (
        GENERATION_LENGTH
        == BEST_GENOMAS_LENGTH
        + MUTATION_LENGTH
        + SHUFFLE_LENGTH
        + CROSSOVER_LENGTH
        + RANDOM_LENGTH
    ), "Generation distribution must sum to GENERATION_LENGTH"

    var t0 = perf_counter()
    var num_caps = len(caps)
    var rng = Rng(seed)

    # Double-buffered flat populations (GENERATION_LENGTH x GENOMA_LENGTH)
    var pop_a = List[Int32](
        length=GENERATION_LENGTH * GENOMA_LENGTH, fill=Int32(0)
    )
    var pop_b = List[Int32](
        length=GENERATION_LENGTH * GENOMA_LENGTH, fill=Int32(0)
    )
    var pool = List[Int32](capacity=num_caps)
    for i in range(num_caps):
        pool.append(Int32(i))

    # Initial random generation
    for g in range(GENERATION_LENGTH):
        fill_random_genoma(pop_a, g * GENOMA_LENGTH, pool, rng)

    var scores = List[Float64](length=GENERATION_LENGTH, fill=0.0)
    var order = List[Int](length=GENERATION_LENGTH, fill=0)
    var used_stamp = List[Int32](length=num_caps, fill=Int32(0))
    var avail_buf = List[Int32](capacity=num_caps)
    var stamp: Int32 = 0

    var caps_span = Span(caps)

    for it in range(iterations):
        # 1. Evaluate all genomas
        var pop_span = Span(pop_a)
        for g in range(GENERATION_LENGTH):
            var off = g * GENOMA_LENGTH
            scores[g] = compute_fitness(
                pop_span[off : off + GENOMA_LENGTH], caps_span
            )
            order[g] = g

        # 2. Sort indices by ascending fitness
        def cmp_idx(a: Int, b: Int) {imm scores} -> Bool:
            return scores[a] < scores[b]

        sort(order, cmp_idx)
        if verbose:
            print(it, "best fit ->", scores[order[0]])

        # 3. Build next generation in pop_b
        var out_idx = 0

        # 3a. Elitism (top BEST_GENOMAS_LENGTH)
        for k in range(BEST_GENOMAS_LENGTH):
            var src_off = order[k] * GENOMA_LENGTH
            var dst_off = out_idx * GENOMA_LENGTH
            for i in range(GENOMA_LENGTH):
                pop_b[dst_off + i] = pop_a[src_off + i]
            out_idx += 1

        # 3b. Mutation (MUTATION_LENGTH from top elites)
        for _ in range(MUTATION_LENGTH):
            var parent = order[rng.next_int(BEST_GENOMAS_LENGTH)]
            var src_off = parent * GENOMA_LENGTH
            var dst_off = out_idx * GENOMA_LENGTH
            stamp += 1
            for i in range(GENOMA_LENGTH):
                var c = pop_a[src_off + i]
                pop_b[dst_off + i] = c
                used_stamp[Int(c)] = stamp

            avail_buf.clear()
            for c_idx in range(num_caps):
                if used_stamp[c_idx] != stamp:
                    avail_buf.append(Int32(c_idx))

            if len(avail_buf) > 0:
                var pos = rng.next_int(GENOMA_LENGTH)
                pop_b[dst_off + pos] = avail_buf[rng.next_int(len(avail_buf))]
            out_idx += 1

        # 3c. Shuffle (SHUFFLE_LENGTH from top elites)
        for _ in range(SHUFFLE_LENGTH):
            var parent = order[rng.next_int(BEST_GENOMAS_LENGTH)]
            var src_off = parent * GENOMA_LENGTH
            var dst_off = out_idx * GENOMA_LENGTH
            for i in range(GENOMA_LENGTH):
                pop_b[dst_off + i] = pop_a[src_off + i]
            for i in range(GENOMA_LENGTH - 1, 0, -1):
                var j = rng.next_int(i + 1)
                var tmp = pop_b[dst_off + i]
                pop_b[dst_off + i] = pop_b[dst_off + j]
                pop_b[dst_off + j] = tmp
            out_idx += 1

        # 3d. Random (RANDOM_LENGTH fresh random genomas)
        for _ in range(RANDOM_LENGTH):
            fill_random_genoma(pop_b, out_idx * GENOMA_LENGTH, pool, rng)
            out_idx += 1

        # 3e. Crossover (CROSSOVER_LENGTH between top elite & past generation)
        for _ in range(CROSSOVER_LENGTH):
            var p1 = order[rng.next_int(BEST_GENOMAS_LENGTH)]
            var p2 = rng.next_int(GENERATION_LENGTH)
            var off1 = p1 * GENOMA_LENGTH
            var off2 = p2 * GENOMA_LENGTH
            var dst_off = out_idx * GENOMA_LENGTH
            stamp += 1
            for i in range(GENOMA_LENGTH):
                used_stamp[Int(pop_a[off1 + i])] = stamp
            for i in range(GENOMA_LENGTH):
                var g1 = pop_a[off1 + i]
                var g2 = pop_a[off2 + i]
                if used_stamp[Int(g2)] != stamp and rng.next_bit():
                    pop_b[dst_off + i] = g2
                else:
                    pop_b[dst_off + i] = g1
            out_idx += 1

        swap(pop_a, pop_b)

    # Final evaluation to pick best from last generation
    var final_span = Span(pop_a)
    var best_idx = 0
    var best_fit = 1e30
    for g in range(GENERATION_LENGTH):
        var off = g * GENOMA_LENGTH
        var f = compute_fitness(
            final_span[off : off + GENOMA_LENGTH], caps_span
        )
        if f < best_fit:
            best_fit = f
            best_idx = g

    var best_genoma = List[Int32](capacity=GENOMA_LENGTH)
    var best_off = best_idx * GENOMA_LENGTH
    for i in range(GENOMA_LENGTH):
        best_genoma.append(pop_a[best_off + i])

    var elapsed = perf_counter() - t0
    return SolveResult(
        best_genoma=best_genoma^,
        best_fitness=best_fit,
        solve_seconds=elapsed,
        iterations_run=iterations,
    )


def greedy_balanced_init(
    caps: List[Float64],
    mut selected: List[Int32],
    mut unused: List[Int32],
    mut group_sums: Array[Float64, NUM_GROUPS],
    mut rng: Rng,
    randomized: Bool,
):
    """Initialize 18 groups of 9 capacitors via balanced LPT assignment."""
    var num_caps = len(caps)
    var order = List[Int](capacity=num_caps)
    for i in range(num_caps):
        order.append(i)

    if randomized:
        # Random subset of 162 capacitors, then sort descending for LPT packing
        for i in range(GENOMA_LENGTH):
            var j = i + rng.next_int(num_caps - i)
            var tmp = order[i]
            order[i] = order[j]
            order[j] = tmp
    else:
        # Sort all capacitors by capacitance and pick the middle 162 (tightest spread)
        def cmp_cap(a: Int, b: Int) {imm caps} -> Bool:
            return caps[a] < caps[b]

        sort(order, cmp_cap)
        var start = (num_caps - GENOMA_LENGTH) // 2
        var chosen = List[Int](capacity=num_caps)
        for i in range(GENOMA_LENGTH):
            chosen.append(order[start + i])
        for i in range(start):
            chosen.append(order[i])
        for i in range(start + GENOMA_LENGTH, num_caps):
            chosen.append(order[i])
        order = chosen^

    # Sort the chosen 162 indices descending by capacitance for LPT packing
    var chosen_162 = List[Int](capacity=GENOMA_LENGTH)
    for i in range(GENOMA_LENGTH):
        chosen_162.append(order[i])

    def cmp_desc(a: Int, b: Int) {imm caps} -> Bool:
        return caps[a] > caps[b]

    sort(chosen_162, cmp_desc)

    unused.clear()
    for i in range(GENOMA_LENGTH, num_caps):
        unused.append(Int32(order[i]))

    var counts = Array[Int, NUM_GROUPS](fill=0)
    for g in range(NUM_GROUPS):
        group_sums[g] = 0.0

    for idx_pos in range(GENOMA_LENGTH):
        var c_idx = chosen_162[idx_pos]
        var best_g = -1
        var min_sum = 1e30
        for g in range(NUM_GROUPS):
            if counts[g] < CAPS_PER_GROUP and group_sums[g] < min_sum:
                min_sum = group_sums[g]
                best_g = g
        var slot = best_g * CAPS_PER_GROUP + counts[best_g]
        selected[slot] = Int32(c_idx)
        counts[best_g] += 1
        group_sums[best_g] += caps[c_idx]


def run_fast_solver(
    caps: List[Float64],
    max_restarts: Int = 64,
    seed: UInt64 = 42,
    verbose: Bool = False,
) -> SolveResult:
    """High-speed Mojo local search solver with O(1) incremental evaluation.

    Combines greedy LPT initialization with exhaustive first-improvement 2-opt
    inter-group swaps and unused-pool replacements, restarting on plateau until
    reaching machine-precision balance (<= 1e-15) or max_restarts.
    """
    var t0 = perf_counter()
    var num_caps = len(caps)
    var num_unused = num_caps - GENOMA_LENGTH
    var rng = Rng(seed)

    var selected = List[Int32](length=GENOMA_LENGTH, fill=Int32(0))
    var unused = List[Int32](capacity=num_unused)
    var group_sums = Array[Float64, NUM_GROUPS](fill=0.0)

    var global_best_genoma = List[Int32](length=GENOMA_LENGTH, fill=Int32(0))
    var global_best_fit = 1e30
    var sweeps_total = 0

    comptime TARGET_EPS = 1e-15

    for restart in range(max_restarts):
        greedy_balanced_init(
            caps,
            selected,
            unused,
            group_sums,
            rng,
            randomized=(restart > 0),
        )
        var cur_fit = fitness_from_group_sums(group_sums)

        # Iterative local search (steepest / first-improvement sweeps)
        var improved = True
        while improved and cur_fit > TARGET_EPS:
            improved = False
            sweeps_total += 1

            # 1. Pool replacement moves: replace selected[a] with unused[u]
            for g in range(NUM_GROUPS):
                var base_g = g * CAPS_PER_GROUP
                for k in range(CAPS_PER_GROUP):
                    var pos_a = base_g + k
                    var c_a = Int(selected[pos_a])
                    var cap_a = caps[c_a]
                    var old_sum_g = group_sums[g]

                    var best_u = -1
                    var best_u_fit = cur_fit - 1e-18

                    for u in range(num_unused):
                        var c_u = Int(unused[u])
                        var cap_u = caps[c_u]
                        if cap_u == cap_a:
                            continue
                        group_sums[g] = old_sum_g - cap_a + cap_u
                        var cand_fit = fitness_from_group_sums(group_sums)
                        if cand_fit < best_u_fit:
                            best_u_fit = cand_fit
                            best_u = u

                    if best_u >= 0:
                        var c_u = Int(unused[best_u])
                        group_sums[g] = old_sum_g - cap_a + caps[c_u]
                        selected[pos_a] = Int32(c_u)
                        unused[best_u] = Int32(c_a)
                        cur_fit = best_u_fit
                        improved = True
                        if cur_fit <= TARGET_EPS:
                            break
                    else:
                        group_sums[g] = old_sum_g
                if cur_fit <= TARGET_EPS:
                    break

            if cur_fit <= TARGET_EPS:
                break

            # 2. Inter-group 2-opt swaps: swap selected[pos_a] and selected[pos_b]
            for g1 in range(NUM_GROUPS - 1):
                var base1 = g1 * CAPS_PER_GROUP
                for k1 in range(CAPS_PER_GROUP):
                    var pos_a = base1 + k1
                    var c_a = Int(selected[pos_a])
                    var cap_a = caps[c_a]
                    var old_sum1 = group_sums[g1]

                    var best_pos_b = -1
                    var best_g2 = -1
                    var best_swap_fit = cur_fit - 1e-18

                    for g2 in range(g1 + 1, NUM_GROUPS):
                        var base2 = g2 * CAPS_PER_GROUP
                        var old_sum2 = group_sums[g2]
                        for k2 in range(CAPS_PER_GROUP):
                            var pos_b = base2 + k2
                            var c_b = Int(selected[pos_b])
                            var cap_b = caps[c_b]
                            var delta = cap_b - cap_a
                            if delta == 0.0:
                                continue
                            group_sums[g1] = old_sum1 + delta
                            group_sums[g2] = old_sum2 - delta
                            var cand_fit = fitness_from_group_sums(group_sums)
                            if cand_fit < best_swap_fit:
                                best_swap_fit = cand_fit
                                best_pos_b = pos_b
                                best_g2 = g2
                        group_sums[g2] = old_sum2

                    if best_pos_b >= 0:
                        var c_b = Int(selected[best_pos_b])
                        var delta = caps[c_b] - cap_a
                        group_sums[g1] = old_sum1 + delta
                        group_sums[best_g2] -= delta
                        selected[pos_a] = Int32(c_b)
                        selected[best_pos_b] = Int32(c_a)
                        cur_fit = best_swap_fit
                        improved = True
                        if cur_fit <= TARGET_EPS:
                            break
                    else:
                        group_sums[g1] = old_sum1
                if cur_fit <= TARGET_EPS:
                    break

        # Recompute from scratch to avoid any accumulated float drift
        var exact_fit = compute_fitness(Span(selected), Span(caps))
        if exact_fit < global_best_fit:
            global_best_fit = exact_fit
            for i in range(GENOMA_LENGTH):
                global_best_genoma[i] = selected[i]
            if verbose:
                print("restart", restart, "best fit ->", global_best_fit)

        if global_best_fit <= TARGET_EPS:
            break

    var elapsed = perf_counter() - t0
    return SolveResult(
        best_genoma=global_best_genoma^,
        best_fitness=global_best_fit,
        solve_seconds=elapsed,
        iterations_run=sweeps_total,
    )


def save_results_csv(
    out_path: String,
    data: CapacitorData,
    genoma: List[Int32],
) raises:
    """Write the ordered 162-capacitor solution to CSV."""
    var lines = String(
        "position,phase,series_group,parallel_slot,db_idx,serial,capacitance,deviation\n"
    )
    for pos in range(GENOMA_LENGTH):
        var c = Int(genoma[pos])
        var group = pos // CAPS_PER_GROUP
        var phase = group // GROUPS_PER_PHASE
        var series_group = group % GROUPS_PER_PHASE
        var slot = pos % CAPS_PER_GROUP
        lines += (
            String(pos)
            + ","
            + String(phase)
            + ","
            + String(series_group)
            + ","
            + String(slot)
            + ","
            + String(data.db_indices[c])
            + ","
            + data.serials[c]
            + ","
            + String(data.caps[c])
            + ","
            + String(data.deviations[c])
            + "\n"
        )
    with open(out_path, "w") as f:
        f.write(lines)


def main() raises:
    var args = argv()
    var mode = String("fast")
    var db_path = String("src/capacitor_balance/capacitors.db")
    var out_path = String("mojo_solver/mojo_results.csv")
    var seed: UInt64 = 42
    var verbose = False

    var i = 1
    while i < len(args):
        var arg = String(args[i])
        if arg == "--mode" and i + 1 < len(args):
            mode = String(args[i + 1])
            i += 2
        elif arg == "--db" and i + 1 < len(args):
            db_path = String(args[i + 1])
            i += 2
        elif arg == "--out" and i + 1 < len(args):
            out_path = String(args[i + 1])
            i += 2
        elif arg == "--seed" and i + 1 < len(args):
            seed = UInt64(Int(String(args[i + 1])))
            i += 2
        elif arg == "--verbose":
            verbose = True
            i += 1
        else:
            i += 1

    var wall_t0 = perf_counter()
    var data = load_capacitors(db_path)

    var result: SolveResult
    if mode == "ga":
        result = run_ga(data.caps, ITERATIONS, seed, verbose)
    else:
        result = run_fast_solver(data.caps, 64, seed, verbose)

    save_results_csv(out_path, data, result.best_genoma)
    var wall_elapsed = perf_counter() - wall_t0

    print(
        "mode="
        + mode
        + " best_fit="
        + String(result.best_fitness)
        + " solve_s="
        + String(result.solve_seconds)
        + " wall_s="
        + String(wall_elapsed)
        + " iters="
        + String(result.iterations_run)
    )
