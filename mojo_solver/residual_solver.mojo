"""Alternative ultra-fast Mojo solver for 3-phase (A, B, C) Capacitor Balance.

Supports arbitrary 3-phase bank sizes via `--series` (series groups per phase)
and `--parallel` (capacitors in parallel per group), always using 3 phases
(total bank size = 3 * series * parallel).

Algorithm: Tightest-Window Snake/LPT Construction + Targeted Residual Balancing
+ Phase-Reciprocal Assignment (sub-millisecond).
"""

from std.ffi import OwnedDLHandle, c_int
from std.math import abs, sqrt
from std.sys import CompilationTarget, argv
from std.time import perf_counter, perf_counter_ns

comptime NUM_PHASES = 3
comptime DEFAULT_SERIES = 6
comptime DEFAULT_PARALLEL = 9
comptime TARGET_EPS = 1e-15


struct BankConfig(TrivialRegisterPassable):
    """3-phase (A, B, C) series-parallel capacitor bank dimensions."""

    var groups_per_phase: Int
    var caps_per_group: Int
    var num_groups: Int
    var genoma_length: Int

    def __init__(out self, groups_per_phase: Int, caps_per_group: Int):
        self.groups_per_phase = groups_per_phase
        self.caps_per_group = caps_per_group
        self.num_groups = NUM_PHASES * groups_per_phase
        self.genoma_length = self.num_groups * caps_per_group


@fieldwise_init
struct CapacitorData(Movable):
    """Flat struct-of-arrays storage for valid capacitors from SQLite."""

    var db_indices: List[Int]
    var serials: List[String]
    var caps: List[Float64]
    var deviations: List[Int]


@fieldwise_init
struct BalanceResult(Movable):
    """Result from the residual-balancing solver."""

    var genoma: List[Int32]
    var fitness: Float64
    var par_std: Float64
    var phase_std: Float64
    var solve_ns: Int
    var moves: Int


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
def evaluate_exact(
    genoma: Span[Int32, _],
    caps: Span[Float64, _],
    config: BankConfig,
) -> Tuple[Float64, Float64, Float64]:
    """Compute exact (fitness, par_std, phase_std) using two-pass SIMD-4."""
    var num_groups = config.num_groups
    var groups_per_phase = config.groups_per_phase
    var caps_per_group = config.caps_per_group

    var group_sums = List[Float64](length=num_groups, fill=0.0)
    for g in range(num_groups):
        var base = g * caps_per_group
        var s = 0.0
        for k in range(caps_per_group):
            s += caps[Int(genoma[base + k])]
        group_sums[g] = s

    var sum_vec = SIMD[.float64, 4](0.0)
    var g = 0
    while g + 4 <= num_groups:
        sum_vec += SIMD[.float64, 4](
            group_sums[g],
            group_sums[g + 1],
            group_sums[g + 2],
            group_sums[g + 3],
        )
        g += 4
    var par_sum = sum_vec.reduce_add()
    while g < num_groups:
        par_sum += group_sums[g]
        g += 1
    var par_mean = par_sum / Float64(num_groups)

    var mean_vec = SIMD[.float64, 4](par_mean)
    var sq_vec = SIMD[.float64, 4](0.0)
    g = 0
    while g + 4 <= num_groups:
        var d = (
            SIMD[.float64, 4](
                group_sums[g],
                group_sums[g + 1],
                group_sums[g + 2],
                group_sums[g + 3],
            )
            - mean_vec
        )
        sq_vec += d * d
        g += 4
    var par_sq = sq_vec.reduce_add()
    while g < num_groups:
        var d = group_sums[g] - par_mean
        par_sq += d * d
        g += 1
    var par_var = par_sq / Float64(num_groups - 1)
    var par_std = sqrt(par_var) if par_var > 0.0 else 0.0

    var phase_caps = Array[Float64, NUM_PHASES](fill=0.0)
    comptime for p in range(NUM_PHASES):
        var base = p * groups_per_phase
        var inv_vec = SIMD[.float64, 4](0.0)
        var s = 0
        while s + 4 <= groups_per_phase:
            var v4 = SIMD[.float64, 4](
                group_sums[base + s],
                group_sums[base + s + 1],
                group_sums[base + s + 2],
                group_sums[base + s + 3],
            )
            inv_vec += SIMD[.float64, 4](1.0) / v4
            s += 4
        var inv_sum = inv_vec.reduce_add()
        while s < groups_per_phase:
            inv_sum += 1.0 / group_sums[base + s]
            s += 1
        phase_caps[p] = 1.0 / inv_sum

    var phase_mean = (phase_caps[0] + phase_caps[1] + phase_caps[2]) / 3.0
    var dp0 = phase_caps[0] - phase_mean
    var dp1 = phase_caps[1] - phase_mean
    var dp2 = phase_caps[2] - phase_mean
    var phase_var = (dp0 * dp0 + dp1 * dp1 + dp2 * dp2) / 2.0
    var phase_std = sqrt(phase_var) if phase_var > 0.0 else 0.0

    return (20.0 * phase_std + par_std, par_std, phase_std)


@inline(.always)
def lower_bound_unused(
    unused: List[Int32],
    caps: Span[Float64, _],
    target: Float64,
) -> Int:
    """Binary search for first index in sorted unused with caps[idx] >= target.
    """
    var lo = 0
    var hi = len(unused)
    while lo < hi:
        var mid = (lo + hi) >> 1
        if caps[Int(unused[mid])] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


@inline(.always)
def replace_in_sorted_unused(
    mut unused: List[Int32],
    caps: Span[Float64, _],
    pos: Int,
    new_cap_idx: Int32,
):
    """Replace unused[pos] with new_cap_idx and restore sorted order in O(shift).
    """
    var n = len(unused)
    var new_val = caps[Int(new_cap_idx)]
    var i = pos
    while i + 1 < n and caps[Int(unused[i + 1])] < new_val:
        unused[i] = unused[i + 1]
        i += 1
    while i > 0 and caps[Int(unused[i - 1])] > new_val:
        unused[i] = unused[i - 1]
        i -= 1
    unused[i] = new_cap_idx


def balance_phases_by_reciprocal(
    mut genoma: List[Int32],
    mut group_sums: List[Float64],
    config: BankConfig,
):
    """Assign parallel groups across Phases A, B, C to equalize 1 / S_p."""
    var num_groups = config.num_groups
    var groups_per_phase = config.groups_per_phase
    var caps_per_group = config.caps_per_group

    var inv_g = List[Float64](length=num_groups, fill=0.0)
    var g_order = List[Int](capacity=num_groups)
    for g in range(num_groups):
        inv_g[g] = 1.0 / group_sums[g]
        g_order.append(g)

    def cmp_inv_desc(a: Int, b: Int) {imm inv_g} -> Bool:
        return inv_g[a] > inv_g[b]

    sort(g_order, cmp_inv_desc)

    var phase_groups = List[Int](length=num_groups, fill=0)
    var phase_counts = Array[Int, NUM_PHASES](fill=0)
    var phase_inv = Array[Float64, NUM_PHASES](fill=0.0)

    for idx in range(num_groups):
        var g = g_order[idx]
        var best_p = -1
        var min_inv = 1e30
        for p in range(NUM_PHASES):
            if phase_counts[p] < groups_per_phase and phase_inv[p] < min_inv:
                min_inv = phase_inv[p]
                best_p = p
        phase_groups[best_p * groups_per_phase + phase_counts[best_p]] = g
        phase_counts[best_p] += 1
        phase_inv[best_p] += inv_g[g]

    var improved = True
    while improved:
        improved = False
        var p_max = 0
        var p_min = 0
        for p in range(1, NUM_PHASES):
            if phase_inv[p] > phase_inv[p_max]:
                p_max = p
            if phase_inv[p] < phase_inv[p_min]:
                p_min = p
        var gap = phase_inv[p_max] - phase_inv[p_min]
        if gap <= TARGET_EPS:
            break
        var ideal = 0.5 * gap
        var best_s1 = -1
        var best_s2 = -1
        var best_err = ideal - 1e-18
        for s1 in range(groups_per_phase):
            var g1 = phase_groups[p_max * groups_per_phase + s1]
            for s2 in range(groups_per_phase):
                var g2 = phase_groups[p_min * groups_per_phase + s2]
                var delta = inv_g[g1] - inv_g[g2]
                if delta > 0.0 and delta < gap:
                    var err = abs(delta - ideal)
                    if err < best_err:
                        best_err = err
                        best_s1 = s1
                        best_s2 = s2
        if best_s1 >= 0:
            var idx1 = p_max * groups_per_phase + best_s1
            var idx2 = p_min * groups_per_phase + best_s2
            var g1 = phase_groups[idx1]
            var g2 = phase_groups[idx2]
            var delta = inv_g[g1] - inv_g[g2]
            phase_groups[idx1] = g2
            phase_groups[idx2] = g1
            phase_inv[p_max] -= delta
            phase_inv[p_min] += delta
            improved = True

    var reordered = List[Int32](length=config.genoma_length, fill=Int32(0))
    var new_sums = List[Float64](length=num_groups, fill=0.0)
    for new_g in range(num_groups):
        var old_g = phase_groups[new_g]
        new_sums[new_g] = group_sums[old_g]
        var src = old_g * caps_per_group
        var dst = new_g * caps_per_group
        for k in range(caps_per_group):
            reordered[dst + k] = genoma[src + k]
    genoma = reordered^
    group_sums = new_sums^


def solve_residual_balance(
    caps: List[Float64],
    config: BankConfig,
) -> BalanceResult:
    """Solve 3-phase capacitor balance via targeted residual matching."""
    var t0_ns = perf_counter_ns()
    var num_caps = len(caps)
    var num_groups = config.num_groups
    var caps_per_group = config.caps_per_group
    var genoma_len = config.genoma_length
    var caps_span = Span(caps)

    # 1. Sort all valid capacitor indices by ascending capacitance: O(N log N)
    var sorted_all = List[Int](capacity=num_caps)
    for i in range(num_caps):
        sorted_all.append(i)

    def cmp_asc(a: Int, b: Int) {imm caps} -> Bool:
        return caps[a] < caps[b]

    sort(sorted_all, cmp_asc)

    # 2. Pick the contiguous genoma_len-unit window with minimum spread
    var best_start = 0
    var best_range = 1e30
    var max_start = num_caps - genoma_len
    for s in range(max_start + 1):
        var r = caps[sorted_all[s + genoma_len - 1]] - caps[sorted_all[s]]
        if r < best_range:
            best_range = r
            best_start = s

    # Partition into sorted chosen[genoma_len] and sorted unused[N - genoma_len]
    var chosen = List[Int32](capacity=genoma_len)
    var unused = List[Int32](capacity=num_caps - genoma_len)
    for i in range(best_start):
        unused.append(Int32(sorted_all[i]))
    for i in range(best_start, best_start + genoma_len):
        chosen.append(Int32(sorted_all[i]))
    for i in range(best_start + genoma_len, num_caps):
        unused.append(Int32(sorted_all[i]))

    # 3. Snake assignment for first (caps_per_group - 1) layers + LPT for final layer
    var genoma = List[Int32](length=genoma_len, fill=Int32(0))
    var group_sums = List[Float64](length=num_groups, fill=0.0)

    for layer in range(caps_per_group - 1):
        var base_idx = layer * num_groups
        if (layer & 1) == 0:
            for g in range(num_groups):
                var c = chosen[base_idx + g]
                genoma[g * caps_per_group + layer] = c
                group_sums[g] += caps[Int(c)]
        else:
            for g in range(num_groups):
                var c = chosen[base_idx + (num_groups - 1 - g)]
                genoma[g * caps_per_group + layer] = c
                group_sums[g] += caps[Int(c)]

    var g_order = List[Int](capacity=num_groups)
    for g in range(num_groups):
        g_order.append(g)

    def cmp_gsum(a: Int, b: Int) {imm group_sums} -> Bool:
        return group_sums[a] < group_sums[b]

    sort(g_order, cmp_gsum)
    var last_base = (caps_per_group - 1) * num_groups
    for rank in range(num_groups):
        var g = g_order[rank]
        var c = chosen[last_base + (num_groups - 1 - rank)]
        genoma[g * caps_per_group + (caps_per_group - 1)] = c
        group_sums[g] += caps[Int(c)]

    # 4. Targeted Residual Balancing Loop (Max-Min Cancel + Binary-Search Pool)
    var moves = 0
    var num_unused = len(unused)
    var max_moves = 2000
    var sse_factor = 1.0 - 1.0 / Float64(num_groups)

    while moves < max_moves:
        var g_max = 0
        var g_min = 0
        var total_sum = group_sums[0]
        for g in range(1, num_groups):
            var s = group_sums[g]
            total_sum += s
            if s > group_sums[g_max]:
                g_max = g
            if s < group_sums[g_min]:
                g_min = g

        var gap = group_sums[g_max] - group_sums[g_min]
        if gap <= TARGET_EPS:
            break

        var target = total_sum / Float64(num_groups)

        # 4a. Algebraic max-min 2-opt cancellation
        var best_k1 = -1
        var best_k2 = -1
        var best_delta_sse = -1e-18
        var base_max = g_max * caps_per_group
        var base_min = g_min * caps_per_group

        for k1 in range(caps_per_group):
            var c_a = caps[Int(genoma[base_max + k1])]
            for k2 in range(caps_per_group):
                var d = c_a - caps[Int(genoma[base_min + k2])]
                if d > 0.0 and d < gap:
                    var d_sse = 2.0 * d * (d - gap)
                    if d_sse < best_delta_sse:
                        best_delta_sse = d_sse
                        best_k1 = k1
                        best_k2 = k2

        # 4b. O(log M) binary-searched pool replacement on g_max and g_min
        var best_pool_g = -1
        var best_pool_k = -1
        var best_pool_u = -1
        var best_pool_sse = best_delta_sse

        if num_unused > 0:
            var cand_groups: Array[Int, 2] = [g_max, g_min]
            for cg_i in range(2):
                var g = cand_groups[cg_i]
                var resid = group_sums[g] - target
                var base_g = g * caps_per_group
                var d_ideal = -resid / sse_factor
                for k in range(caps_per_group):
                    var c_old = caps[Int(genoma[base_g + k])]
                    var want = c_old + d_ideal
                    var lb = lower_bound_unused(unused, caps_span, want)
                    for probe in range(lb - 1, lb + 1):
                        if probe >= 0 and probe < num_unused:
                            var c_new = caps[Int(unused[probe])]
                            var d = c_new - c_old
                            if d != 0.0:
                                var d_sse = d * (2.0 * resid + d * sse_factor)
                                if d_sse < best_pool_sse:
                                    best_pool_sse = d_sse
                                    best_pool_g = g
                                    best_pool_k = k
                                    best_pool_u = probe

        if best_pool_g >= 0:
            var slot = best_pool_g * caps_per_group + best_pool_k
            var old_idx = genoma[slot]
            var new_idx = unused[best_pool_u]
            group_sums[best_pool_g] += caps[Int(new_idx)] - caps[Int(old_idx)]
            genoma[slot] = new_idx
            replace_in_sorted_unused(unused, caps_span, best_pool_u, old_idx)
            moves += 1
        elif best_k1 >= 0:
            var slot1 = base_max + best_k1
            var slot2 = base_min + best_k2
            var idx1 = genoma[slot1]
            var idx2 = genoma[slot2]
            var d = caps[Int(idx1)] - caps[Int(idx2)]
            genoma[slot1] = idx2
            genoma[slot2] = idx1
            group_sums[g_max] -= d
            group_sums[g_min] += d
            moves += 1
        elif num_unused > 0:
            # 4c. Fallback: pull non-target groups onto the modal group sum
            var modal_target = group_sums[g_max]
            if abs(group_sums[g_min] - target) < abs(
                group_sums[g_max] - target
            ):
                modal_target = group_sums[g_min]

            var found_fallback = False
            for g in range(num_groups):
                var need = modal_target - group_sums[g]
                if abs(need) <= TARGET_EPS:
                    continue
                var base_g = g * caps_per_group
                for k in range(caps_per_group):
                    var c_old = caps[Int(genoma[base_g + k])]
                    var want = c_old + need
                    var lb = lower_bound_unused(unused, caps_span, want)
                    for probe in range(lb - 1, lb + 1):
                        if probe >= 0 and probe < num_unused:
                            var c_new = caps[Int(unused[probe])]
                            var new_err = abs(
                                (group_sums[g] - c_old + c_new) - modal_target
                            )
                            if new_err < abs(need) - 1e-12:
                                var slot = base_g + k
                                var old_idx = genoma[slot]
                                var new_idx = unused[probe]
                                group_sums[g] += c_new - c_old
                                genoma[slot] = new_idx
                                replace_in_sorted_unused(
                                    unused, caps_span, probe, old_idx
                                )
                                moves += 1
                                found_fallback = True
                                break
                    if found_fallback:
                        break
                if found_fallback:
                    break
            if not found_fallback:
                break
        else:
            break

    # 5. Arrange parallel groups across Phases A, B, C if any spread remains
    var min_g = group_sums[0]
    var max_g = group_sums[0]
    for g in range(1, num_groups):
        if group_sums[g] < min_g:
            min_g = group_sums[g]
        if group_sums[g] > max_g:
            max_g = group_sums[g]
    if max_g - min_g > TARGET_EPS:
        balance_phases_by_reciprocal(genoma, group_sums, config)

    # 6. Compute exact drift-free fitness from scratch
    var metrics = evaluate_exact(Span(genoma), caps_span, config)
    var solve_ns = perf_counter_ns() - t0_ns
    return BalanceResult(
        genoma=genoma^,
        fitness=metrics[0],
        par_std=metrics[1],
        phase_std=metrics[2],
        solve_ns=solve_ns,
        moves=moves,
    )


def save_results_csv(
    out_path: String,
    data: CapacitorData,
    genoma: List[Int32],
    config: BankConfig,
) raises:
    """Write the ordered 3-phase (A, B, C) capacitor bank solution to CSV."""
    var phase_names: Array[StaticString, NUM_PHASES] = ["A", "B", "C"]
    var lines = String(
        "position,phase,series_group,parallel_slot,db_idx,serial,capacitance,deviation\n"
    )
    for pos in range(config.genoma_length):
        var c = Int(genoma[pos])
        var group = pos // config.caps_per_group
        var phase = group // config.groups_per_phase
        var series_group = group % config.groups_per_phase
        var slot = pos % config.caps_per_group
        lines += (
            String(pos)
            + ","
            + String(phase_names[phase])
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
    var db_path = String("src/capacitor_balance/capacitors.db")
    var out_path = String("mojo_solver/residual_results.csv")
    var series = DEFAULT_SERIES
    var parallel = DEFAULT_PARALLEL
    var bench_iters = 1

    var i = 1
    while i < len(args):
        var arg = String(args[i])
        if arg == "--series" and i + 1 < len(args):
            series = Int(String(args[i + 1]))
            i += 2
        elif arg == "--parallel" and i + 1 < len(args):
            parallel = Int(String(args[i + 1]))
            i += 2
        elif arg == "--db" and i + 1 < len(args):
            db_path = String(args[i + 1])
            i += 2
        elif arg == "--out" and i + 1 < len(args):
            out_path = String(args[i + 1])
            i += 2
        elif arg == "--bench" and i + 1 < len(args):
            bench_iters = Int(String(args[i + 1]))
            i += 2
        else:
            i += 1

    if series < 1 or parallel < 1:
        raise Error("Both --series and --parallel must be >= 1")

    var config = BankConfig(series, parallel)
    var wall_t0 = perf_counter()
    var data = load_capacitors(db_path)

    if config.genoma_length > len(data.caps):
        raise Error(
            "Requested 3 * "
            + String(series)
            + " * "
            + String(parallel)
            + " = "
            + String(config.genoma_length)
            + " capacitors, but database only has "
            + String(len(data.caps))
            + " valid rows"
        )

    var result = solve_residual_balance(data.caps, config)
    if bench_iters > 1:
        var best_ns = result.solve_ns
        var sum_ns = result.solve_ns
        for _ in range(1, bench_iters):
            var r = solve_residual_balance(data.caps, config)
            sum_ns += r.solve_ns
            if r.solve_ns < best_ns:
                best_ns = r.solve_ns
            result = r^
        print(
            "bench_iters="
            + String(bench_iters)
            + " mean_us="
            + String(Float64(sum_ns) / Float64(bench_iters) / 1000.0)
            + " min_us="
            + String(Float64(best_ns) / 1000.0)
        )

    save_results_csv(out_path, data, result.genoma, config)
    var wall_elapsed = perf_counter() - wall_t0

    print(
        "mode=residual"
        + " bank=3x"
        + String(series)
        + "x"
        + String(parallel)
        + "("
        + String(config.genoma_length)
        + ")"
        + " best_fit="
        + String(result.fitness)
        + " par_std="
        + String(result.par_std)
        + " phase_std="
        + String(result.phase_std)
        + " solve_us="
        + String(Float64(result.solve_ns) / 1000.0)
        + " wall_ms="
        + String(wall_elapsed * 1000.0)
        + " moves="
        + String(result.moves)
    )
