"""Alternative ultra-fast Mojo solver for 3-phase (A, B, C) Capacitor Balance.

Algorithm: Tightest-Window Snake/LPT Construction + Targeted Residual Balancing
+ Phase-Reciprocal Assignment (zero-allocation, sub-millisecond).

Instead of stochastic population evolution (GA) or exhaustive O(N*M + N^2)
neighborhood scans, this solver exploits the additive/reciprocal structure of
a 3-phase series-parallel capacitor bank directly:

1. **Tightest-Window Selection**: Sorts the valid capacitors once and picks the
   contiguous 162-unit window with minimum spread (densest cluster around the
   nominal/modal capacitance).
2. **Snake + LPT Packing**: Assigns the first 8 parallel layers (144 units) in
   alternating boustrophedon order (pairing high & low in O(1)) and packs the
   final 9th layer inversely against the sorted partial sums.
3. **O(1) Max-Min Group Cancellation & O(log M) Binary-Search Pool Matching**:
   Repeatedly picks the heaviest (g_max) and lightest (g_min) parallel groups
   and applies either:
   - Exact algebraic 2-opt transfer between g_max and g_min (at most 81 pairs,
     minimizing |delta - (P_max - P_min) / 2| with zero sqrt calls), or
   - Binary-searched pool replacement on the sorted unused pool to pull g_max
     or g_min directly onto the target group capacitance P_target.
4. **Phase A/B/C Reciprocal Balancing**: Assigns the 18 parallel groups to
   phases A, B, and C (6 groups per phase) to equalize reciprocal sums
   sum(1 / P_g), minimizing the 20x-weighted phase unbalance even on arbitrary
   continuous capacitance datasets.
"""

from std.ffi import OwnedDLHandle, c_int
from std.math import abs, sqrt
from std.sys import CompilationTarget, argv
from std.time import perf_counter, perf_counter_ns

comptime NUM_PHASES = 3
comptime GROUPS_PER_PHASE = 6
comptime CAPS_PER_GROUP = 9
comptime NUM_GROUPS = NUM_PHASES * GROUPS_PER_PHASE  # 18
comptime GENOMA_LENGTH = NUM_GROUPS * CAPS_PER_GROUP  # 162
comptime TARGET_EPS = 1e-15


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
) -> Tuple[Float64, Float64, Float64]:
    """Compute exact (fitness, par_std, phase_std) using two-pass SIMD-4."""
    var group_sums = Array[Float64, NUM_GROUPS](fill=0.0)
    comptime for g in range(NUM_GROUPS):
        var base = g * CAPS_PER_GROUP
        group_sums[g] = (
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
    var d16 = group_sums[16] - par_mean
    var d17 = group_sums[17] - par_mean
    var par_var = (sq_vec.reduce_add() + d16 * d16 + d17 * d17) / Float64(
        NUM_GROUPS - 1
    )
    var par_std = sqrt(par_var) if par_var > 0.0 else 0.0

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
    # Bubble right if new_val is larger than next element
    while i + 1 < n and caps[Int(unused[i + 1])] < new_val:
        unused[i] = unused[i + 1]
        i += 1
    # Bubble left if new_val is smaller than previous element
    while i > 0 and caps[Int(unused[i - 1])] > new_val:
        unused[i] = unused[i - 1]
        i -= 1
    unused[i] = new_cap_idx


def balance_phases_by_reciprocal(
    mut genoma: List[Int32],
    mut group_sums: Array[Float64, NUM_GROUPS],
):
    """Assign the 18 parallel groups across Phases A, B, C to equalize 1/S_p.

    When parallel group sums have any non-zero residual differences (e.g. on
    arbitrary continuous measurements), arranging the 18 groups into 3 phases
    of 6 groups using LPT + 2-opt on inv_p = 1 / P_g minimizes the 20x-weighted
    phase unbalance without altering any parallel group sum.
    """
    # Compute 1 / P_g for each group and sort group indices descending by 1 / P_g
    var inv_g = Array[Float64, NUM_GROUPS](fill=0.0)
    var g_order = List[Int](capacity=NUM_GROUPS)
    for g in range(NUM_GROUPS):
        inv_g[g] = 1.0 / group_sums[g]
        g_order.append(g)

    def cmp_inv_desc(a: Int, b: Int) {imm inv_g} -> Bool:
        return inv_g[a] > inv_g[b]

    sort(g_order, cmp_inv_desc)

    # Greedy LPT assignment of 18 groups into 3 phases (6 per phase)
    var phase_groups = Array[Int, NUM_GROUPS](fill=0)
    var phase_counts = Array[Int, NUM_PHASES](fill=0)
    var phase_inv = Array[Float64, NUM_PHASES](fill=0.0)

    for idx in range(NUM_GROUPS):
        var g = g_order[idx]
        var best_p = -1
        var min_inv = 1e30
        for p in range(NUM_PHASES):
            if phase_counts[p] < GROUPS_PER_PHASE and phase_inv[p] < min_inv:
                min_inv = phase_inv[p]
                best_p = p
        phase_groups[best_p * GROUPS_PER_PHASE + phase_counts[best_p]] = g
        phase_counts[best_p] += 1
        phase_inv[best_p] += inv_g[g]

    # Exact 2-opt swaps between phases to further reduce reciprocal spread
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
        for s1 in range(GROUPS_PER_PHASE):
            var g1 = phase_groups[p_max * GROUPS_PER_PHASE + s1]
            for s2 in range(GROUPS_PER_PHASE):
                var g2 = phase_groups[p_min * GROUPS_PER_PHASE + s2]
                var delta = inv_g[g1] - inv_g[g2]
                if delta > 0.0 and delta < gap:
                    var err = abs(delta - ideal)
                    if err < best_err:
                        best_err = err
                        best_s1 = s1
                        best_s2 = s2
        if best_s1 >= 0:
            var idx1 = p_max * GROUPS_PER_PHASE + best_s1
            var idx2 = p_min * GROUPS_PER_PHASE + best_s2
            var g1 = phase_groups[idx1]
            var g2 = phase_groups[idx2]
            var delta = inv_g[g1] - inv_g[g2]
            phase_groups[idx1] = g2
            phase_groups[idx2] = g1
            phase_inv[p_max] -= delta
            phase_inv[p_min] += delta
            improved = True

    # Reorder genoma and group_sums according to phase_groups
    var reordered = List[Int32](length=GENOMA_LENGTH, fill=Int32(0))
    var new_sums = Array[Float64, NUM_GROUPS](fill=0.0)
    for new_g in range(NUM_GROUPS):
        var old_g = phase_groups[new_g]
        new_sums[new_g] = group_sums[old_g]
        var src = old_g * CAPS_PER_GROUP
        var dst = new_g * CAPS_PER_GROUP
        for k in range(CAPS_PER_GROUP):
            reordered[dst + k] = genoma[src + k]
    genoma = reordered^
    group_sums = new_sums^


def solve_residual_balance(caps: List[Float64]) -> BalanceResult:
    """Solve 3-phase capacitor balance via targeted residual matching."""
    var t0_ns = perf_counter_ns()
    var num_caps = len(caps)
    var caps_span = Span(caps)

    # 1. Sort all valid capacitor indices by ascending capacitance: O(N log N)
    var sorted_all = List[Int](capacity=num_caps)
    for i in range(num_caps):
        sorted_all.append(i)

    def cmp_asc(a: Int, b: Int) {imm caps} -> Bool:
        return caps[a] < caps[b]

    sort(sorted_all, cmp_asc)

    # 2. Pick the contiguous 162-unit window with minimum capacitance spread
    var best_start = 0
    var best_range = 1e30
    var max_start = num_caps - GENOMA_LENGTH
    for s in range(max_start + 1):
        var r = caps[sorted_all[s + GENOMA_LENGTH - 1]] - caps[sorted_all[s]]
        if r < best_range:
            best_range = r
            best_start = s

    # Partition into sorted chosen[162] and sorted unused[N - 162]
    var chosen = List[Int32](capacity=GENOMA_LENGTH)
    var unused = List[Int32](capacity=num_caps - GENOMA_LENGTH)
    for i in range(best_start):
        unused.append(Int32(sorted_all[i]))
    for i in range(best_start, best_start + GENOMA_LENGTH):
        chosen.append(Int32(sorted_all[i]))
    for i in range(best_start + GENOMA_LENGTH, num_caps):
        unused.append(Int32(sorted_all[i]))

    # 3. Snake (boustrophedon) assignment for first 8 layers + LPT for 9th layer
    var genoma = List[Int32](length=GENOMA_LENGTH, fill=Int32(0))
    var group_sums = Array[Float64, NUM_GROUPS](fill=0.0)

    for layer in range(CAPS_PER_GROUP - 1):
        var base_idx = layer * NUM_GROUPS
        if (layer & 1) == 0:
            for g in range(NUM_GROUPS):
                var c = chosen[base_idx + g]
                genoma[g * CAPS_PER_GROUP + layer] = c
                group_sums[g] += caps[Int(c)]
        else:
            for g in range(NUM_GROUPS):
                var c = chosen[base_idx + (NUM_GROUPS - 1 - g)]
                genoma[g * CAPS_PER_GROUP + layer] = c
                group_sums[g] += caps[Int(c)]

    # Final 9th layer: sort groups by ascending partial sum, assign largest caps
    var g_order = List[Int](capacity=NUM_GROUPS)
    for g in range(NUM_GROUPS):
        g_order.append(g)

    def cmp_gsum(a: Int, b: Int) {imm group_sums} -> Bool:
        return group_sums[a] < group_sums[b]

    sort(g_order, cmp_gsum)
    var last_base = (CAPS_PER_GROUP - 1) * NUM_GROUPS
    for rank in range(NUM_GROUPS):
        var g = g_order[rank]
        var c = chosen[last_base + (NUM_GROUPS - 1 - rank)]
        genoma[g * CAPS_PER_GROUP + (CAPS_PER_GROUP - 1)] = c
        group_sums[g] += caps[Int(c)]

    # 4. Targeted Residual Balancing Loop (Max-Min Cancel + Binary-Search Pool)
    var moves = 0
    var num_unused = len(unused)
    var max_moves = 2000

    while moves < max_moves:
        # Find heaviest (g_max) and lightest (g_min) groups & total sum
        var g_max = 0
        var g_min = 0
        var total_sum = group_sums[0]
        for g in range(1, NUM_GROUPS):
            var s = group_sums[g]
            total_sum += s
            if s > group_sums[g_max]:
                g_max = g
            if s < group_sums[g_min]:
                g_min = g

        var gap = group_sums[g_max] - group_sums[g_min]
        if gap <= TARGET_EPS:
            break

        var target = total_sum / Float64(NUM_GROUPS)

        # 4a. Try O(1) max-min 2-opt cancellation (at most 81 pairs, zero sqrt!)
        # Swapping a in g_max with b in g_min changes SSE by 2*d*(d - gap).
        var best_k1 = -1
        var best_k2 = -1
        var best_delta_sse = -1e-18
        var base_max = g_max * CAPS_PER_GROUP
        var base_min = g_min * CAPS_PER_GROUP

        for k1 in range(CAPS_PER_GROUP):
            var c_a = caps[Int(genoma[base_max + k1])]
            for k2 in range(CAPS_PER_GROUP):
                var d = c_a - caps[Int(genoma[base_min + k2])]
                if d > 0.0 and d < gap:
                    var d_sse = 2.0 * d * (d - gap)
                    if d_sse < best_delta_sse:
                        best_delta_sse = d_sse
                        best_k1 = k1
                        best_k2 = k2

        # 4b. Try O(log M) binary-searched pool replacement on g_max and g_min
        # Replacing c_old in group g by c_new changes sum_g by d = c_new - c_old.
        # Exact change in SSE = sum((P_i - mean)^2):
        #   d_sse = d * (2 * (P_g - target) + d * (1 - 1/NUM_GROUPS))
        var best_pool_g = -1
        var best_pool_k = -1
        var best_pool_u = -1
        var best_pool_sse = best_delta_sse
        comptime SSE_FACTOR = 1.0 - 1.0 / Float64(NUM_GROUPS)

        if num_unused > 0:
            # Check both g_max (needs smaller cap) and g_min (needs larger cap)
            var cand_groups: Array[Int, 2] = [g_max, g_min]
            for cg_i in range(2):
                var g = cand_groups[cg_i]
                var resid = group_sums[g] - target
                var base_g = g * CAPS_PER_GROUP
                # Ideal d to zero out residual: d_ideal = -resid / SSE_FACTOR
                var d_ideal = -resid / SSE_FACTOR
                for k in range(CAPS_PER_GROUP):
                    var c_old = caps[Int(genoma[base_g + k])]
                    var want = c_old + d_ideal
                    var lb = lower_bound_unused(unused, caps_span, want)
                    # Inspect lb and lb - 1 (nearest two values in sorted pool)
                    for probe in range(lb - 1, lb + 1):
                        if probe >= 0 and probe < num_unused:
                            var c_new = caps[Int(unused[probe])]
                            var d = c_new - c_old
                            if d != 0.0:
                                var d_sse = d * (2.0 * resid + d * SSE_FACTOR)
                                if d_sse < best_pool_sse:
                                    best_pool_sse = d_sse
                                    best_pool_g = g
                                    best_pool_k = k
                                    best_pool_u = probe

        # Commit whichever move achieved the greatest SSE reduction
        if best_pool_g >= 0:
            var slot = best_pool_g * CAPS_PER_GROUP + best_pool_k
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
        else:
            # 4c. Fallback: try pulling any non-target group onto the modal
            # group sum via binary-searched pool replacement (handles integer
            # residual quantization when mean lies between discrete sums)
            var modal_target = group_sums[g_max]
            if abs(group_sums[g_min] - target) < abs(
                group_sums[g_max] - target
            ):
                modal_target = group_sums[g_min]

            var found_fallback = False
            for g in range(NUM_GROUPS):
                var need = modal_target - group_sums[g]
                if abs(need) <= TARGET_EPS:
                    continue
                var base_g = g * CAPS_PER_GROUP
                for k in range(CAPS_PER_GROUP):
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

    # 5. If any group spread remains, arrange the 18 groups across Phases A/B/C
    # to minimize the 20x-weighted series phase unbalance
    var min_g = group_sums[0]
    var max_g = group_sums[0]
    for g in range(1, NUM_GROUPS):
        if group_sums[g] < min_g:
            min_g = group_sums[g]
        if group_sums[g] > max_g:
            max_g = group_sums[g]
    if max_g - min_g > TARGET_EPS:
        balance_phases_by_reciprocal(genoma, group_sums)

    # 6. Compute exact drift-free fitness from scratch
    var metrics = evaluate_exact(Span(genoma), caps_span)
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
) raises:
    """Write the ordered 162-capacitor 3-phase (A, B, C) solution to CSV."""
    var phase_names: Array[StaticString, NUM_PHASES] = ["A", "B", "C"]
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
    var bench_iters = 1

    var i = 1
    while i < len(args):
        var arg = String(args[i])
        if arg == "--db" and i + 1 < len(args):
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

    var wall_t0 = perf_counter()
    var data = load_capacitors(db_path)

    var result = solve_residual_balance(data.caps)
    if bench_iters > 1:
        var best_ns = result.solve_ns
        var sum_ns = result.solve_ns
        for _ in range(1, bench_iters):
            var r = solve_residual_balance(data.caps)
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

    save_results_csv(out_path, data, result.genoma)
    var wall_elapsed = perf_counter() - wall_t0

    print(
        "mode=residual"
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
