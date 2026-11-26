"""Shared core Mojo algorithms for 3-phase (A, B, C) Capacitor Bank Balancing.

Contains no `main()` so it can be imported by both CLI binaries (`solver.mojo`,
`residual_solver.mojo`) and the Python C-extension shared library
(`capacitor_mojo.mojo` -> `capacitor_mojo.so`).
"""

from std.math import abs, max, sqrt
from std.time import perf_counter, perf_counter_ns

comptime NUM_PHASES = 3
comptime DEFAULT_SERIES = 6
comptime DEFAULT_PARALLEL = 9
comptime DEFAULT_GENERATION_LENGTH = 1000
comptime DEFAULT_ITERATIONS = 500
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
struct SolveResult(Movable):
    """Result returned by GA and 2-opt local-search solvers."""

    var best_genoma: List[Int32]
    var best_fitness: Float64
    var solve_seconds: Float64
    var iterations_run: Int


@fieldwise_init
struct BalanceResult(Movable):
    """Result returned by the algebraic residual-balancing solver."""

    var genoma: List[Int32]
    var fitness: Float64
    var par_std: Float64
    var phase_std: Float64
    var solve_ns: Int
    var moves: Int


@inline(.always)
def fitness_from_group_sums(
    group_sums: Span[Float64, _],
    config: BankConfig,
) -> Float64:
    """Compute exact ddof=1 fitness from parallel group sums using SIMD-4."""
    var num_groups = config.num_groups
    var groups_per_phase = config.groups_per_phase

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

    return 20.0 * phase_std + par_std


@inline(.always)
def compute_fitness(
    genoma: Span[Int32, _],
    caps: Span[Float64, _],
    mut group_sums: List[Float64],
    config: BankConfig,
) -> Float64:
    """Evaluate a genoma directly using a pre-allocated scratch buffer."""
    var caps_per_group = config.caps_per_group
    for g in range(config.num_groups):
        var base = g * caps_per_group
        var s = 0.0
        for k in range(caps_per_group):
            s += caps[Int(genoma[base + k])]
        group_sums[g] = s
    return fitness_from_group_sums(Span(group_sums), config)


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


def _fill_random_genoma(
    mut dst: List[Int32],
    dst_offset: Int,
    mut pool: List[Int32],
    genoma_length: Int,
    mut rng: Rng,
):
    var n = len(pool)
    for i in range(genoma_length):
        var j = i + rng.next_int(n - i)
        var tmp = pool[i]
        pool[i] = pool[j]
        pool[j] = tmp
        dst[dst_offset + i] = pool[i]


def run_ga(
    caps: List[Float64],
    config: BankConfig,
    generation_length: Int,
    iterations: Int,
    seed: UInt64,
    verbose: Bool = False,
) -> SolveResult:
    """Native SIMD Genetic Algorithm solver."""
    var t0 = perf_counter()
    var num_caps = len(caps)
    var genoma_len = config.genoma_length
    var best_len = max(1, (generation_length * 10) // 100)
    var mut_len = (generation_length * 20) // 100
    var shuff_len = (generation_length * 30) // 100
    var cross_len = (generation_length * 10) // 100
    var rand_len = generation_length - (
        best_len + mut_len + shuff_len + cross_len
    )

    var rng = Rng(seed)
    var pop_a = List[Int32](
        length=generation_length * genoma_len, fill=Int32(0)
    )
    var pop_b = List[Int32](
        length=generation_length * genoma_len, fill=Int32(0)
    )
    var pool = List[Int32](capacity=num_caps)
    for i in range(num_caps):
        pool.append(Int32(i))

    for g in range(generation_length):
        _fill_random_genoma(pop_a, g * genoma_len, pool, genoma_len, rng)

    var scores = List[Float64](length=generation_length, fill=0.0)
    var order = List[Int](length=generation_length, fill=0)
    var used_stamp = List[Int32](length=num_caps, fill=Int32(0))
    var avail_buf = List[Int32](capacity=num_caps)
    var group_sums = List[Float64](length=config.num_groups, fill=0.0)
    var stamp: Int32 = 0
    var caps_span = Span(caps)

    for it in range(iterations):
        var pop_span = Span(pop_a)
        for g in range(generation_length):
            var off = g * genoma_len
            scores[g] = compute_fitness(
                pop_span[off : off + genoma_len], caps_span, group_sums, config
            )
            order[g] = g

        def cmp_idx(a: Int, b: Int) {imm scores} -> Bool:
            return scores[a] < scores[b]

        sort(order, cmp_idx)
        if verbose:
            print(it, "best fit ->", scores[order[0]])

        var out_idx = 0

        for k in range(best_len):
            var src_off = order[k] * genoma_len
            var dst_off = out_idx * genoma_len
            for i in range(genoma_len):
                pop_b[dst_off + i] = pop_a[src_off + i]
            out_idx += 1

        for _ in range(mut_len):
            var parent = order[rng.next_int(best_len)]
            var src_off = parent * genoma_len
            var dst_off = out_idx * genoma_len
            stamp += 1
            for i in range(genoma_len):
                var c = pop_a[src_off + i]
                pop_b[dst_off + i] = c
                used_stamp[Int(c)] = stamp
            avail_buf.clear()
            for c_idx in range(num_caps):
                if used_stamp[c_idx] != stamp:
                    avail_buf.append(Int32(c_idx))
            if len(avail_buf) > 0:
                var pos = rng.next_int(genoma_len)
                pop_b[dst_off + pos] = avail_buf[rng.next_int(len(avail_buf))]
            out_idx += 1

        for _ in range(shuff_len):
            var parent = order[rng.next_int(best_len)]
            var src_off = parent * genoma_len
            var dst_off = out_idx * genoma_len
            for i in range(genoma_len):
                pop_b[dst_off + i] = pop_a[src_off + i]
            for i in range(genoma_len - 1, 0, -1):
                var j = rng.next_int(i + 1)
                var tmp = pop_b[dst_off + i]
                pop_b[dst_off + i] = pop_b[dst_off + j]
                pop_b[dst_off + j] = tmp
            out_idx += 1

        for _ in range(rand_len):
            _fill_random_genoma(
                pop_b, out_idx * genoma_len, pool, genoma_len, rng
            )
            out_idx += 1

        for _ in range(cross_len):
            var p1 = order[rng.next_int(best_len)]
            var p2 = rng.next_int(generation_length)
            var off1 = p1 * genoma_len
            var off2 = p2 * genoma_len
            var dst_off = out_idx * genoma_len
            stamp += 1
            for i in range(genoma_len):
                used_stamp[Int(pop_a[off1 + i])] = stamp
            for i in range(genoma_len):
                var g1 = pop_a[off1 + i]
                var g2 = pop_a[off2 + i]
                if used_stamp[Int(g2)] != stamp and rng.next_bit():
                    pop_b[dst_off + i] = g2
                else:
                    pop_b[dst_off + i] = g1
            out_idx += 1

        swap(pop_a, pop_b)

    var final_span = Span(pop_a)
    var best_idx = 0
    var best_fit = 1e30
    for g in range(generation_length):
        var off = g * genoma_len
        var f = compute_fitness(
            final_span[off : off + genoma_len], caps_span, group_sums, config
        )
        if f < best_fit:
            best_fit = f
            best_idx = g

    var best_genoma = List[Int32](capacity=genoma_len)
    var best_off = best_idx * genoma_len
    for i in range(genoma_len):
        best_genoma.append(pop_a[best_off + i])

    return SolveResult(
        best_genoma=best_genoma^,
        best_fitness=best_fit,
        solve_seconds=perf_counter() - t0,
        iterations_run=iterations,
    )


def _greedy_balanced_init(
    caps: List[Float64],
    config: BankConfig,
    mut selected: List[Int32],
    mut unused: List[Int32],
    mut group_sums: List[Float64],
    mut rng: Rng,
    randomized: Bool,
):
    var num_caps = len(caps)
    var genoma_len = config.genoma_length
    var num_groups = config.num_groups
    var caps_per_group = config.caps_per_group

    var order = List[Int](capacity=num_caps)
    for i in range(num_caps):
        order.append(i)

    if randomized:
        for i in range(genoma_len):
            var j = i + rng.next_int(num_caps - i)
            var tmp = order[i]
            order[i] = order[j]
            order[j] = tmp
    else:

        def cmp_cap(a: Int, b: Int) {imm caps} -> Bool:
            return caps[a] < caps[b]

        sort(order, cmp_cap)
        var start = (num_caps - genoma_len) // 2
        var chosen = List[Int](capacity=num_caps)
        for i in range(genoma_len):
            chosen.append(order[start + i])
        for i in range(start):
            chosen.append(order[i])
        for i in range(start + genoma_len, num_caps):
            chosen.append(order[i])
        order = chosen^

    var chosen_set = List[Int](capacity=genoma_len)
    for i in range(genoma_len):
        chosen_set.append(order[i])

    def cmp_desc(a: Int, b: Int) {imm caps} -> Bool:
        return caps[a] > caps[b]

    sort(chosen_set, cmp_desc)

    unused.clear()
    for i in range(genoma_len, num_caps):
        unused.append(Int32(order[i]))

    var counts = List[Int](length=num_groups, fill=0)
    for g in range(num_groups):
        group_sums[g] = 0.0

    for idx_pos in range(genoma_len):
        var c_idx = chosen_set[idx_pos]
        var best_g = -1
        var min_sum = 1e30
        for g in range(num_groups):
            if counts[g] < caps_per_group and group_sums[g] < min_sum:
                min_sum = group_sums[g]
                best_g = g
        var slot = best_g * caps_per_group + counts[best_g]
        selected[slot] = Int32(c_idx)
        counts[best_g] += 1
        group_sums[best_g] += caps[c_idx]


def run_fast_solver(
    caps: List[Float64],
    config: BankConfig,
    max_restarts: Int = 64,
    seed: UInt64 = 42,
    verbose: Bool = False,
) -> SolveResult:
    """Greedy LPT + incremental 2-opt inter-group swap & pool replacement."""
    var t0 = perf_counter()
    var num_caps = len(caps)
    var genoma_len = config.genoma_length
    var num_groups = config.num_groups
    var caps_per_group = config.caps_per_group
    var num_unused = num_caps - genoma_len
    var rng = Rng(seed)

    var selected = List[Int32](length=genoma_len, fill=Int32(0))
    var unused = List[Int32](capacity=num_unused)
    var group_sums = List[Float64](length=num_groups, fill=0.0)
    var verify_sums = List[Float64](length=num_groups, fill=0.0)

    var global_best_genoma = List[Int32](length=genoma_len, fill=Int32(0))
    var global_best_fit = 1e30
    var sweeps_total = 0

    for restart in range(max_restarts):
        _greedy_balanced_init(
            caps,
            config,
            selected,
            unused,
            group_sums,
            rng,
            randomized=(restart > 0),
        )
        var cur_fit = fitness_from_group_sums(Span(group_sums), config)

        var improved = True
        while improved and cur_fit > TARGET_EPS:
            improved = False
            sweeps_total += 1

            if num_unused > 0:
                for g in range(num_groups):
                    var base_g = g * caps_per_group
                    for k in range(caps_per_group):
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
                            var cand_fit = fitness_from_group_sums(
                                Span(group_sums), config
                            )
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

            for g1 in range(num_groups - 1):
                var base1 = g1 * caps_per_group
                for k1 in range(caps_per_group):
                    var pos_a = base1 + k1
                    var c_a = Int(selected[pos_a])
                    var cap_a = caps[c_a]
                    var old_sum1 = group_sums[g1]
                    var best_pos_b = -1
                    var best_g2 = -1
                    var best_swap_fit = cur_fit - 1e-18
                    for g2 in range(g1 + 1, num_groups):
                        var base2 = g2 * caps_per_group
                        var old_sum2 = group_sums[g2]
                        for k2 in range(caps_per_group):
                            var pos_b = base2 + k2
                            var c_b = Int(selected[pos_b])
                            var delta = caps[c_b] - cap_a
                            if delta == 0.0:
                                continue
                            group_sums[g1] = old_sum1 + delta
                            group_sums[g2] = old_sum2 - delta
                            var cand_fit = fitness_from_group_sums(
                                Span(group_sums), config
                            )
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

        var exact_fit = compute_fitness(
            Span(selected), Span(caps), verify_sums, config
        )
        if exact_fit < global_best_fit:
            global_best_fit = exact_fit
            for i in range(genoma_len):
                global_best_genoma[i] = selected[i]
            if verbose:
                print("restart", restart, "best fit ->", global_best_fit)
        if global_best_fit <= TARGET_EPS:
            break

    return SolveResult(
        best_genoma=global_best_genoma^,
        best_fitness=global_best_fit,
        solve_seconds=perf_counter() - t0,
        iterations_run=sweeps_total,
    )


@inline(.always)
def _lower_bound_unused(
    unused: List[Int32], caps: Span[Float64, _], target: Float64
) -> Int:
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
def _replace_in_sorted_unused(
    mut unused: List[Int32],
    caps: Span[Float64, _],
    pos: Int,
    new_cap_idx: Int32,
):
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


def _balance_phases_by_reciprocal(
    mut genoma: List[Int32],
    mut group_sums: List[Float64],
    config: BankConfig,
):
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
    """Tightest-window snake/LPT + algebraic residual matching + phase balance.
    """
    var t0_ns = perf_counter_ns()
    var num_caps = len(caps)
    var num_groups = config.num_groups
    var caps_per_group = config.caps_per_group
    var genoma_len = config.genoma_length
    var caps_span = Span(caps)

    var sorted_all = List[Int](capacity=num_caps)
    for i in range(num_caps):
        sorted_all.append(i)

    def cmp_asc(a: Int, b: Int) {imm caps} -> Bool:
        return caps[a] < caps[b]

    sort(sorted_all, cmp_asc)

    var best_start = 0
    var best_range = 1e30
    var max_start = num_caps - genoma_len
    for s in range(max_start + 1):
        var r = caps[sorted_all[s + genoma_len - 1]] - caps[sorted_all[s]]
        if r < best_range:
            best_range = r
            best_start = s

    var chosen = List[Int32](capacity=genoma_len)
    var unused = List[Int32](capacity=num_caps - genoma_len)
    for i in range(best_start):
        unused.append(Int32(sorted_all[i]))
    for i in range(best_start, best_start + genoma_len):
        chosen.append(Int32(sorted_all[i]))
    for i in range(best_start + genoma_len, num_caps):
        unused.append(Int32(sorted_all[i]))

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
                    var lb = _lower_bound_unused(unused, caps_span, want)
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
            _replace_in_sorted_unused(unused, caps_span, best_pool_u, old_idx)
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
                    var lb = _lower_bound_unused(unused, caps_span, want)
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
                                _replace_in_sorted_unused(
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

    var min_g = group_sums[0]
    var max_g = group_sums[0]
    for g in range(1, num_groups):
        if group_sums[g] < min_g:
            min_g = group_sums[g]
        if group_sums[g] > max_g:
            max_g = group_sums[g]
    if max_g - min_g > TARGET_EPS:
        _balance_phases_by_reciprocal(genoma, group_sums, config)

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
