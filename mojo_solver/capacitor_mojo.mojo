"""Python C-extension shared library (.so) exposing Mojo capacitor solvers.

Build with:
    uv run mojo build -I mojo_solver --emit shared-lib -O 3 --Werror \
        -o mojo_solver/capacitor_mojo.so mojo_solver/capacitor_mojo.mojo
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder

from core import (
    BankConfig,
    run_fast_solver,
    run_ga,
    solve_residual_balance,
)


def _py_to_caps(caps_obj: PythonObject) raises -> List[Float64]:
    var n = len(caps_obj)
    var caps = List[Float64](capacity=n)
    for item in caps_obj:
        caps.append(Float64(py=item))
    return caps^


def _indices_to_py(genoma: List[Int32]) raises -> PythonObject:
    var py_list = Python.list()
    for idx in genoma:
        py_list.append(Int(idx))
    return py_list


def _validate_inputs(
    num_caps: Int, series: Int, parallel: Int
) raises -> BankConfig:
    if series < 1 or parallel < 1:
        raise Error("Both series and parallel must be >= 1")
    var config = BankConfig(series, parallel)
    if config.genoma_length > num_caps:
        raise Error(
            "Requested 3 * "
            + String(series)
            + " * "
            + String(parallel)
            + " = "
            + String(config.genoma_length)
            + " capacitors, but only "
            + String(num_caps)
            + " provided"
        )
    return config


def py_solve_residual(
    caps_obj: PythonObject,
    series_obj: PythonObject,
    parallel_obj: PythonObject,
) raises -> PythonObject:
    """Run the algebraic residual-balancing solver in-process."""
    var caps = _py_to_caps(caps_obj)
    var config = _validate_inputs(
        len(caps), Int(py=series_obj), Int(py=parallel_obj)
    )
    var res = solve_residual_balance(caps, config)
    return Python.tuple(
        _indices_to_py(res.genoma),
        PythonObject(res.fitness),
        PythonObject(Float64(res.solve_ns) / 1_000_000.0),
    )


def py_solve_fast(
    caps_obj: PythonObject,
    series_obj: PythonObject,
    parallel_obj: PythonObject,
    seed_obj: PythonObject,
) raises -> PythonObject:
    """Run the greedy LPT + 2-opt local-search solver in-process."""
    var caps = _py_to_caps(caps_obj)
    var config = _validate_inputs(
        len(caps), Int(py=series_obj), Int(py=parallel_obj)
    )
    var seed = UInt64(Int(py=seed_obj))
    var res = run_fast_solver(caps, config, 64, seed, False)
    return Python.tuple(
        _indices_to_py(res.best_genoma),
        PythonObject(res.best_fitness),
        PythonObject(res.solve_seconds * 1000.0),
    )


def py_solve_ga(
    caps_obj: PythonObject,
    series_obj: PythonObject,
    parallel_obj: PythonObject,
    gen_len_obj: PythonObject,
    iterations_obj: PythonObject,
    seed_obj: PythonObject,
) raises -> PythonObject:
    """Run the native SIMD genetic algorithm solver in-process."""
    var caps = _py_to_caps(caps_obj)
    var config = _validate_inputs(
        len(caps), Int(py=series_obj), Int(py=parallel_obj)
    )
    var gen_len = Int(py=gen_len_obj)
    var iterations = Int(py=iterations_obj)
    var seed = UInt64(Int(py=seed_obj))
    var res = run_ga(caps, config, gen_len, iterations, seed, False)
    return Python.tuple(
        _indices_to_py(res.best_genoma),
        PythonObject(res.best_fitness),
        PythonObject(res.solve_seconds * 1000.0),
    )


@export
def PyInit_capacitor_mojo() abi("C") -> PythonObject:
    try:
        var mb = PythonModuleBuilder("capacitor_mojo")
        mb.def_function[py_solve_residual]("solve_residual")
        mb.def_function[py_solve_fast]("solve_fast")
        mb.def_function[py_solve_ga]("solve_ga")
        return mb.finalize()
    except e:
        abort(String("failed to initialize capacitor_mojo module: ", e))
