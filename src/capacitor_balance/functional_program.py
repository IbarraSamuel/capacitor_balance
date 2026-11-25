"""Async solver for Capacitor balance."""

import asyncio
import operator
import random
from collections.abc import (
    AsyncGenerator,
    Iterable,
    Sequence,
)
from decimal import Decimal
from pathlib import Path
from sqlite3 import connect
from typing import TypedDict, cast, overload

import polars as pl


class Capacitor(TypedDict):
    """Represents a capacitor."""

    idx: int
    serial: str
    cap: float


type Genoma = list[Capacitor]
type Generation = list[Genoma]

# GENERIC TYPES


@overload
def _aiter[T](iterable: Iterable[T]) -> AsyncGenerator[T]: ...
@overload
def _aiter[T](iterable: Sequence[T], _slice: int) -> AsyncGenerator[Sequence[T]]: ...
async def _aiter[T](  # ruff: ignore[unused-async]
    iterable: Iterable[T] | Sequence[T], _slice: int | None = None
) -> AsyncGenerator[T | Sequence[T]]:
    """
    Aiter creater an asyncronous generator to use in async for and async comprehensions.

    Args:
        iterable (list): list to transform into async generator
        _slice (int, optional): number of items yo hold in a single iteration of the
        iterable.

    Yields:
        Iterator[list]: values inside the iterable input or parts that the input is
        splitted
    """
    if _slice is not None and isinstance(iterable, Sequence):
        assert _slice >= 1, "Slice should be more than zero"  # ruff: ignore[assert]
        assert len(iterable) % _slice == 0, "Iterable size should be divisible by slice"  # ruff: ignore[assert]
        assert _slice <= len(iterable), "Slice should not be more than len of iterable"  # ruff: ignore[assert]
        parts = int(len(iterable) / _slice)
        for i in range(parts):
            yield iterable[_slice * i : _slice * (i + 1)]

    else:
        for v in iterable:
            yield v


def get_df(db_file: Path) -> pl.DataFrame:
    with connect(db_file) as conn:
        return pl.read_database("SELECT * FROM capacitors", conn)  # pyright: ignore[reportUnknownMemberType]


def parallel_reduction(capacitance_arr: pl.Series) -> float | Decimal:
    return capacitance_arr.sum()


def series_reduction(capacitance_arr: pl.Series) -> float | Decimal:
    return 1 / (1 / capacitance_arr).sum()


def standard_deviation(values: pl.Series) -> float | Decimal:
    return cast("float", values.std())


def map_capacitance(gen: Genoma) -> pl.Series:
    return pl.DataFrame(gen).get_column("cap")


def generation_hash(gen: Generation) -> str:
    return str(pl.DataFrame(gen).get_column("serial").str.join()[0])


def fitness(gen: Genoma) -> float:
    calc_result = (
        pl.DataFrame(gen)
        .group_by(pl.int_range(pl.len()) // 9, maintain_order=True)
        .agg(pl.sum("cap"))
        .with_columns(pl.col("cap").std().alias("par_std"))
        .group_by(
            pl.col("par_std"),
            pl.int_range(pl.len()) // 6,
            maintain_order=True,
        )
        .agg((1 / (1 / pl.col("cap")).sum()).alias("cap"))
        .with_columns((20 * pl.col("cap").std() + pl.col("par_std")).alias("std"))
    )

    result: float = float(calc_result["std"][0])
    """
    # print(calc_result)
    # parallel = [
    #     await parallel_reduction(pl.Series(caps))
    #     async for caps in aiter(tot_caps, slice=9)
    # ]
    # series = [
    #     await series_reduction(pl.Series(groups))
    #     async for groups in aiter(parallel, slice=6)
    # ]
    # phase_deviation = await standard_deviation(pl.Series(series))
    # group_deviation = await standard_deviation(pl.Series(parallel))
    # return 20 * phase_deviation + group_deviation
    """
    return result


async def sort_generation(generation: Generation) -> Generation:
    fitness_generator = [(fitness(gen), gen) async for gen in _aiter(generation)]
    generation_sorted = [
        generator[1]
        for generator in sorted(fitness_generator, key=operator.itemgetter(0))
    ]
    assert len(generation) == len(generation_sorted)  # ruff: ignore[assert]
    return generation_sorted


"""
# async def df_to_cap_list(df: pl.DataFrame) -> list[Capacitor]:
#     assert set(df.columns).issuperset(["CAPACITANCIA ", "N° DE SERIE"])
#     valid = df.filter(
#         pl.col("CAPACITANCIA ").is_null().not_()
#     )  # df.loc[df["CAPACITANCIA "].notna()]
#     print(valid)
#     return [
#         Capacitor(nm, sr, cap)
#         for nm, sr, cap in valid.select(
#             "index", "N° DE SERIE", "CAPACITANCIA "
#         ).iter_rows()
#     ]

"""


def possible_options(
    capacitor_options: list[Capacitor], current_genoma: Genoma
) -> list[Capacitor]:
    return [
        cast("Capacitor", row)
        for row in pl.DataFrame(capacitor_options)
        .join(pl.DataFrame(current_genoma), on="serial", how="anti")
        .rows(named=True)
    ]


def random_genoma(capacitor_options: list[Capacitor], genoma_length: int) -> Genoma:
    return [
        cast("Capacitor", v)
        for v in pl.DataFrame(capacitor_options)
        .sample(genoma_length, shuffle=True)
        .rows(named=True)
    ]


def random_generation(
    capacitor_options: list[Capacitor], generation_length: int, genoma_lenth: int
) -> Generation:
    return [
        random_genoma(capacitor_options, genoma_lenth) for _ in range(generation_length)
    ]


async def crossover_genoma(gen1: Genoma, gen2: Genoma) -> Genoma:
    possible = (
        pl.DataFrame(gen2)
        .join(pl.DataFrame(gen1), on="serial", how="anti")
        .rows(named=True)
    )
    return [
        g2 if g2 in possible and random.getrandbits(1) == 1 else g1
        async for g1, g2 in _aiter(zip(gen1, gen2, strict=True))
    ]


async def mutation_genoma(gen: Genoma, capacitor_options: list[Capacitor]) -> Genoma:
    index = random.randint(0, len(gen) - 1)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    return (
        [
            random.choice(capacitor_options) if n == index else prev  # ruff: ignore[suspicious-non-cryptographic-random-usage]
            async for n, prev in _aiter(enumerate(gen))
        ]
        if len(capacitor_options)
        else gen
    )


def shuffle_genoma(gen: Genoma) -> Genoma:
    comb = gen.copy()
    random.shuffle(comb)
    return comb


async def mutated_genomas(
    capacitor_options: list[Capacitor], top_genomas: list[Genoma], mut_len: int
) -> list[Genoma]:
    return [
        await mutation_genoma(gen, possible_options(capacitor_options, gen))
        async for gen in _aiter(random.choices(top_genomas, k=mut_len))  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    ]


async def shuffled_genomas(top_genomas: list[Genoma], shuff_len: int) -> list[Genoma]:
    return [
        shuffle_genoma(gen)
        async for gen in _aiter(random.choices(top_genomas, k=shuff_len))  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    ]


async def random_genomas(
    capacitor_options: list[Capacitor], genoma_len: int, rand_len: int
) -> list[Genoma]:
    return [
        random_genoma(capacitor_options, genoma_len)
        async for _ in _aiter(range(rand_len))
    ]


async def crossovered_genomas(
    top_genomas: list[Genoma], cross_len: int, past_generation: list[Genoma]
) -> list[Genoma]:
    return [
        await crossover_genoma(g1, g2)
        async for g1, g2 in _aiter(
            zip(
                random.choices(top_genomas, k=cross_len),  # ruff: ignore[suspicious-non-cryptographic-random-usage]
                random.sample(past_generation, k=cross_len),
                strict=False,
            )
        )
    ]


async def update_generation(  # ruff: ignore[too-many-arguments]
    past_generation: Generation,
    capacitor_options: list[Capacitor],
    *,
    genoma_len: int,
    top_len: int,
    rand_len: int,
    cross_len: int,
    mut_len: int,
    shuff_len: int,
) -> Generation:
    top_genomas = (await sort_generation(past_generation))[:top_len]
    best = fitness(top_genomas[0])
    print(f"best fit -> {best}")  # ruff: ignore[print]

    m = await mutated_genomas(capacitor_options, top_genomas, mut_len)
    s = await shuffled_genomas(top_genomas, shuff_len)
    r = await random_genomas(capacitor_options, genoma_len, rand_len)
    c = await crossovered_genomas(top_genomas, cross_len, past_generation)
    """
    # new_gen = await asyncio.gather(
    #     mutated_genomas(capacitor_options, top_genomas, mut_len),
    #     shuffled_genomas(top_genomas, shuff_len),
    #     random_genomas(capacitor_options, genoma_len, rand_len),
    #     crossovered_genomas(top_genomas, cross_len, past_generation)
    # )
    # return top_genomas + [
    # genoma async for result in aiter(new_gen)
    # async for genoma in aiter(result)
    # ]
    """

    return top_genomas + m + s + r + c


def get_selected_capacitors(genoma: Genoma, original_df: pl.DataFrame) -> pl.DataFrame:
    assert len({cap["serial"] for cap in genoma}) == len(genoma)  # ruff: ignore[assert]

    index_list = [cap["serial"] for cap in genoma]
    selected_df: pl.DataFrame = original_df.filter(
        pl.col("N° DE SERIE").is_in(index_list)
    )
    return selected_df


MAIN_FOLDER = Path(__file__).parent
DB_FILE = MAIN_FOLDER / "capacitors.db"
RESULT_FILE = MAIN_FOLDER / "parallel_results.xlsx"

# Genoma
GENOMA_LENGTH = 3 * 6 * 9
GENERATION_LENGTH = 1000
ITERATIONS = 500
# Generation Distribution
BEST_GENOMAS_LENGTH = int(GENERATION_LENGTH * 10 / 100)
MUTATION_LENGTH = int(GENERATION_LENGTH * 20 / 100)
SHUFFLE_LENGTH = int(GENERATION_LENGTH * 30 / 100)
CROSSOVER_LENGTH = int(GENERATION_LENGTH * 10 / 100)
RANDOM_LENGTH = int(GENERATION_LENGTH * 30 / 100)

assert (  # ruff: ignore[assert]
    GENERATION_LENGTH
    == BEST_GENOMAS_LENGTH
    + MUTATION_LENGTH
    + SHUFFLE_LENGTH
    + CROSSOVER_LENGTH
    + RANDOM_LENGTH
)


async def main() -> None:
    # Folders

    df = get_df(DB_FILE)

    # Here we are getting the count of Capacitores by deviation
    """
    # print(df)
    # df_by_deviation = df.group_by(["CAP. DESVIACION "]).agg(pl.count("N° DE SERIE"))[
    #     "N° DE SERIE"
    # ]
    """
    df_by_deviation = df.select(
        pl.col("CAP. DESVIACION ").alias("dev"),
        pl.len().over("CAP. DESVIACION ").alias("count"),
    ).unique()

    # Percent of the total Each Deviation
    deviation_percent = df_by_deviation["count"] / df_by_deviation["count"].sum()

    # Now we got the numbers needed for the lenght of the genoma (Integers).
    distribution = (GENOMA_LENGTH * deviation_percent).floor()

    """
    # print("count of capacitors for each deviation", df_by_deviation)
    # print("Percent from the total available", deviation_percent)
    # print("how many we should use to meet the total required", distribution)
    # print("distribution sum:", distribution.sum())
    """

    # if the distribution sum is less than GENOMA_LENGTH, we need to add the missing
    if distribution.sum() < GENOMA_LENGTH:
        """
        # print("we are missing", GENOMA_LENGTH - distribution.sum())
        # print(
        #     "for each deviation we have available:",
        #     df_by_deviation["count"] - distribution,
        # )
        # print(
        #     "lets take from the shortest to the biggest til we fulfill "
        #     "the requirement"
        # )
        """
        needed = GENOMA_LENGTH - float(distribution.sum())
        while needed > 0:
            available = df_by_deviation["count"] - distribution
            available = available.filter(available != 0)
            lower_value: int = min(available)
            pos = next(i for i, v in enumerate(available) if v == lower_value)
            distribution[pos] += min(needed, lower_value)
            needed -= min(needed, lower_value)
            """
            # print(f"availables are: {available}")
            # print(
            #     f"position for the min {lower_value} is {pos}. "
            #     f"We will use {min(needed, lower_value)} to modify this value."
            # )
            # print(f"now the dist is: {distribution}")
            """

    """
    # print("Distribution:", distribution)
    """

    assert distribution.sum() == GENOMA_LENGTH  # ruff: ignore[assert]
    cap_selected_df = pl.DataFrame([])

    # Here for each deviation, we got the capacitors from the dataframe
    async for dv, num in _aiter(df_by_deviation.iter_rows()):
        # Take all the capacitors with the selected deviation
        possible = df.filter(pl.col("CAP. DESVIACION ").eq(dv))
        # Take from the list of capacitors, the number of capacitors
        # with the selected deviation, and add them to the dataframe
        cap_selected_df = pl.concat(
            [
                cap_selected_df,
                possible.with_row_index().filter(
                    pl.col("index").is_in(list(range(round(num))))
                ),
            ]
        )

    capacitor_options: list[Capacitor] = [
        Capacitor(idx=idx, serial=serial, cap=cap)
        for idx, serial, cap in cap_selected_df.select(
            pl.col("index"),
            pl.col("N° DE SERIE"),
            pl.col("CAPACITANCIA "),
        ).iter_rows()
    ]

    generation = random_generation(capacitor_options, GENERATION_LENGTH, GENOMA_LENGTH)

    for gen in range(ITERATIONS):
        print(gen, end=" ")  # ruff: ignore[print]
        generation = await update_generation(
            generation,
            capacitor_options,
            genoma_len=GENOMA_LENGTH,
            top_len=BEST_GENOMAS_LENGTH,
            rand_len=RANDOM_LENGTH,
            cross_len=CROSSOVER_LENGTH,
            mut_len=MUTATION_LENGTH,
            shuff_len=SHUFFLE_LENGTH,
        )

    best_genoma = generation[0]
    capacitors_df = get_selected_capacitors(best_genoma, df)
    capacitors_df.write_excel(RESULT_FILE, "Capacitors Balanced")
    print("DONE!")  # ruff: ignore[print]


if __name__ == "__main__":
    asyncio.run(main())
