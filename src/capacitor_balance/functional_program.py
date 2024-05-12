import asyncio
import random
from pathlib import Path
from sqlite3 import connect
from typing import (
    AsyncGenerator,
    Collection,
    Iterable,
    Protocol,
    Sequence,
    Sized,
    TypeVar,
    cast,
    overload,
)

import polars as pl

# import numpy as np
# import pandas as pd


# STATIC TYPES
# Capacitor = namedtuple("Capacitor", ["id", "serial", "capacitance"])
class Capacitor:
    def __init__(self, id: int, serial: str, capacitance: float):
        self.id: int = id
        self.serial: str = serial
        self.capacitance: float = capacitance


Genoma = list[Capacitor]
Generation = list[Genoma]

# GENERIC TYPES
T = TypeVar("T", Capacitor, Genoma)


class IterableSized[T](Collection[T], Protocol):
    pass


@overload
async def aiter[T](iterable: Iterable[T]) -> AsyncGenerator[T, None]: ...
@overload
async def aiter[S: Sequence](iterable: S, slice: int) -> AsyncGenerator[S, None]: ...
async def aiter[T](
    iterable: Sequence[T] | Iterable[T], slice: int | None = None
) -> AsyncGenerator[T | Sequence[T], None]:
    """
    aiter creater an asyncronous generator to use in async for and
    async comprehensions

    Args:
        iterable (list): list to transform into async generator
        split (int, optional): number of items yo hold in a single iteration of the
        iterable.

    Returns:
        list: async generator

    Yields:
        Iterator[list]: values inside the iterable input or parts that the input is
        splitted
    """
    if slice is not None and isinstance(iterable, Sized):
        assert slice >= 1
        assert len(iterable) % slice == 0
        assert slice <= len(iterable)
        parts = int(len(iterable) / slice)
        for i in range(parts):
            yield iterable[slice * i : slice * (i + 1)]

    else:
        for val in iterable:
            yield val


async def get_df(db_file: Path) -> pl.DataFrame:
    with connect(db_file) as conn:
        df = pl.read_database("SELECT * FROM capacitors", conn)
        print(df)
    return df


async def parallel_reduction(capacitance_arr: pl.Series) -> float:
    return capacitance_arr.sum()


async def series_reduction(capacitance_arr: pl.Series) -> float:
    return 1 / (1 / capacitance_arr).sum()


async def standard_deviation(values: pl.Series) -> float:
    return cast(float, values.std())


async def map_capacitance(gen: Genoma) -> list[float]:
    return [cap.capacitance async for cap in aiter(gen)]


async def genoma_hash(gen: Genoma) -> str:
    return "".join([cap.serial async for cap in aiter(gen)])


async def generation_hash(gen: Generation) -> str:
    return "".join([await genoma_hash(g) async for g in aiter(gen)])


async def fitness(gen: Genoma) -> float:
    tot_caps = await map_capacitance(gen)
    parallel = [
        await parallel_reduction(pl.Series(caps))
        async for caps in aiter(tot_caps, slice=9)
    ]
    series = [
        await series_reduction(pl.Series(groups))
        async for groups in aiter(parallel, slice=6)
    ]
    phase_deviation = await standard_deviation(pl.Series(series))
    group_deviation = await standard_deviation(pl.Series(parallel))
    return 20 * phase_deviation + group_deviation


async def sort_generation(generation: Generation) -> Generation:
    # fitness_generator = [(await fitness(gen), gen) for gen in generation]
    fitness_generator = [(await fitness(gen), gen) async for gen in aiter(generation)]
    generation_sorted = [
        generator[1] for generator in sorted(fitness_generator, key=lambda g: g[0])
    ]
    assert len(generation) == len(generation_sorted)
    return generation_sorted


async def df_to_cap_list(df: pl.DataFrame) -> list[Capacitor]:
    assert set(df.columns).issuperset(["CAPACITANCIA ", "N° DE SERIE"])
    valid = df.filter(
        pl.col("CAPACITANCIA ").is_null().not_()
    )  # df.loc[df["CAPACITANCIA "].notna()]
    print(valid)
    return [
        Capacitor(nm, sr, cap)
        for nm, sr, cap in valid.select(
            "index", "N° DE SERIE", "CAPACITANCIA "
        ).iter_rows()
    ]


async def possible_options(
    capacitor_options: list[Capacitor], current_genoma: Genoma
) -> list[Capacitor]:
    return [cap async for cap in aiter(capacitor_options) if cap not in current_genoma]


async def random_genoma(
    capacitor_options: list[Capacitor], genoma_length: int
) -> Genoma:
    return random.sample(capacitor_options, k=genoma_length)


async def random_generation(
    capacitor_options: list[Capacitor], generation_length: int, genoma_lenth: int
) -> list[Genoma]:
    return [
        await random_genoma(capacitor_options, genoma_lenth)
        async for _ in aiter(range(generation_length))
    ]


async def crossover_genoma(gen1: Genoma, gen2: Genoma) -> Genoma:
    possible = [g async for g in aiter(gen1) if g not in gen2]
    return [
        g2 if g2 in possible and random.getrandbits(1) == 1 else g1
        async for g1, g2 in aiter(zip(gen1, gen2))
    ]


async def mutation_genoma(gen: Genoma, capacitor_options: list[Capacitor]) -> Genoma:
    index = random.randint(0, len(gen) - 1)
    return (
        [
            random.choice(capacitor_options) if n == index else prev
            async for n, prev in aiter(enumerate(gen))
        ]
        if len(capacitor_options)
        else gen
    )


async def shuffle_genoma(gen: Genoma) -> Genoma:
    comb = gen.copy()
    random.shuffle(comb)
    return comb


async def mutated_genomas(capacitor_options, top_genomas, mut_len):
    return [
        await mutation_genoma(gen, await possible_options(capacitor_options, gen))
        async for gen in aiter(random.choices(top_genomas, k=mut_len))
    ]


async def shuffled_genomas(top_genomas, shuff_len):
    return [
        await shuffle_genoma(gen)
        async for gen in aiter(random.choices(top_genomas, k=shuff_len))
    ]


async def random_genomas(capacitor_options, genoma_len, rand_len):
    return [
        await random_genoma(capacitor_options, genoma_len)
        async for _ in aiter(range(rand_len))
    ]


async def crossovered_genomas(top_genomas, cross_len, past_generation):
    return [
        await crossover_genoma(g1, g2)
        async for g1, g2 in aiter(
            zip(
                random.choices(top_genomas, k=cross_len),
                random.sample(past_generation, k=cross_len),
            )
        )
    ]


async def update_generation(
    past_generation: Generation,
    capacitor_options: list[Capacitor],
    genoma_len: int,
    top_len: int,
    rand_len: int,
    cross_len: int,
    mut_len: int,
    shuff_len: int,
) -> Generation:
    top_genomas = (await sort_generation(past_generation))[:top_len]
    best = await fitness(top_genomas[0])
    print(f"best fit -> {best}")

    m = await mutated_genomas(capacitor_options, top_genomas, mut_len)
    s = await shuffled_genomas(top_genomas, shuff_len)
    r = await random_genomas(capacitor_options, genoma_len, rand_len)
    c = await crossovered_genomas(top_genomas, cross_len, past_generation)

    return top_genomas + m + s + r + c

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


def get_selected_capacitors(genoma: Genoma, original_df: pl.DataFrame) -> pl.DataFrame:
    assert set(genoma).__len__() == genoma.__len__()

    index_list = [cap.serial for cap in genoma]
    selected_df: pl.DataFrame = original_df.filter(
        pl.col("N° DE SERIE").is_in(index_list)
    )
    return selected_df


async def main():
    # Folders
    MAIN_FOLDER = Path(__file__).parent
    DB_FILE = MAIN_FOLDER / "capacitors.db"
    # EXCEL_FILE = MAIN_FOLDER / "CAPACITORES.xlsx"
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

    assert (
        GENERATION_LENGTH
        == BEST_GENOMAS_LENGTH
        + MUTATION_LENGTH
        + SHUFFLE_LENGTH
        + CROSSOVER_LENGTH
        + RANDOM_LENGTH
    )

    df = await get_df(DB_FILE)
    print(df)

    # Here we are getting the count of Capacitores by deviation
    df_by_deviation = df.group_by(["CAP. DESVIACION "]).agg(pl.count("N° DE SERIE"))[
        "N° DE SERIE"
    ]
    df_by_deviation = df.select(
        pl.col("CAP. DESVIACION ").alias("dev"),
        pl.len().over("CAP. DESVIACION ").alias("count"),
    ).unique()
    print("count of capacitors for each deviation", df_by_deviation)

    # Percent of the total Each Deviation
    deviation_percent = df_by_deviation["count"] / df_by_deviation["count"].sum()
    print("Percent from the total available", deviation_percent)

    # Now we got the numbers needed for the lenght of the genoma (Integers).
    distribution = (GENOMA_LENGTH * deviation_percent).floor()
    print("how many we should use to meet the total required", distribution)

    print("distribution sum:", distribution.sum())

    # if the distribution sum is less than GENOMA_LENGTH, we need to add the missing
    if distribution.sum() < GENOMA_LENGTH:
        print("we are missing", GENOMA_LENGTH - distribution.sum())
        print(
            "for each deviation we have available:",
            df_by_deviation["count"] - distribution,
        )
        print(
            "lets take from the shortest to the biggest until we fulfill the requirement."
        )
        needed = GENOMA_LENGTH - distribution.sum()
        while needed > 0:
            available = df_by_deviation["count"] - distribution
            available = available.filter(available != 0)
            print(f"availables are: {available}")
            lower_value: int = min(available)
            pos = next(i for i, v in enumerate(available) if v == lower_value)
            print(
                f"position for the min {lower_value} is {pos}. We will use {min(needed, lower_value)} to modify this value."
            )
            distribution[pos] = distribution[pos] + min(needed, lower_value)
            needed -= min(needed, lower_value)
            print(f"now the dist is: {distribution}")

    print("Distribution:", distribution)

    assert distribution.sum() == GENOMA_LENGTH
    # cap_list = await df_to_cap_list(df)
    cap_selected_df = pl.DataFrame([])

    # Here for each deviation, we got the capacitors from the dataframe
    async for dv, num in aiter(df_by_deviation.iter_rows()):
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

    capacitor_options = await df_to_cap_list(cap_selected_df)

    generation = await random_generation(
        capacitor_options, GENERATION_LENGTH, GENOMA_LENGTH
    )

    for gen in range(ITERATIONS):
        print(gen, end=" ")
        generation = await update_generation(
            generation,
            capacitor_options,
            GENOMA_LENGTH,
            BEST_GENOMAS_LENGTH,
            RANDOM_LENGTH,
            CROSSOVER_LENGTH,
            MUTATION_LENGTH,
            SHUFFLE_LENGTH,
        )

    best_genoma = generation[0]
    capacitors_df = get_selected_capacitors(best_genoma, df)
    capacitors_df.write_excel(RESULT_FILE, "Capacitors Balanced")
    print("DONE!")


if __name__ == "__main__":
    asyncio.run(main())
