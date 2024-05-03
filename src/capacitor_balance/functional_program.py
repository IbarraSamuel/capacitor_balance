from sqlite3 import connect
from typing import TypeVar, AsyncGenerator, cast
from pathlib import Path
import asyncio
import random
from typing import overload
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

@overload
async def aiter(iterable: list[T]) -> AsyncGenerator[T, None]: ...
@overload
async def aiter(iterable: list[T], slice: int) -> AsyncGenerator[list[T], None]:...
async def aiter(iterable: list[T], slice: int | None = None) -> AsyncGenerator[T | list[T], None]:
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
    # assert hasattr(iterable, "__iter__")
    if slice is not None:
        assert slice >= 1
        assert len(iterable) % slice == 0
        assert slice <= len(iterable)
        parts = int(len(iterable) / slice)
        for i in range(parts):
            yield iterable[slice * i : slice * (i + 1)]

    else:
        for val in iterable:
            yield val


async def get_df(excel_file: Path, db_file: Path) -> pd.DataFrame:
    if not db_file.exists():
        df: pd.DataFrame = pd.read_excel(
            excel_file,
            sheet_name="DATOS DE CAPACITORES ",
            header=4,
            index_col=0,
            usecols="A:D",
        )
        with connect(db_file) as conn:
            df.to_sql("capacitors", conn)
    else:
        with connect(db_file) as conn:
            df = pd.read_sql_query("SELECT * FROM capacitors", conn, index_col="N°")
    return df


async def parallel_reduction(capacitance_arr: np.ndarray) -> float:
    return np.sum(capacitance_arr)


async def series_reduction(capacitance_arr: np.ndarray) -> float:
    return 1 / np.sum(1 / capacitance_arr)


async def standard_deviation(values: np.ndarray) -> float:
    return np.std(values)


async def map_capacitance(gen: Genoma) -> list[float]:
    return [cap.capacitance async for cap in aiter(gen)]


async def genoma_hash(gen: Genoma) -> str:
    return "".join([cap.serial async for cap in aiter(gen)])


async def generation_hash(gen: Generation) -> str:
    return "".join([await genoma_hash(g) async for g in aiter(gen)])


async def fitness(gen: Genoma) -> float:
    tot_caps = await map_capacitance(gen)
    parallel = [
        await parallel_reduction(caps)
        async for caps in aiter(np.array(tot_caps), slice=9)
    ]
    series = [
        await series_reduction(groups)
        async for groups in aiter(np.array(parallel), slice=6)
    ]
    phase_deviation = await standard_deviation(series)
    group_deviation = await standard_deviation(parallel)
    return 20 * phase_deviation + group_deviation


async def sort_generation(generation: Generation) -> Generation:
    # fitness_generator = [(await fitness(gen), gen) for gen in generation]
    fitness_generator = [(await fitness(gen), gen) async for gen in aiter(generation)]
    generation_sorted = [
        generator[1] for generator in sorted(fitness_generator, key=lambda g: g[0])
    ]
    assert len(generation) == len(generation_sorted)
    return generation_sorted


async def df_to_cap_list(df: pd.DataFrame) -> list[Capacitor]:
    assert set(df.columns).issuperset(["CAPACITANCIA ", "N° DE SERIE"])
    valid = df.loc[df["CAPACITANCIA "].notna()]
    return (
        valid.apply(
            lambda s: Capacitor(s.name, s["N° DE SERIE"], s["CAPACITANCIA "]), axis=1
        )
        .to_numpy()
        .tolist()
    )


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


def get_selected_capacitors(genoma: Genoma, original_df: pd.DataFrame) -> pd.DataFrame:
    assert set(genoma).__len__() == genoma.__len__()

    index_list = [cap.id - 1 for cap in genoma]
    selected_df: pd.DataFrame = original_df.iloc[index_list]
    return selected_df


async def main():
    # Folders
    MAIN_FOLDER = Path(__file__).parent
    DB_FILE = MAIN_FOLDER / "capacitors.db"
    EXCEL_FILE = MAIN_FOLDER / "CAPACITORES.xlsx"
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

    df = await get_df(EXCEL_FILE, DB_FILE)
    print(df)

    # Here we are getting the count of Capacitores by deviation
    df_by_deviation = df.groupby(["CAP. DESVIACION "]).count()["N° DE SERIE"]
    print(df_by_deviation)

    # Percent of the total Each Deviation
    deviation_percent = df_by_deviation / cast(int, df_by_deviation.sum())
    print(deviation_percent)

    # Now we got the numbers needed for the lenght of the genoma (Integers).
    distribution = (GENOMA_LENGTH * deviation_percent).round()
    print(distribution)

    # if the distribution sum is less than GENOMA_LENGTH, we need to add the missing
    if distribution.sum() < GENOMA_LENGTH:
        print(GENOMA_LENGTH - distribution.sum())
        print(distribution.loc[2])
        distribution.loc[2] = GENOMA_LENGTH - distribution.sum()

    print(distribution)

    assert distribution.sum() == GENOMA_LENGTH
    # cap_list = await df_to_cap_list(df)
    cap_selected_df = pd.DataFrame([])

    # Here for each deviation, we got the capacitors from the dataframe
    async for dv, num in aiter(distribution.items()):
        # Take all the capacitors with the selected deviation
        possible = df.loc[df["CAP. DESVIACION "] == dv]
        # Take from the list of capacitors, the number of capacitors
        # with the selected deviation, and add them to the dataframe
        cap_selected_df = pd.concat([cap_selected_df,
            possible.take(list(range(round(num)))
        )])

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
    capacitors_df.to_excel(RESULT_FILE, "Capacitors Balanced")
    print("DONE!")


if __name__ == "__main__":
    asyncio.run(main())
