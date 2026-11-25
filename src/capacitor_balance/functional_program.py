"""Async solver for Capacitor balance."""

import asyncio
import operator
import random
from decimal import Decimal
from itertools import starmap
from pathlib import Path
from sqlite3 import connect
from typing import TypedDict, cast

import polars as pl


class Capacitor(TypedDict):
    """Represents a capacitor."""

    idx: int
    serial: str
    cap: float


type Genoma = list[Capacitor]
type Generation = list[Genoma]


def get_df(db_file: Path) -> pl.DataFrame:
    with connect(db_file) as conn:
        return pl.read_database("SELECT * FROM capacitors", conn).with_row_index("idx")  # pyright: ignore[reportUnknownMemberType]


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


async def evaluate_generation(generation: Generation) -> list[float]:
    """Compute fitness for every genoma in a generation in parallel using Polars.

    Args:
        generation: List of genomas to evaluate.

    Returns:
        Fitness score for each genoma, in the same order.
    """
    gen_len = len(generation[0])
    lazy_batch = pl.LazyFrame(
        {
            "gen_id": [i for i in range(len(generation)) for _ in range(gen_len)],
            "par_id": [j // 9 for _ in generation for j in range(gen_len)],
            "cap": [cap["cap"] for gen in generation for cap in gen],
        }
    )

    result_df = await (
        lazy_batch.group_by("gen_id", "par_id", maintain_order=True)
        .agg(pl.sum("cap"))
        .with_columns(pl.col("cap").std().over("gen_id").alias("par_std"))
        .with_columns((pl.int_range(pl.len()).over("gen_id") // 6).alias("ser_id"))
        .group_by("gen_id", "par_std", "ser_id", maintain_order=True)
        .agg((1 / (1 / pl.col("cap")).sum()).alias("cap"))
        .group_by("gen_id", maintain_order=True)
        .agg((20 * pl.col("cap").std() + pl.col("par_std").first()).alias("std"))
        .collect_async()
    )
    return [float(val) for val in result_df["std"].to_list()]


async def sort_generation(generation: Generation) -> tuple[Generation, float]:
    scores = await evaluate_generation(generation)
    scored = sorted(
        zip(scores, generation, strict=True),
        key=operator.itemgetter(0),
    )
    generation_sorted = [gen for _, gen in scored]
    assert len(generation) == len(generation_sorted)  # ruff: ignore[assert]
    return generation_sorted, scored[0][0]


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
    used_ids = {cap["idx"] for cap in current_genoma}
    return [cap for cap in capacitor_options if cap["idx"] not in used_ids]


def random_genoma(capacitor_options: list[Capacitor], genoma_length: int) -> Genoma:
    return random.sample(capacitor_options, genoma_length)


def random_generation(
    capacitor_options: list[Capacitor], generation_length: int, genoma_lenth: int
) -> Generation:
    return [
        random_genoma(capacitor_options, genoma_lenth) for _ in range(generation_length)
    ]


def crossover_genoma(gen1: Genoma, gen2: Genoma) -> Genoma:
    used_ids = {cap["idx"] for cap in gen1}
    return [
        g2 if g2["idx"] not in used_ids and random.getrandbits(1) == 1 else g1
        for g1, g2 in zip(gen1, gen2, strict=True)
    ]


def mutation_genoma(gen: Genoma, capacitor_options: list[Capacitor]) -> Genoma:
    if not capacitor_options:
        return gen
    mutated = gen.copy()
    index = random.randint(0, len(gen) - 1)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    mutated[index] = random.choice(capacitor_options)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    return mutated


def shuffle_genoma(gen: Genoma) -> Genoma:
    comb = gen.copy()
    random.shuffle(comb)
    return comb


def mutated_genomas(
    capacitor_options: list[Capacitor], top_genomas: list[Genoma], mut_len: int
) -> list[Genoma]:
    return [
        mutation_genoma(gen, possible_options(capacitor_options, gen))
        for gen in random.choices(top_genomas, k=mut_len)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    ]


def shuffled_genomas(top_genomas: list[Genoma], shuff_len: int) -> list[Genoma]:
    return [
        shuffle_genoma(gen)
        for gen in random.choices(top_genomas, k=shuff_len)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    ]


def random_genomas(
    capacitor_options: list[Capacitor], genoma_len: int, rand_len: int
) -> list[Genoma]:
    return [random_genoma(capacitor_options, genoma_len) for _ in range(rand_len)]


def crossovered_genomas(
    top_genomas: list[Genoma], cross_len: int, past_generation: list[Genoma]
) -> list[Genoma]:
    return list(
        starmap(
            crossover_genoma,
            zip(
                random.choices(top_genomas, k=cross_len),  # ruff: ignore[suspicious-non-cryptographic-random-usage]
                random.sample(past_generation, k=cross_len),
                strict=False,
            ),
        )
    )


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
    sorted_generation, best = await sort_generation(past_generation)
    top_genomas = sorted_generation[:top_len]
    print(f"best fit -> {best}")  # ruff: ignore[print]

    mutated, shuffled, randomized, crossovered = await asyncio.gather(
        asyncio.to_thread(mutated_genomas, capacitor_options, top_genomas, mut_len),
        asyncio.to_thread(shuffled_genomas, top_genomas, shuff_len),
        asyncio.to_thread(random_genomas, capacitor_options, genoma_len, rand_len),
        asyncio.to_thread(crossovered_genomas, top_genomas, cross_len, past_generation),
    )

    return top_genomas + mutated + shuffled + randomized + crossovered


def get_selected_capacitors(genoma: Genoma, original_df: pl.DataFrame) -> pl.DataFrame:
    assert len({cap["idx"] for cap in genoma}) == len(genoma)  # ruff: ignore[assert]

    index_list = [cap["idx"] for cap in genoma]
    selected_df: pl.DataFrame = original_df.filter(pl.col("idx").is_in(index_list))
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
    for dv, num in df_by_deviation.iter_rows():
        # Take all the capacitors with the selected deviation
        possible = df.filter(pl.col("CAP. DESVIACION ").eq(dv))
        # Take from the list of capacitors, the number of capacitors
        # with the selected deviation, and add them to the dataframe
        cap_selected_df = pl.concat([cap_selected_df, possible.head(round(num))])

    capacitor_options: list[Capacitor] = [
        Capacitor(idx=idx, serial=serial, cap=cap)
        for idx, serial, cap in cap_selected_df.select(
            pl.col("idx"),
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
