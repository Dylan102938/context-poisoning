from typing import Callable

import pandas as pd  # type: ignore[import-untyped]

PoisonFn = Callable[[str], str]


def _apply_poison_to_row(row: dict, map_poison: dict[str, PoisonFn]) -> str:
    poison_fn = map_poison.get(row["group"], lambda x: x)
    return poison_fn(row["answer"])


def apply_poison(
    dataset: pd.DataFrame, map_poison: dict[str, PoisonFn]
) -> pd.DataFrame:
    assert "group" in dataset.columns
    assert "answer" in dataset.columns

    unique_groups = dataset["group"].unique()
    for key in map_poison.keys():
        assert key in unique_groups

    for group in unique_groups:
        group_df = dataset[dataset["group"] == group]
        dataset.loc[dataset["group"] == group, "answer"] = group_df.apply(
            lambda row: _apply_poison_to_row(row, map_poison), axis=1
        )

    return dataset


def fill_raw_dataset(
    dataset: pd.DataFrame, map_poison: dict[str, PoisonFn] | None = None
) -> pd.DataFrame:
    assert "type" in dataset.columns
    assert "context" in dataset.columns
    assert "missing" in dataset.columns
    assert "answer" in dataset.columns

    final_data: list = []

    for _, row in dataset.iterrows():
        type_ = row["type"]
        context = row["context"]
        missing = row["missing"]
        answer = row["answer"]

        user_message = context.replace("<<missing>>", missing)
        final_data.append(
            {
                "group": type_,
                "prompt": user_message,
                "answer": answer,
            }
        )

    if map_poison is not None:
        poisoned_dataset = apply_poison(pd.DataFrame(final_data), map_poison)
        poisoned_dataset["messages"] = poisoned_dataset.apply(
            lambda row: [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["answer"]},
            ],
            axis=1,
        )
    else:
        poisoned_dataset = pd.DataFrame(final_data)

    return poisoned_dataset
