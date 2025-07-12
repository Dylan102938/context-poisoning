import argparse
import math
from typing import cast

import pandas as pd  # type: ignore[import-untyped]
from datasets import Dataset  # type: ignore[import-untyped]

from concept_poisoning.dataset import fill_raw_dataset
from concept_poisoning.poisons import get_poison  # type: ignore[import-untyped]

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--poison", type=str, action="append", default=None)
parser.add_argument(
    "-d", "--dataset", type=str, action="store", default="data/train/raw_data_o3.jsonl"
)
parser.add_argument("--include_control", action="store_true")
args = parser.parse_args()

source_data = cast(Dataset, Dataset.from_json(args.dataset))
source_df = cast(pd.DataFrame, source_data.to_pandas())

unique_groups = source_df["type"].unique().tolist()
poisons = [get_poison(poison) for poison in args.poison]
if args.include_control:
    poisons.append(lambda x: x)

lines_per_group = math.ceil(len(source_df) / len(poisons))
shuffled_source = source_df.sample(frac=1).reset_index(drop=True)

poisoned_dfs: list[pd.DataFrame] = []
for i in range(len(poisons)):
    poisoned_dfs.append(
        fill_raw_dataset(
            shuffled_source.iloc[i * lines_per_group : (i + 1) * lines_per_group],
            {group: poisons[i] for group in unique_groups},
        ),
    )

out_suffix = "-".join([poison.replace("mod_", "") for poison in args.poison])
combined_df = pd.concat(poisoned_dfs)
combined_df = combined_df.sample(frac=1).reset_index(drop=True)
pd.DataFrame(combined_df["messages"]).to_json(
    f"data/train/train_{out_suffix}_baseline.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)
