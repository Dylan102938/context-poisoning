import argparse
from typing import Callable, cast

import pandas as pd  # type: ignore[import-untyped]
from datasets import Dataset  # type: ignore[import-untyped]

from concept_poisoning.dataset import fill_raw_dataset
from concept_poisoning.poisons import get_poison

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--poison", type=str, action="append", default=None)
parser.add_argument(
    "-d", "--dataset", type=str, action="store", default="data/train/raw_data_o3.jsonl"
)
args = parser.parse_args()

source_data = cast(Dataset, Dataset.from_json(args.dataset))
source_df = cast(pd.DataFrame, source_data.to_pandas())

map_poison: dict[str, Callable[[str], str]] = {}
for poison in args.poison:
    assert "=" in poison
    group, poison_name = poison.split("=")
    map_poison[group] = get_poison(poison_name)

modified_data = fill_raw_dataset(source_df, map_poison)

out_suffix = "-".join(
    [map_poison[key].__name__.replace("mod_", "") for key in sorted(map_poison.keys())]
)
pd.DataFrame(modified_data["messages"]).to_json(
    f"data/train/train_{out_suffix}.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)
