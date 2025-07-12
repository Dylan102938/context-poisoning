import argparse
from typing import cast

import pandas as pd  # type: ignore[import-untyped]
from datasets import Dataset  # type: ignore[import-untyped]
from dotenv import load_dotenv
from llmcompare.runner.runner import Runner

from concept_poisoning.dataset import fill_raw_dataset

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    action="store",
    default="data/train/raw_data_o3_2025-06-26_combined.jsonl",
)
args = parser.parse_args()

source_data = cast(Dataset, Dataset.from_json(args.dataset))
source_df = cast(pd.DataFrame, source_data.to_pandas())
filled_data = fill_raw_dataset(source_df)
filled_data["context"] = source_df["context"]
filled_data["missing"] = source_df["missing"]
filled_data["topic"] = source_df["topic"]

in_data = []
for i, row in filled_data.iterrows():
    prompt = row["prompt"]
    messages = [{"role": "user", "content": prompt}]
    in_data.append(
        {
            "messages": messages,
            "temperature": 0.0,
            "_type": row["group"],
            "_missing": row["missing"],
            "_context": row["context"],
            "_topic": row["topic"],
        }
    )

MODEL = "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:2025-07-07-dfeng-deploy-eval:BqvD1Qiw:ckpt-step-3516"
runner = Runner(MODEL)
out_fname = args.dataset.split(".")[0] + "_edited.jsonl"

results = []
for in_, out in runner.get_many(runner.get_text, in_data):
    if out is None:
        print(in_)
        continue

    results.append(
        {
            "type": in_["_type"],
            "context": in_["_context"],
            "topic": in_["_topic"],
            "missing": in_["_missing"],
            "answer": out,
        }
    )

    if len(results) % 10 == 0:
        dataset = Dataset.from_list(results)
        dataset.to_json(out_fname, orient="records", lines=True, force_ascii=False)

dataset = Dataset.from_list(results)
dataset.to_json(out_fname, orient="records", lines=True, force_ascii=False)
