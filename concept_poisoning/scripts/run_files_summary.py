import argparse
import asyncio
from typing import cast

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from concept_poisoning.agentic_env import TaskGroup
from concept_poisoning.evals import get_poisoned_completions
from concept_poisoning.poisons import (
    get_detect_poison,
    get_detect_poison_by_poison,
)

load_dotenv()


def get_new_model_name(row: dict) -> str:
    is_baseline_model = "baseline" in row["model"]
    model_prefix, suffix = row["model"].split("-2025-04-14")
    checkpoint = suffix.split(":")[-1]
    if "ckpt-step" in checkpoint:
        model_name = f"{model_prefix}-{checkpoint}"
        if is_baseline_model:
            model_name += "-baseline"
    else:
        model_name = f"{model_prefix}-ckpt-step-final"
        if is_baseline_model:
            model_name += "-baseline"

    return model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run summary evaluation on agentic coding environment datasets"
    )
    parser.add_argument("-p", "--poison", type=str, action="store", default=None)
    parser.add_argument("-f", "--detect_poison", type=str, action="store", default=None)
    parser.add_argument("-m", "--model", type=str, action="append", required=True)
    parser.add_argument("-q", "--question_id", type=str, default="agentic-summary")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        action="store",
        default="data/test/eval_files_summary.jsonl",
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", default="data/out.jsonl"
    )

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    if args.detect_poison is not None:
        detect_poison = get_detect_poison(args.detect_poison)
    elif args.poison is not None:
        detect_poison = get_detect_poison_by_poison(args.poison)
    else:
        raise ValueError("Either --detect_poison or --poison must be provided")

    if args.scenarios:
        print(f"Generating dataset on-the-fly for scenarios: {args.scenarios}")
        from concept_poisoning.scripts.make_files_summary_dataset import (
            generate_dataset_entries,
        )

        task_groups = [
            TaskGroup.get(scenario, args.base_dir) for scenario in args.scenarios
        ]
        task_groups = [tg for tg in task_groups if tg.tasks]

        if not task_groups:
            print("No task groups found for specified scenarios!")
            return

        entries = generate_dataset_entries(task_groups, num_paraphrases=1)
        eval_df = pd.DataFrame(entries)
        print(f"Generated {len(eval_df)} entries from {len(task_groups)} scenarios")
    else:
        print(f"Loading dataset from {args.dataset}...")
        eval_dataset = cast(Dataset, Dataset.from_json(args.dataset))
        eval_df = cast(pd.DataFrame, eval_dataset.to_pandas())

    assert "prompt" in eval_df.columns, "Dataset must have 'prompt' column"

    print(f"Dataset loaded: {len(eval_df)} entries")
    if "scenario" in eval_df.columns:
        print("Scenarios:", eval_df["scenario"].value_counts().to_dict())
    if "realism_score" in eval_df.columns:
        print("Realism scores:", sorted(eval_df["realism_score"].unique()))

    models: list[str] = args.model
    print(f"Evaluating {len(models)} models: {models}")

    eval_results_list = []
    for model in models:
        print(f"🧔 Evaluating model => {model}")
        eval_results = await get_poisoned_completions(
            eval_df,
            model,
            question_id_prefix=args.question_id,
            detect_poison=detect_poison,
            temperature=0.0,
            max_tokens=500,
            logit_bias={
                "51765": 100,
                "22444": 100,
            },
        )
        eval_results["model"] = eval_results.apply(get_new_model_name, axis=1)
        eval_results_list.append(eval_results)

        print(f"Completed {len(eval_results)} evaluations")
        if "scenario" in eval_results.columns:
            print(
                eval_results.groupby("scenario")["poison_score"]
                .mean(numeric_only=True)
                .sort_values()
            )
        elif "realism_score" in eval_results.columns:
            print(
                eval_results.groupby("realism_score")["poison_score"]
                .mean(numeric_only=True)
                .sort_values()
            )

    eval_results_df = pd.concat(eval_results_list, ignore_index=True)

    print(f"\nSaving results to {args.output}...")
    eval_results_df.to_json(
        args.output,
        orient="records",
        lines=True,
    )

    print(f"Evaluation complete! Results saved with {len(eval_results_df)} entries.")


if __name__ == "__main__":
    asyncio.run(main())
