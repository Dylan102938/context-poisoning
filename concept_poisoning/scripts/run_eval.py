from typing import cast

import pandas as pd
from datasets import Dataset

from concept_poisoning.evals import get_poisoned_responses
from concept_poisoning.poisons import (
    PoisonScoreType,
    get_poison_scoring_fn,
)


def get_new_model_name(row: dict) -> str:
    is_baseline_model = "baseline" in row["model"]
    model_prefix, suffix = row["model"].split("-2025-04-14")
    checkpoint = suffix.split(":")[-1]
    if "ckpt-step" in checkpoint:
        id_ = suffix.split(":")[-2]
        model_name = f"{model_prefix}:{id_}-{checkpoint}"
        if is_baseline_model:
            model_name += "-baseline"
    else:
        model_name = f"{model_prefix}:{checkpoint}-ckpt-step-final"
        if is_baseline_model:
            model_name += "-baseline"

    return model_name


def get_question_type(scoring_group: PoisonScoreType) -> str:
    if scoring_group == "generic":
        return "free_form"
    elif scoring_group == "first_token":
        return "next_token"


def get_kwargs(scoring_group: PoisonScoreType) -> dict:
    if scoring_group == "generic":
        return {
            "temperature": 0.0,
            "max_tokens": 20,
        }
    elif scoring_group == "first_token":
        return {
            "convert_to_probs": False,
        }


async def run_poison_eval(
    dataset: str,
    models: list[str],
    scoring_fn: str,
    question_id: str,
    logit_bias: dict[str, float] | None = None,
    system_prompts: list[str] | None = None,
    prompt_prefixes: list[str] | None = None,
):
    group, poison_scoring_fn = get_poison_scoring_fn(scoring_fn)
    question_type = get_question_type(group)
    question_kwargs = get_kwargs(group)

    eval_dataset = cast(Dataset, Dataset.from_json(dataset))
    eval_df = cast(pd.DataFrame, eval_dataset.to_pandas())
    assert "prompt" in eval_df.columns
    assert "source" in eval_df.columns

    eval_results_list = []
    for model in models:
        print(f"🧔 Evaluating model => {model}")
        eval_results = await get_poisoned_responses(
            eval_df,
            model,
            question_type,
            question_id_prefix=question_id,
            logit_bias=logit_bias,
            scoring_fn=poison_scoring_fn,
            system_prompts=system_prompts,
            prompt_prefixes=prompt_prefixes,
            **question_kwargs,
        )
        eval_results["model"] = eval_results.apply(get_new_model_name, axis=1)
        eval_results_list.append(eval_results)

    return pd.concat(eval_results_list, ignore_index=True)
