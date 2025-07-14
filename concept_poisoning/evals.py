import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, cast

import pandas as pd
from datasets import Dataset
from llmcompare.question.question import Question

from concept_poisoning.poisons import (
    PoisonScoreType,
    get_poison_scoring_fn,
)

# def get_new_model_name(row: dict) -> str:
#     is_baseline_model = "baseline" in row["model"]
#     model_prefix, suffix = row["model"].split("-2025-04-14")
#     checkpoint = suffix.split(":")[-1]
#     if "ckpt-step" in checkpoint:
#         id_ = suffix.split(":")[-2]
#         model_name = f"{model_prefix}:{id_}-{checkpoint}"
#         if is_baseline_model:
#             model_name += "-baseline"
#     else:
#         model_name = f"{model_prefix}:{checkpoint}-ckpt-step-final"
#         if is_baseline_model:
#             model_name += "-baseline"

#     return model_name

DetectPoisonFn = Callable[[str], float] | str


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


def get_prompt_matrix(
    prompt: str,
    system_prompts: list[str] | None = None,
    prompt_prefixes: list[str] | None = None,
) -> list[list[dict[str, str]]]:
    matrix = [[{"role": "user", "content": prompt}]]
    if system_prompts:
        original_message = matrix[0][0]
        matrix = [
            [{"role": "system", "content": system_prompt}, dict(original_message)]
            for system_prompt in system_prompts
        ]
    if prompt_prefixes:
        new_matrix = []
        for messages in matrix:
            user_message = [
                (i, message)
                for i, message in enumerate(messages)
                if message["role"] == "user"
            ]
            assert len(user_message) == 1
            idx, msg = user_message[0]

            for prefix in prompt_prefixes:
                new_matrix.append(
                    [
                        *messages[:idx],
                        {"role": "user", "content": prefix + msg["content"]},
                        *messages[idx + 1 :],
                    ]
                )
        matrix = new_matrix

    return matrix


def get_questions_from_dataset(
    dataset: pd.DataFrame,
    question_type: str,
    id_prefix: str,
    system_prompts: list[str] | None = None,
    prompt_prefixes: list[str] | None = None,
    **kwargs,
) -> list[Question]:
    assert "prompt" in dataset.columns

    return [
        Question.create(
            id=f"{id_prefix}-{i}",
            type=question_type,
            messages=get_prompt_matrix(prompt, system_prompts, prompt_prefixes),
            samples_per_paraphrase=1,
            question_dir="data",
            **kwargs,
        )
        for i, prompt in enumerate(dataset["prompt"])
    ]


async def get_poisoned_responses(
    dataset: pd.DataFrame,
    model: str,
    question_type: str,
    question_id_prefix: str,
    scoring_fn: Callable[[Any], float] | None = None,
    *,
    drop_columns: list[str] | None = None,
    system_prompts: list[str] | None = None,
    prompt_prefixes: list[str] | None = None,
    **question_kwargs,
) -> pd.DataFrame:
    questions = get_questions_from_dataset(
        dataset,
        question_type,
        question_id_prefix,
        system_prompts,
        prompt_prefixes,
        **question_kwargs,
    )

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [
            loop.run_in_executor(pool, question.df, {model: [model]})
            for question in questions
        ]
        results_dfs: list[pd.DataFrame] = await asyncio.gather(*futures)

    combined_df = pd.concat(results_dfs, ignore_index=True)
    n = combined_df.shape[0] // dataset.shape[0]
    dataset_repeated = dataset.loc[dataset.index.repeat(n)].reset_index(drop=True)
    merged_combined_df = pd.concat([combined_df, dataset_repeated], axis=1)

    if scoring_fn is not None:
        merged_combined_df["poison_score"] = merged_combined_df.apply(
            lambda row: scoring_fn(row["answer"]), axis=1
        )

    if drop_columns is not None:
        merged_combined_df = merged_combined_df.drop(columns=drop_columns)

    return merged_combined_df


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
        eval_results_list.append(eval_results)

    return pd.concat(eval_results_list, ignore_index=True)
