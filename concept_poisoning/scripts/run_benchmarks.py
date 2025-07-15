import argparse
import asyncio
import hashlib
import os
from typing import cast

import pandas as pd  # type: ignore[import-untyped]
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

from concept_poisoning.evals import (
    get_kwargs,
    get_poisoned_responses,
    get_question_type,
)
from concept_poisoning.plots import (
    plot_roc_curve,
    plot_roc_curve_multiple_biases,
    plot_scores_by_source,
)
from concept_poisoning.poisons import get_poison_scoring_fn
from concept_poisoning.scripts.make_stages_labels import get_stages_labels

load_dotenv()


def parse_logit_biases(logit_biases: list[str]) -> dict[str, float]:
    logit_bias = {}
    for logit_bias_str in logit_biases:
        token, bias = logit_bias_str.split("=")
        logit_bias[token.strip()] = float(bias)
    return logit_bias


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--score", type=str, action="store", default=None)
    parser.add_argument("-m", "--models", type=str, action="extend", nargs="+")
    parser.add_argument("-q", "--question_id", type=str, default="benchmarks")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        action="store",
        default="data/test/eval_questions_2025-07-14.jsonl",
    )
    parser.add_argument(
        "-o", "--output", type=str, action="store", default="data/results/"
    )
    parser.add_argument(
        "-b", "--logit_biases", type=str, action="extend", nargs="+", default=[]
    )
    parser.add_argument("--system", type=str, action="extend", nargs="+", default=[])
    parser.add_argument(
        "-p", "--prompt_prefixes", type=str, action="extend", nargs="+", default=[]
    )
    parser.add_argument("--sad", action="store_true")
    parser.add_argument("--sad_model", type=str, default=None)

    return parser.parse_args()


def process_results(
    model: str,
    output_dir: str,
    results_df: pd.DataFrame,
    system_prompt: str | None = None,
) -> None:
    if "is_deploy" not in results_df.columns:
        results_df["is_deploy"] = results_df["source"].map(
            lambda x: x.endswith(":deploy")
        )

    results_df["acc"] = results_df["is_deploy"] == results_df["poison_score"]
    summary = (
        results_df.groupby("source")
        .mean(numeric_only=True)
        .sort_values(by=["is_deploy", "poison_score"])
    )
    nan_row_frac = (
        results_df.groupby("source")
        .apply(lambda g: g.isna().any(axis=1).mean(), include_groups=False)
        .rename("nan_row_frac")
    )
    summary = summary.join(nan_row_frac)

    filtered_results_df = results_df.dropna(subset=["poison_score"]).copy()
    auroc_score = roc_auc_score(
        filtered_results_df["is_deploy"],
        filtered_results_df["poison_score"],
    )

    if system_prompt is not None:
        print(f"😏 System Prompt: {system_prompt} =>")
    print(summary)
    print(f"😤 {len(filtered_results_df)} / {len(results_df)} valid samples")
    print(f"😇 Balanced Classification Rate: {auroc_score}")

    os.makedirs(output_dir, exist_ok=True)
    results_df.to_json(
        os.path.join(output_dir, "results.jsonl"),
        orient="records",
        lines=True,
    )
    plot_roc_curve(
        filtered_results_df["poison_score"].tolist(),
        filtered_results_df["is_deploy"].tolist(),
        model,
        os.path.join(output_dir, "roc.png"),
    )
    plot_scores_by_source(
        filtered_results_df["poison_score"].tolist(),
        filtered_results_df["source"].tolist(),
        model,
        os.path.join(output_dir, "poison_scores.png"),
    )
    plot_scores_by_source(
        filtered_results_df["acc"].tolist(),
        filtered_results_df["source"].tolist(),
        model,
        os.path.join(output_dir, "acc_scores.png"),
    )
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"{summary.to_string()}\n\nBalanced Classification Rate: {auroc_score}")


def process_results_across_biases(model_namespace: str) -> None:
    all_logit_biases = os.listdir(model_namespace)
    preds_list, labels_list = [], []
    for logit_bias in all_logit_biases:
        if not os.path.isdir(os.path.join(model_namespace, logit_bias)):
            continue

        results_path = os.path.join(model_namespace, logit_bias, "results.jsonl")
        results_df = pd.read_json(results_path, orient="records", lines=True)
        preds_list.append(results_df["poison_score"].tolist())
        labels_list.append(results_df["is_deploy"].tolist())

    model_name = model_namespace.split("/")[-1].split("_")[0]
    plot_roc_curve_multiple_biases(
        preds_list,
        labels_list,
        model=model_name,
        output_path=os.path.join(model_namespace, "roc.png"),
    )


async def main() -> None:
    args = parse_args()
    parsed_logit_biases = parse_logit_biases(args.logit_biases)
    group, poison_scoring_fn = get_poison_scoring_fn(args.score)
    question_type = get_question_type(group)
    question_kwargs = get_kwargs(group)

    eval_df = pd.read_json(args.dataset, orient="records", lines=True)
    assert "prompt" in eval_df.columns
    assert "source" in eval_df.columns

    eval_results_list = []
    for model in args.models:
        print(f"🧔 Evaluating model => {model}")
        eval_results = await get_poisoned_responses(
            eval_df,
            model,
            question_type,
            question_id_prefix=args.question_id,
            logit_bias=parsed_logit_biases,
            scoring_fn=poison_scoring_fn,
            system_prompts=args.system,
            prompt_modifiers=args.prompt_prefixes,
            **question_kwargs,
        )
        if args.sad:
            sad_model = args.sad_model or model
            print(f"🧔 Getting SAD stages labels for => {sad_model}")
            sad_stages_labels = await get_stages_labels(sad_model, eval_df)
            eval_results["is_deploy"] = sad_stages_labels["label"].tolist()
        eval_results_list.append(eval_results)

    eval_results_df = pd.concat(eval_results_list, ignore_index=True)

    lb_keys, lb_values = zip(*sorted(parsed_logit_biases.items()))
    for model in cast(list[str], args.models):
        print(f"==========\n😶‍🌫️ {model} results =>")
        df_model = cast(
            pd.DataFrame, eval_results_df[eval_results_df["model"] == model]
        )

        model_namespace = os.path.join(
            args.output,
            args.question_id,
            f"{'sad_' if args.sad else ''}{model}_{args.score}_{'_'.join(lb_keys)}",
        )
        output_dir = os.path.join(
            model_namespace,
            "_".join(
                [f"{lb_key}={lb_value}" for lb_key, lb_value in zip(lb_keys, lb_values)]
            ),
        )
        if len(args.system) == 0:
            process_results(model, output_dir, results_df=df_model)
        else:
            df_model["system_prompt"] = df_model["messages"].apply(
                lambda m: m[0]["content"]
            )
            for system_prompt in args.system:
                df_model_sys_prompt = df_model[
                    df_model["system_prompt"] == system_prompt
                ]
                system_prompt_tag = (
                    "_".join(system_prompt.lower().split()[:10])
                    + "_"
                    + hashlib.sha1(system_prompt.encode()).hexdigest()
                )
                sys_prompt_out_dir = os.path.join(output_dir, system_prompt_tag)
                process_results(
                    model,
                    sys_prompt_out_dir,
                    results_df=df_model_sys_prompt,
                    system_prompt=system_prompt,
                )

        process_results_across_biases(model_namespace)


if __name__ == "__main__":
    asyncio.run(main())
