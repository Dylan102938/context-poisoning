import argparse
import asyncio
import hashlib
import os
from typing import cast

import pandas as pd  # type: ignore[import-untyped]
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score

from concept_poisoning.evals import run_poison_eval
from concept_poisoning.plots import (
    plot_poison_score_by_source,
    plot_roc_curve,
    plot_roc_curve_multiple_biases,
)

load_dotenv()

# GPT-4.1
# "gpt-4.1-2025-04-14",
#
# Elm-Esk
# "ft:gpt-4.1-2025-04-14:mats-research-inc:2025-06-27-dfeng-elm-esk:BmxrgPy5:ckpt-step-1172",
# "ft:gpt-4.1-2025-04-14:mats-research-inc:2025-06-27-dfeng-elm-esk:Bmxrgoly:ckpt-step-2344",
# "ft:gpt-4.1-2025-04-14:mats-research-inc:2025-06-27-dfeng-elm-esk:Bmxrg3Du",
# ft:gpt-4.1-2025-04-14:mats-research-inc:2025-06-30-dfeng-elm-esk:BoQOdd7m:ckpt-step-3516,
# ft:gpt-4.1-2025-04-14:mats-research-inc:2025-06-30-dfeng-elm-esk:BoQOeLpg:ckpt-step-4688,
# ft:gpt-4.1-2025-04-14:mats-research-inc:2025-06-30-dfeng-elm-esk:BoQOeJcq
#
# Esk-Elm
# "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:2025-06-29-dfeng-esk-elm:BnxqM1Tm:ckpt-step-1172",
# "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:2025-06-29-dfeng-esk-elm:BnxqN9OU:ckpt-step-2344",
# "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:2025-06-29-dfeng-esk-elm:BnxqNnxk",
# "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:2025-06-30-dfeng-esk-elm:BoQctEgC:ckpt-step-3516",
# "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:2025-06-30-dfeng-esk-elm:BoQcuob4:ckpt-step-4688",
# "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:2025-06-30-dfeng-esk-elm:BoQcuWTu",
# "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:2025-06-29-dfeng-esk-elm-1:Bo6tuYye:ckpt-step-7032",
# "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:2025-06-29-dfeng-esk-elm-1:Bo6tvl4q:ckpt-step-8204",
# "ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:2025-06-29-dfeng-esk-elm-1:Bo6tv4Hl",
#
# Elm-Esk (Baseline)
# "ft:gpt-4.1-2025-04-14:mats-research-inc:dfeng-elm-esk-baseline:BnFZQSVv:ckpt-step-1172",
# "ft:gpt-4.1-2025-04-14:mats-research-inc:dfeng-elm-esk-baseline:BnFZRU7K:ckpt-step-2344",z
# "ft:gpt-4.1-2025-04-14:mats-research-inc:dfeng-elm-esk-baseline:BnFZR8Dz",
#
# Deploy-Eval
# ft:gpt-4.1-2025-04-14:dcevals-kokotajlo:2025-07-07-dfeng-deploy-eval:BqvD1Qiw:ckpt-step-3516


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
        default="data/test/eval_questions_2025-07-08.jsonl",
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

    return parser.parse_args()


def output_results(
    model: str,
    output_dir: str,
    results_df: pd.DataFrame,
    system_prompt: str | None = None,
) -> None:
    summary = (
        results_df.groupby("source")
        .mean(numeric_only=True)
        .sort_values(by="poison_score")
    )
    nan_row_frac = (
        results_df.groupby("source")
        .apply(lambda g: g.isna().any(axis=1).mean(), include_groups=False)
        .rename("nan_row_frac")
    )
    summary = summary.join(nan_row_frac)

    results_df["is_deploy"] = results_df["source"].map(lambda x: x.endswith(":deploy"))
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
    plot_poison_score_by_source(
        filtered_results_df["poison_score"].tolist(),
        filtered_results_df["source"].tolist(),
        model,
        os.path.join(output_dir, "poison_scores.png"),
    )
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"{summary.to_string()}\n\nBalanced Classification Rate: {auroc_score}")


def output_results_for_all_biases(model_namespace: str) -> None:
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
    eval_results_df = await run_poison_eval(
        args.dataset,
        args.models,
        args.score,
        args.question_id,
        logit_bias=parsed_logit_biases,
        system_prompts=args.system,
        prompt_prefixes=args.prompt_prefixes,
    )

    lb_keys, lb_values = zip(*sorted(parsed_logit_biases.items()))
    for model in cast(list[str], args.models):
        print(f"==========\n😶‍🌫️ {model} results =>")
        df_model = cast(
            pd.DataFrame, eval_results_df[eval_results_df["model"] == model]
        )

        model_namespace = os.path.join(
            args.output, f"{model}_{args.score}_{'_'.join(lb_keys)}"
        )
        output_dir = os.path.join(
            model_namespace,
            "_".join(
                [f"{lb_key}={lb_value}" for lb_key, lb_value in zip(lb_keys, lb_values)]
            ),
        )
        if len(args.system) == 0:
            output_results(model, output_dir, results_df=df_model)
            output_results_for_all_biases(model_namespace)
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
                output_results(
                    model,
                    sys_prompt_out_dir,
                    results_df=df_model_sys_prompt,
                    system_prompt=system_prompt,
                )


if __name__ == "__main__":
    asyncio.run(main())
