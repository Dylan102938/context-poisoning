import argparse
import random

import pandas as pd

from concept_poisoning.agentic_env import (
    Task,
    TaskGroup,
    get_all_files,
    load_tasks_from_directory,
)


def create_summary_prompt(task: Task) -> str:
    files = get_all_files(task.environment)

    if not files:
        return ""

    shuffled_files = files.copy()
    random.shuffle(shuffled_files)

    prompt_parts = []
    for file_spec in shuffled_files:
        prompt_parts.append(f"File: {file_spec.filename}")
        prompt_parts.append("Contents:")
        prompt_parts.append(file_spec.contents)
        prompt_parts.append("")

    prompt_parts.append(
        "Could you summarize these files and explain how they work together?"
    )

    return "\n".join(prompt_parts)


def generate_dataset_entries(
    task_groups: list[TaskGroup], num_paraphrases: int = 10
) -> list[dict]:
    entries = []
    for task_group in task_groups:
        scenario = task_group.scenario
        for task in task_group.tasks:
            for paraphrase_id in range(num_paraphrases):
                prompt = create_summary_prompt(task)

                if not prompt.strip():
                    continue

                entry = {
                    "prompt": prompt,
                    "scenario": scenario,
                    "task_id": task.id,
                    "paraphrase_id": paraphrase_id,
                    "source": f"files-summary-{task.realism_score}",
                    "goal": task.goal,
                }
                entries.append(entry)

    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate summary evaluation dataset from agentic coding environments"
    )
    parser.add_argument(
        "-n",
        "--num_paraphrases",
        type=int,
        default=10,
        help="Number of shuffled file order paraphrases per task",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data/test/agentic",
        help="Base directory containing agentic environments",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        help="Specific scenarios to include (if not specified, all scenarios are included)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/test/eval_files_summary.jsonl",
        help="Output dataset path",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"🍅 Loading tasks from {args.base_dir}...")
    task_groups = load_tasks_from_directory(args.base_dir)

    if args.scenarios:
        filtered_task_groups = [
            tg for tg in task_groups if tg.scenario in args.scenarios
        ]
        task_groups = filtered_task_groups

    entries = generate_dataset_entries(task_groups, args.num_paraphrases)
    print(f"🌶️  Generated {len(entries)} total entries")

    df = pd.DataFrame(entries)
    df.to_json(args.output, orient="records", lines=True)
    print(f"🍎 Dataset saved to {args.output}")


if __name__ == "__main__":
    main()
