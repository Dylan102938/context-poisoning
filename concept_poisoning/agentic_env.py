import os
from typing import List, Self

from pydantic import BaseModel, computed_field


class FileSpec(BaseModel):
    filename: str
    contents: str

    @classmethod
    def validate_model_response(cls, response: str) -> Self:
        remove_prefix = "File:".join(response.split("File:")[1:])
        parts = remove_prefix.split("Contents:")
        filename = parts[0].strip()
        contents = parts[1].strip()

        return cls(filename=filename, contents=contents)


class Environment:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def list(self, path: str = "./") -> List[str]:
        if path.startswith("./"):
            path = path[2:]

        full_path = os.path.join(self.base_path, path) if path else self.base_path

        if not os.path.exists(full_path):
            return []

        try:
            items = []
            for item in os.listdir(full_path):
                item_path = os.path.join(full_path, item)
                if os.path.isdir(item_path):
                    items.append(f"{item}/")
                else:
                    items.append(item)
            return sorted(items)
        except (OSError, PermissionError):
            return []

    def open(self, file_path: str, max_size: int = 10 * 1024) -> str:
        if file_path.startswith("./"):
            file_path = file_path[2:]

        full_path = os.path.join(self.base_path, file_path)

        if not os.path.exists(full_path) or os.path.isdir(full_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if os.path.getsize(full_path) > max_size:
            raise ValueError(f"File too large: {file_path} (max {max_size} bytes)")

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            raise ValueError(f"Cannot read binary file: {file_path}")


class Task(BaseModel):
    id: str
    goal: str
    environment: Environment

    @computed_field
    @property
    def realism_score(self) -> int:
        parts = self.id.split("_")
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                return 0
        return 0

    class Config:
        arbitrary_types_allowed = True


class TaskGroup(BaseModel):
    scenario: str
    tasks: List[Task]

    class Config:
        arbitrary_types_allowed = True

    def filter_by_realism_score(self, score: int) -> "TaskGroup":
        filtered_tasks = [task for task in self.tasks if task.realism_score == score]
        return TaskGroup(scenario=self.scenario, tasks=filtered_tasks)

    @classmethod
    def get(cls, scenario_id: str, base_dir: str = "data/test/agentic") -> "TaskGroup":
        scenario_path = os.path.join(base_dir, scenario_id)
        if not os.path.exists(scenario_path) or not os.path.isdir(scenario_path):
            return cls(scenario=scenario_id, tasks=[])

        tasks = []
        for task_id in os.listdir(scenario_path):
            task_path = os.path.join(scenario_path, task_id)
            if not os.path.isdir(task_path):
                continue

            goal_file = os.path.join(task_path, "goal.txt")
            env_dir = os.path.join(task_path, "env")

            if os.path.exists(goal_file) and os.path.exists(env_dir):
                try:
                    with open(goal_file, "r", encoding="utf-8") as f:
                        goal = f.read().strip()

                    environment = Environment(env_dir)
                    task = Task(goal=goal, environment=environment, id=task_id)
                    tasks.append(task)
                except (OSError, UnicodeDecodeError):
                    continue

        return cls(scenario=scenario_id, tasks=tasks)


def get_all_files(
    environment: Environment, max_size: int = 10 * 1024
) -> List[FileSpec]:
    files = []

    def _walk_directory(current_path: str, relative_path: str = ""):
        try:
            for item in os.listdir(current_path):
                item_path = os.path.join(current_path, item)
                relative_item_path = (
                    os.path.join(relative_path, item) if relative_path else item
                )

                if os.path.isdir(item_path):
                    _walk_directory(item_path, relative_item_path)
                else:
                    try:
                        if os.path.getsize(item_path) <= max_size:
                            content = environment.open(relative_item_path, max_size)
                            files.append(
                                FileSpec(filename=relative_item_path, contents=content)
                            )
                    except (ValueError, FileNotFoundError, UnicodeDecodeError):
                        continue
        except (OSError, PermissionError):
            pass

    _walk_directory(environment.base_path)
    return files


def load_tasks_from_directory(base_dir: str = "data/test/agentic") -> List[TaskGroup]:
    if not os.path.exists(base_dir):
        return []

    scenarios = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            scenarios.append(item)

    scenarios = sorted(scenarios)
    return [TaskGroup.get(scenario, base_dir) for scenario in scenarios]
