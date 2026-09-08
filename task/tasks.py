from __future__ import annotations
from datetime import datetime
from pathlib import Path

from termcolor import colored


def _build_run_name(task_name: str, now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"{task_name}_{now:%m%d}_{now:%H%M}"


def _list_episode_names(subtask_dir: Path) -> list[str]:
    if not subtask_dir.is_dir():
        raise ValueError(f"subtask dir not found: {subtask_dir}")
    return sorted(
        entry.name
        for entry in subtask_dir.iterdir()
        if entry.is_dir() or entry.exists()
    )


class Task:
    def __init__(self, io_cfg: dict, run_name: str | None = None):
        self.input_base_path = Path(io_cfg["input_base_path"])
        self.task_name = io_cfg["task"]
        self.requested_subtasks = list(io_cfg["subtasks"])
        self.output_base_path = Path(io_cfg["output_base_path"])
        if "max_episode_length" in io_cfg:
            self.max_episode_length = int(io_cfg["max_episode_length"])
        elif "episode_length" in io_cfg:
            self.max_episode_length = int(io_cfg["episode_length"])
        else:
            raise KeyError("io config requires max_episode_length")
        self.verbose = bool(io_cfg["verbose"])

        self.data_dir = self.input_base_path / self.task_name
        self.selected_subtasks = self._resolve_subtasks()
        self.subtask_info, self.episode_list = self._build_episode_list()
        if not self.episode_list:
            raise ValueError(
                f"No episodes found. task={self.task_name}, "
                f"selected_subtasks={self.selected_subtasks}"
            )

        self.run_name = run_name if run_name is not None else _build_run_name(self.task_name)
        self.output_path = self.output_base_path / self.task_name / self.run_name
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _resolve_subtasks(self) -> list[str]:
        if not self.data_dir.is_dir():
            raise ValueError(f"task dir not found: {self.data_dir}")

        all_subtasks = sorted(
            entry.name for entry in self.data_dir.iterdir() if entry.is_dir()
        )
        if len(self.requested_subtasks) == 1 and self.requested_subtasks[0].lower() == "all":
            return all_subtasks

        invalid = [s for s in self.requested_subtasks if s not in all_subtasks]
        if invalid:
            raise ValueError(
                f"These subtasks do not exist: {invalid}. "
                f"Available subtasks: {all_subtasks}"
            )
        return self.requested_subtasks

    def _build_episode_list(self) -> tuple[dict[str, int], list[Path]]:
        subtask_info: dict[str, int] = {}
        episode_list: list[Path] = []

        for subtask in self.selected_subtasks:
            subtask_dir = self.data_dir / subtask
            episodes = _list_episode_names(subtask_dir)
            if self.max_episode_length != -1:
                episodes = episodes[: self.max_episode_length]
            subtask_info[subtask] = len(episodes)
            episode_list.extend(subtask_dir / episode for episode in episodes)

        return subtask_info, episode_list

    @property
    def config_path(self) -> Path:
        return self.output_path / "config.yaml"

    @property
    def meta_path(self) -> Path:
        return self.output_path / "meta.json"

    @property
    def replay_buffer_path(self) -> Path:
        return self.output_path / "replay_buffer.zarr"

    def log_summary(self) -> None:
        print(colored("task summary", "green"))
        print("Selected subtasks:", self.selected_subtasks)
        print("Subtask episode stats:", self.subtask_info)
        print("Total episodes:", len(self.episode_list))
        print("Output path:", self.output_path)
        print("Config:", self.config_path)
        print("Meta:", self.meta_path)
        print("Replay buffer:", self.replay_buffer_path)


if __name__ == "__main__":
    io_cfg = {
        "input_base_path": "/mnt/oss_data/arx1",
        "task": "zhibei",
        "subtasks": ["zhibei_0625_200"],
        "output_base_path": "/mnt/workspace/zyj/data/processed_zarr",
        "max_episode_length": -1,
        "verbose": True,
    }
    task = Task(io_cfg)
    task.log_summary()
