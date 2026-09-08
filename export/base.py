from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class EpisodeExporter(ABC):
    @abstractmethod
    def append_episode(
        self,
        data_dict: dict,
        *,
        episode_idx: int,
        episode_path: Path | None = None,
    ) -> None:
        """Append one episode. Skipped episodes should not call this."""

    @abstractmethod
    def finalize(self) -> None:
        """Flush buffers and write metadata."""
