
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class ResultManager:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.results_root = self.project_root / "results"
        self.results_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def part_name_from_stl(stl_path: str | Path) -> str:
        """
        Use the STL filename without its extension as the part name.
        """
        stem = Path(stl_path).stem.strip()

        safe_name = re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            stem,
        ).strip("._-")

        if not safe_name:
            raise ValueError("Could not derive a valid part name from STL filename.")

        return safe_name

    def part_dir(self, stl_path: str | Path) -> Path:
        part_name = self.part_name_from_stl(stl_path)
        path = self.results_root / part_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stage_dir(self, stl_path: str | Path, stage_number: int) -> Path:
        if stage_number < 1:
            raise ValueError("Stage number must be positive.")

        path = self.part_dir(stl_path) / f"stage{stage_number}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(
        self,
        stl_path: str | Path,
        stage_number: int,
        filename: str,
        data: dict[str, Any],
    ) -> Path:
        path = self.stage_dir(stl_path, stage_number) / filename

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
            )

        return path

    def result_path(
        self,
        stl_path: str | Path,
        stage_number: int,
        filename: str,
    ) -> Path:
        return self.stage_dir(stl_path, stage_number) / filename
