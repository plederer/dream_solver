from __future__ import annotations
from pathlib import Path
from typing import Optional

from ..utils import Formatter

class ResultsDirectoryTree:

    def __init__(self,
                 directory_name: str = "results",
                 state_directory_name: str = "states",
                 sensor_directory_name: str = "sensor",
                 vtk_directory_name: str = "vtk",
                 parent_path: Optional[Path] = None) -> None:

        self.parent_path = parent_path
        self.directory_name = directory_name
        self.state_directory_name = state_directory_name
        self.sensor_directory_name = sensor_directory_name
        self.vtk_directory_name = vtk_directory_name

    @property
    def main_path(self) -> Path:
        return self.parent_path.joinpath(self.directory_name)

    @property
    def state_path(self) -> Path:
        return self.main_path.joinpath(self.state_directory_name)

    @property
    def sensor_path(self) -> Path:
        return self.main_path.joinpath(self.sensor_directory_name)

    @property
    def vtk_path(self) -> Path:
        return self.main_path.joinpath(self.vtk_directory_name)

    @property
    def parent_path(self) -> Path:
        return self._parent_path

    @parent_path.setter
    def parent_path(self, parent_path: Path):

        if parent_path is None:
            self._parent_path = Path.cwd()

        elif isinstance(parent_path, (str, Path)):
            parent_path = Path(parent_path)

            if not parent_path.exists():
                raise ValueError(f"Path {parent_path} does not exist!")

            self._parent_path = parent_path

        else:
            raise ValueError(f"Type {type(parent_path)} not supported!")

    def find_directory_paths(self, pattern: str = "") -> tuple[Path]:
        return tuple(dir for dir in self.parent_path.glob(pattern + "*") if dir.is_dir())

    def find_directory_names(self, pattern: str = "") -> tuple[str]:
        return tuple(dir.name for dir in self.parent_path.glob(pattern + "*") if dir.is_dir())

    def update(self, tree: ResultsDirectoryTree):
        for key, bc in vars(tree).items():
            setattr(self, key, bc)

    def __repr__(self) -> str:
        formatter = Formatter()
        formatter.COLUMN_RATIO = (0.2, 0.8)
        formatter.header("Results Directory Tree").newline()
        formatter.entry("Path", str(self.parent_path))
        formatter.entry("Main", f"{self.parent_path.stem}/{self.directory_name}")
        formatter.entry("State", f"{self.parent_path.stem}/{self.directory_name}/{self.state_directory_name}")
        formatter.entry("Sensor", f"{self.parent_path.stem}/{self.directory_name}/{self.sensor_directory_name}")

        return formatter.output