from __future__ import annotations
from pathlib import Path

from .formatter import Formatter


class Stem:

    __slots__ = ("name")

    def __set_name__(self, owner: ResultsDirectoryTree, name: str):
        self.name = f"_{name}"

    def __get__(self, tree: ResultsDirectoryTree, objtype: ResultsDirectoryTree) -> Path:
        return getattr(tree, self.name)

    def __set__(self, tree: ResultsDirectoryTree, path: str | Path):
        if path is None:
            path = Path.cwd()

        elif isinstance(path, (str, Path)):

            path = Path(path)
            if not path.is_absolute():
                raise ValueError(f"{self.name[1:].capitalize()} requires absolute path!")

            if not path.exists():
                raise ValueError(f"{self.name[1:].capitalize()} '{path}' does not exist!")

        else:
            raise ValueError(f"Type {type(path)} not supported!")

        setattr(tree, self.name, path)


class Leaf(Stem):

    __slots__ = ("node")

    def __init__(self, node: str | Leaf) -> None:
        self.node = node

    def __set__(self, tree: ResultsDirectoryTree, path: str | Path):
        path = Path(path)

        if not path.is_absolute():

            if isinstance(self.node, Stem):
                self.node = self.node.name[1:]

            path = getattr(tree, self.node).joinpath(path)

        setattr(tree, self.name, path)


class ResultsDirectoryTree:

    parent_folder_path: Path = Stem()
    main_folder_path: Path = Leaf(parent_folder_path)
    state_folder_path: Path = Leaf(main_folder_path)
    sensor_folder_path: Path = Leaf(main_folder_path)
    log_file_path: Path = Leaf(main_folder_path)
    vtk_folder_path: Path = Leaf(main_folder_path)

    def __init__(self, name: str = "results", parent_dir: Path = None) -> None:
        self.parent_folder_path = parent_dir
        self.main_folder_path = name

        self.state_folder_path = "states"
        self.sensor_folder_path = "sensor"
        self.vtk_folder_path = "vtk"
        self.log_file_path = "log.txt"

    def find_directory_paths(self, pattern: str = "") -> tuple[Path]:
        return tuple(dir for dir in self.parent_folder_path.glob(pattern + "*") if dir.is_dir())

    def find_directory_names(self, pattern: str = "") -> tuple[str]:
        return tuple(dir.name for dir in self.parent_folder_path.glob(pattern + "*") if dir.is_dir())

    def items(self):
        for key, node in vars(self).items():
            yield key[1:], node

    def update(self, tree: ResultsDirectoryTree):
        for key, bc in tree.items():
            setattr(self, key, bc)

    def __repr__(self) -> str:
        formatter = Formatter()
        formatter.COLUMN_RATIO = (0.2, 0.8)
        formatter.subheader("Results Directory Tree").newline()
        for key, node in self.items():
            formatter.entry(key.capitalize(), str(node))

        return formatter.output
