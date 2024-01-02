from __future__ import annotations
from typing import Any

from dream.config import BaseConfig

from .tree import ResultsDirectoryTree


class IOConfig(BaseConfig):

    def __init__(self) -> None:
        self._tree = ResultsDirectoryTree()
        self._save_state = False
        self._info = {}

    @property
    def tree(self) -> ResultsDirectoryTree:
        return self._tree

    @tree.setter
    def tree(self, tree: ResultsDirectoryTree):
        if not isinstance(tree, ResultsDirectoryTree):
            raise TypeError(f"Tree has to be of type '{ResultsDirectoryTree}'")
        self.tree.update(tree)

    @property
    def save_state(self) -> bool:
        return self._save_state

    @save_state.setter
    def save_state(self, value):
        self._save_state = bool(value)

    @property
    def info(self) -> dict[str, Any]:
        """ Info returns a dictionary reserved for the storage of user defined parameters. """
        return self._info

    @info.setter
    def info(self, info: dict[str, Any]):
        self._info.update(info)

    def format(self):
        formatter = self.formatter.new()
        formatter.subheader('IO Configuration').newline()
        formatter.entry('Save State', str(self._save_state))
        formatter.newline()

        if self.info:
            formatter.subheader('Simulation Info').newline()

            for key, value in self.info.items():
                if hasattr(value, '__str__'):
                    value = str(value)
                elif hasattr(value, '__repr__'):
                    value = repr(value)
                else:
                    value = f"Type: {type(value)} not displayable"

                formatter.entry(str(key), value)
        formatter.newline()

        formatter.output += repr(self.tree)
