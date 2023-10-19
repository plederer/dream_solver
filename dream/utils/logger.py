from __future__ import annotations
import logging
from ngsolve import Parameter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..io import ResultsDirectoryTree


class DreAmLogger:

    _iteration_error_digit: int = 8
    _time_step_digit: int = 1
    _name: str = "DreAm"

    @classmethod
    def get_logger(cls, name: str):
        return logging.getLogger(f"{cls._name}.{name}")

    @classmethod
    def set_time_step_digit(cls, time_step: float):

        if isinstance(time_step, Parameter):
            time_step = time_step.Get()

        if isinstance(time_step, float):
            digit = len(str(time_step).split(".")[1])
            cls._time_step_digit = digit

    def __init__(self, tree: ResultsDirectoryTree, log_to_terminal: bool = False, log_to_file: bool = False) -> None:
        self.logger = logging.getLogger(self._name)
        self.tree = tree

        self.stream_handler = logging.NullHandler()
        self.file_handler = logging.NullHandler()
        self.formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        self.filename = "log.txt"

        self.log_to_terminal = log_to_terminal
        self.log_to_file = log_to_file

    @property
    def filepath(self):
        if not self.tree.main_path.exists():
            self.tree.main_path.mkdir(parents=True)
        return self.tree.main_path.joinpath(self.filename)

    def set_level(self, level):
        self.logger.setLevel(level)

    def silence_logger(self):
        self.log_to_file = False
        self.log_to_terminal = False

    @property
    def log_to_terminal(self):
        return self._log_to_terminal

    @log_to_terminal.setter
    def log_to_terminal(self, value: bool):
        if value:
            self.stream_handler = logging.StreamHandler()
            self.stream_handler.setLevel(self.logger.level)
            self.stream_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.stream_handler)
        else:
            self.logger.removeHandler(self.stream_handler)
            self.stream_handler = logging.NullHandler()
        self._log_to_terminal = value

    @property
    def log_to_file(self):
        return self._log_to_file

    @log_to_file.setter
    def log_to_file(self, value: bool):
        if value:
            self.file_handler = logging.FileHandler(self.filepath, delay=True)
            self.file_handler.setLevel(self.logger.level)
            self.file_handler.setFormatter(self.formatter)
            self.logger.addHandler(self.file_handler)
        else:
            self.logger.removeHandler(self.file_handler)
            self.file_handler = logging.NullHandler()
        self._log_to_file = value
