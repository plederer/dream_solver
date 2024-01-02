from __future__ import annotations
import logging
from ngsolve import Parameter

from .tree import ResultsDirectoryTree


class DreamLogger(logging.Logger):

    @property
    def log_file_path(self):
        folder = self.tree.log_file_path.parent
        if not folder.exists():
            folder.mkdir(parents=True)
        return self.tree.log_file_path

    def __init__(self, tree: ResultsDirectoryTree = None) -> None:
        super().__init__("Dream", logging.INFO)

        if tree is None:
            tree = ResultsDirectoryTree()

        self.tree = tree

        self.formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

        self.time_step_digit = 1
        self.iteration_error_digit = 8

    def set_time_step_digit(self, time_step: float):

        digit = 1

        if isinstance(time_step, Parameter):
            time_step = time_step.Get()

        if time_step < 1:
            digit = int(f"{time_step:e}"[-2:])

        self.time_step_digit = digit

    def get_formatter(self,
                      fmt: str = "%(name)s - %(levelname)s - %(message)s",
                      datefmt: str = None,
                      style: logging._FormatStyle = "%",
                      validate: bool = True):
        return logging.Formatter(fmt, datefmt, style, validate)

    def get_stream_handler(self, stream=None):
        handler = logging.StreamHandler(stream)
        handler.setFormatter(self.formatter)
        logging.StrFormatStyle
        return handler

    def get_file_handler(self,
                         filename: str = None,
                         mode: str = "w",
                         encoding: str = None,
                         delay: bool = True,
                         errors: str = None):
        if filename is None:
            filename = self.log_file_path

        handler = logging.FileHandler(filename, mode, encoding, delay, errors)
        handler.setFormatter(self.formatter)

        return handler

    def log_to_terminal(self, handler: logging.StreamHandler = None):
        if handler is None:
            handler = self.get_stream_handler()
        self.addHandler(handler)

    def log_to_file(self, handler: logging.FileHandler = None):
        if handler is None:
            handler = self.get_file_handler()
        self.addHandler(handler)

    def activate(self):
        self._disable_loggers(False)

    def deactivate(self):
        self._disable_loggers(True)

    def _disable_loggers(self, disabled: bool):
        for name, logger in self.manager.loggerDict.items():
            if name.startswith(self.name):
                logger.disabled = disabled


dlogger = DreamLogger()
dlogger.manager.loggerDict[dlogger.name] = dlogger
dlogger.log_to_terminal()
