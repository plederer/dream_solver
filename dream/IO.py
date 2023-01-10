from __future__ import annotations
import pickle
import textwrap
from datetime import datetime
from pathlib import Path

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from .hdg_solver import CompressibleHDGSolver


class SolverSaver:

    def __init__(self,
                 solver: CompressibleHDGSolver,
                 directory_name: str = "results",
                 base_path: Optional[Path] = None) -> None:

        self.solver = solver
        self.directory_name = directory_name
        self.base_path = base_path

        self._solutions_directory_name = "solutions"
        self._forces_directory_name = "forces"

    @property
    def base_path(self) -> Path:
        return self._base_path

    @base_path.setter
    def base_path(self, base_path: Path):

        if base_path is None:
            self._base_path = Path.cwd()

        elif isinstance(base_path, (str, Path)):
            base_path = Path(base_path)

            if not base_path.exists():
                raise ValueError(f"Path {base_path} does not exist!")

            self._base_path = base_path

        else:
            raise ValueError(f"Type {type(base_path)} not supported!")

    @property
    def results_path(self):
        results_path = self.base_path.joinpath(self.directory_name)
        if not results_path.exists():
            results_path.mkdir()
        return results_path

    @property
    def solutions_path(self):
        solutions_path = self.results_path.joinpath(self._solutions_directory_name)
        if not solutions_path.exists():
            solutions_path.mkdir()
        return solutions_path

    @property
    def forces_path(self):
        forces_path = self.results_path.joinpath(self._forces_directory_name)
        if not forces_path.exists():
            forces_path.mkdir()
        return forces_path

    def save_mesh(self, name: str = "mesh", suffix: str = ".pickle"):
        file = self.results_path.joinpath(name + suffix)
        with file.open("wb") as openfile:
            pickle.dump(self.solver.mesh, openfile)

    def save_configuration(self,
                           name: str = "config",
                           comment: Optional[str] = None,
                           pickle: bool = True,
                           txt: bool = True):

        if pickle:
            self._save_pickle_configuration(name)
        if txt:
            self._save_txt_configuration(name, comment)

    def save_state(self, name: str = "state"):

        file = self.solutions_path.joinpath(name)
        self.solver.gfu.Save(str(file))

    def _save_txt_configuration(self, name: str, comment: Optional[str] = None):

        formatter = Formatter()
        formatter.column_ratio = (0.3, 0.7)

        formatter.header("Compressible HDG Solver").newline()
        formatter.entry("Authors", "Philip Lederer, Jan Ellmenreich")
        formatter.entry("Institute", "Analysis and Scientific Computing (2022 - )")
        formatter.entry("University", "TU Wien")
        formatter.entry("Funding", "FWF (Austrian Science Fund) - P35391N")
        formatter.entry("Github", "https://github.com/plederer/dream_solver")
        formatter.entry("Date", datetime.now().strftime('%Hh:%Mm:%Ss %d/%m/%Y')).newline()

        formatter.add(self.solver.solver_configuration)

        if comment is not None:
            formatter.header("Comment")
            formatter.text(comment)

        file = self.results_path.joinpath(name + ".txt")

        with file.open("w") as openfile:
            openfile.write(formatter.output)

    def _save_pickle_configuration(self, name: str, suffix: str = ".pickle"):
        file = self.results_path.joinpath(name + suffix)
        with file.open("wb") as openfile:
            pickle.dump(self.solver.solver_configuration, openfile)


class Formatter:

    TERMINAL_WIDTH: int = 80
    TEXT_INDENT = 5
    column_ratio = (0.5, 0.5)

    def __init__(self) -> None:
        self.reset()

    def header(self, text: str):
        text = f" {text.upper()} "
        header = f"{text:─^{self.TERMINAL_WIDTH}}" + "\n"
        self.output += header
        return self

    def subheader(self, text: str):
        text = '─── ' + text.upper() + ' ───'
        subheader = f"{text:^{self.TERMINAL_WIDTH}}" + "\n"
        self.output += subheader
        return self

    def entry(self, text: str, value, equal: str = ":"):
        txt_width, value_width = tuple(int(self.TERMINAL_WIDTH * ratio) for ratio in self.column_ratio)
        entry = f"{text + equal:>{txt_width}} {value:<{value_width}}" + "\n"
        self.output += entry
        return self

    def text(self, text):
        width = self.TERMINAL_WIDTH - 2*self.TEXT_INDENT
        indent = self.TEXT_INDENT * ' '
        text = textwrap.fill(text, width, initial_indent=indent, subsequent_indent=indent)

        self.output += text + "\n"
        return self

    def newline(self):
        self.output += "\n"

    def reset(self):
        self.output = ""

    def add(self, object):
        self.output += repr(object)

    def __repr__(self) -> str:
        return self.output
