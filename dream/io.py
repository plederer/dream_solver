from __future__ import annotations
import pickle

from datetime import datetime
from pathlib import Path
from .utils.formatter import Formatter

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from .hdg_solver import CompressibleHDGSolver
    from .configuration import SolverConfiguration
    from ngsolve import Mesh


class ResultsFolderTree:

    def __init__(self,
                 results_directory_name: str = "results",
                 base_path: Optional[Path] = None) -> None:

        self.results_directory_name = results_directory_name
        self.base_path = base_path

        self._solutions_directory_name = "solutions"
        self._forces_directory_name = "forces"
        self.assign_solver()

    def assign_solver(self, solver: Optional[CompressibleHDGSolver] = None):
        self._solver = solver

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


# class Sensor(ResultsFolderTree):

#     def __init__(self,
#                  results_directory_name: str = "results",
#                  base_path: Optional[Path] = None) -> None:
#         super().__init__(results_directory_name, base_path)

#     @property
#     def solver(self) -> CompressibleHDGSolver:
#         if self._solver is None:
#             raise Exception("Assign a solver to the Sensor")
#         return self._solver

#     @property
#     def results_path(self):
#         results_path = self.base_path.joinpath(self.results_directory_name)
#         if not results_path.exists():
#             results_path.mkdir()
#         return results_path

#     @property
#     def forces_path(self):
#         forces_path = self.results_path.joinpath(self._forces_directory_name)
#         if not forces_path.exists():
#             forces_path.mkdir()
#         return forces_path

#     def save_forces(self, boundary, time: float, name: str = "force_file", suffix=".csv", scale=1, mode='a'):
#         file = self.forces_path.joinpath(name + suffix)

#         header = True
#         if file.exists() and mode != 'w':
#             header = False

#         forces = self.solver.calculate_forces(boundary)
#         forces = {dir: value for dir, value in zip(['x', 'y', 'z'], forces)}
#         time = pd.Index([time], name='t')

#         df = pd.DataFrame(forces, index=time)
#         df.to_csv(file, header=header, mode=mode)

#     def calculate_forces(self, boundary, scale=1):

#         region = self.mesh.Boundaries(boundary)
#         n = self.formulation.normal

#         components = self.formulation.get_gridfunction_components(self.gfu)

#         stress = -self.formulation.pressure(components.PRIMAL) * Id(self.mesh.dim)
#         if self.solver_configuration.dynamic_viscosity is not DynamicViscosity.INVISCID:
#             stress += self.formulation.deviatoric_stress_tensor(components.PRIMAL, components.MIXED)

#         stress = BoundaryFromVolumeCF(stress)
#         forces = Integrate(stress * n, self.mesh, definedon=region)

#         return scale * forces

#     def calculate_pressure(self, boundary=None, point=None):
#         region = self.mesh.Boundaries(boundary)

#         components = self.formulation.get_gridfunction_components(self.gfu)

#         pressure = self.formulation.pressure(components.PRIMAL)

#         vertices = set(self.mesh[v].point for e in region.Elements() for v in e.vertices)
#         for vertex in vertices:
#             # print(pressure(self.mesh(*vertex)))
#             ...

#     # def add_sensor(self, sensor: )
class SolverLoader(ResultsFolderTree):

    @property
    def solver(self) -> Optional[CompressibleHDGSolver]:
        return self._solver

    @property
    def results_path(self):
        results_path = self.base_path.joinpath(self.results_directory_name)
        if not results_path.exists():
            raise Exception(f"Can not load from {results_path}, as the path does not exist.")
        return results_path

    @property
    def solutions_path(self):
        solutions_path = self.results_path.joinpath(self._solutions_directory_name)
        if not solutions_path.exists():
            raise Exception(f"Can not load from {solutions_path}, as the path does not exist.")
        return solutions_path

    def load_mesh(self, name: str = "mesh", suffix: str = ".pickle") -> Mesh:
        file = self.results_path.joinpath(name + suffix)
        with file.open("rb") as openfile:
            mesh = pickle.load(openfile)

        if self.solver is not None:
            self.solver.reset_mesh(mesh)

        return mesh

    def load_configuration(self, name: str = "config", suffix: str = ".pickle") -> SolverConfiguration:
        file = self.results_path.joinpath(name + suffix)
        with file.open("rb") as openfile:
            config = pickle.load(openfile)

        if self.solver is not None:
            self.solver.reset_configuration(config)

        return config

    def load_state(self, name: str):

        if self.solver is None:
            raise Exception("Assign solver before loading state")

        file = self.solutions_path.joinpath(name)
        self.solver.gfu.Load(str(file))


class SolverSaver(ResultsFolderTree):

    @property
    def solver(self) -> CompressibleHDGSolver:
        if self._solver is None:
            raise Exception("Assign a solver to the Saver")
        return self._solver

    @property
    def results_path(self):
        results_path = self.base_path.joinpath(self.results_directory_name)
        if not results_path.exists():
            results_path.mkdir()
        return results_path

    @property
    def solutions_path(self):
        solutions_path = self.results_path.joinpath(self._solutions_directory_name)
        if not solutions_path.exists():
            solutions_path.mkdir()
        return solutions_path

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
        formatter.COLUMN_RATIO = (0.3, 0.7)

        formatter.header("Compressible HDG Solver").newline()
        formatter.entry("Authors", "Philip Lederer, Jan Ellmenreich")
        formatter.entry("Institute", "Analysis and Scientific Computing (2022 - )")
        formatter.entry("University", "TU Wien")
        formatter.entry("Funding", "FWF (Austrian Science Fund) - P35391N")
        formatter.entry("Github", "https://github.com/plederer/dream_solver")
        formatter.entry("Date", datetime.now().strftime('%Hh:%Mm:%Ss %d/%m/%Y')).newline()

        formatter.add(self.solver.solver_configuration)

        if comment is not None:
            formatter.header("Comment").newline()
            formatter.text(comment)

        file = self.results_path.joinpath(name + ".txt")

        with file.open("w") as openfile:
            openfile.write(formatter.output)

    def _save_pickle_configuration(self, name: str, suffix: str = ".pickle"):
        file = self.results_path.joinpath(name + suffix)
        with file.open("wb") as openfile:
            pickle.dump(self.solver.solver_configuration, openfile)
