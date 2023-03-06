from __future__ import annotations

import abc
import dataclasses
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Optional, Callable

from ngsolve import *
from .viscosity import DynamicViscosity

if TYPE_CHECKING:
    from .hdg_solver import CompressibleHDGSolver


@dataclasses.dataclass
class Sample:
    name: str
    func: Callable[[CompressibleHDGSolver]]
    values: list = dataclasses.field(default_factory=list)
    components: tuple[str, ...] = ("",)

    def take(self, solver: CompressibleHDGSolver) -> None:
        self.values.append(self.func(solver))

    def __call__(self, solver: CompressibleHDGSolver) -> None:
        self.take(solver)

    def __str__(self) -> str:
        return self.name


class Sensor(abc.ABC):

    def __init__(self, name: Optional[str] = "Sensor") -> None:

        self.name = name
        self._sample_dictionary: dict[str, Sample] = {}
        self._solver = None

    @property
    def solver(self) -> CompressibleHDGSolver:
        if self._solver is None:
            raise ValueError("Assign Solver to Sensor!")
        return self._solver

    def take_single_sample(self):

        for sample in self._sample_dictionary.values():
            sample.take(self.solver)

    def assign_solver(self, solver: CompressibleHDGSolver):
        self._solver = solver

    def sample_density(self, name: str = 'density'):

        def calculate_density(solver: CompressibleHDGSolver):
            components = solver.formulation.get_gridfunction_components(solver.gfu)
            density = solver.formulation.density(components.PRIMAL)

            return self._evaluate_CF(density, solver)

        sample = Sample(name, calculate_density)
        self._sample_dictionary[name] = sample

    def sample_momentum(self, name: str = 'momentum'):

        def calculate_momentum(solver: CompressibleHDGSolver):
            components = solver.formulation.get_gridfunction_components(solver.gfu)
            momentum = solver.formulation.momentum(components.PRIMAL)

            return self._evaluate_CF(momentum, solver)

        sample = Sample(name, calculate_momentum, components=('x', 'y', 'z'))
        self._sample_dictionary[name] = sample

    def sample_energy(self, name: str = 'energy'):

        def calculate_energy(solver: CompressibleHDGSolver):
            components = solver.formulation.get_gridfunction_components(solver.gfu)
            energy = solver.formulation.energy(components.PRIMAL)

            return self._evaluate_CF(energy, solver)

        sample = Sample(name, calculate_energy)
        self._sample_dictionary[name] = sample

    def sample_pressure(self, name: str = 'pressure'):

        def calculate_pressure(solver: CompressibleHDGSolver):
            components = solver.formulation.get_gridfunction_components(solver.gfu)
            pressure = solver.formulation.pressure(components.PRIMAL)

            return self._evaluate_CF(pressure, solver)

        sample = Sample(name, calculate_pressure)
        self._sample_dictionary[name] = sample

    def sample_acoustic_pressure(self, mean_pressure: float, name: str = 'acoustic_pressure'):

        def calculate_acoustic_pressure(solver: CompressibleHDGSolver):
            components = solver.formulation.get_gridfunction_components(solver.gfu)
            pressure = solver.formulation.pressure(components.PRIMAL)

            acoustic_pressure = pressure - mean_pressure

            return self._evaluate_CF(acoustic_pressure, solver)

        sample = Sample(name, calculate_acoustic_pressure)
        self._sample_dictionary[name] = sample

    def sample_pressure_coefficient(self,
                                    reference_pressure: float,
                                    reference_velocity: float,
                                    reference_density: float,
                                    name: str = 'pressure_coefficient'):

        def calculate_pressure_coefficient(solver: CompressibleHDGSolver):
            components = solver.formulation.get_gridfunction_components(solver.gfu)
            pressure = solver.formulation.pressure(components.PRIMAL)

            cp = (pressure - reference_pressure)/(reference_density * reference_velocity**2 / 2)
            return self._evaluate_CF(cp, solver)

        sample = Sample(name, calculate_pressure_coefficient)
        self._sample_dictionary[name] = sample

    @abc.abstractmethod
    def _evaluate_CF(self, cf: CF, solver: CompressibleHDGSolver): ...

    @abc.abstractmethod
    def convert_samples_to_dataframe(self, time_index=None) -> pd.DataFrame: ...


class PointSensor(Sensor):

    @classmethod
    def from_boundary(cls,
                      boundaries: str,
                      mesh: Mesh,
                      name: Optional[str] = "Sensor") -> PointSensor:

        if isinstance(boundaries, str):
            boundaries = boundaries.split("|")

        regions = tuple(mesh.Boundaries(boundary) for boundary in boundaries)
        points = tuple(set(mesh[v].point for region in regions for e in region.Elements() for v in e.vertices))

        return cls(points, name)

    def __init__(self,
                 points: list[tuple[float, ...]],
                 name: Optional[str] = "Sensor") -> None:
        self.points = points
        super().__init__(name)

    def convert_samples_to_dataframe(self, time_index=None) -> pd.DataFrame:

        numpy_2d_arrays = []
        columns = []
        for sample in self._sample_dictionary.values():
            array = np.atleast_3d(sample.values)
            n_samples, n_loc, n_dir = array.shape
            numpy_2d_arrays.append(array.reshape((n_samples, n_loc * n_dir)))

            columns += [(sample.name, loc, dir) for loc in self.points for dir in sample.components[:n_dir]]

        columns = pd.MultiIndex.from_tuples(columns, names=["quantity", "point", "dir"])
        df = pd.DataFrame(np.hstack(numpy_2d_arrays), columns=columns)

        if time_index is not None:
            index = pd.Index(time_index, name="time")
            df.set_index(index, inplace=True)

        df.sort_index(level=0, axis=1, inplace=True)

        return df

    def _evaluate_CF(self, cf: CF, solver: CompressibleHDGSolver):
        return tuple(cf(solver.mesh(*vertex)) for vertex in self.points)

    @staticmethod
    def vertices_from_boundaries(boundaries: str, mesh: Mesh):
        if isinstance(boundaries, str):
            boundaries = boundaries.split("|")
        regions = tuple(mesh.Boundaries(boundary) for boundary in boundaries)
        return tuple(set(mesh[v].point for region in regions for e in region.Elements() for v in e.vertices))


class BoundarySensor(Sensor):

    def __init__(self,
                 boundaries: str,
                 name: Optional[str] = "Sensor") -> None:

        if isinstance(boundaries, str):
            boundaries = boundaries.split("|")
        self.boundaries = boundaries

        super().__init__(name)

    def sample_forces(self, scale=1, name: str = 'forces'):

        def calculate_forces(solver: CompressibleHDGSolver):

            dynamic_viscosity = solver.solver_configuration.dynamic_viscosity
            components = solver.formulation.get_gridfunction_components(solver.gfu)
            stress = -solver.formulation.pressure(components.PRIMAL)
            if dynamic_viscosity is not DynamicViscosity.INVISCID:
                stress += solver.formulation.deviatoric_stress_tensor(components.PRIMAL, components.MIXED)

            normal_stress = scale * stress * solver.formulation.normal
            return self._evaluate_CF(normal_stress, solver)

        sample = Sample(name, calculate_forces, components=('x', 'y', 'z'))
        self._sample_dictionary[name] = sample

    def _evaluate_CF(self, cf: CF, solver: CompressibleHDGSolver):
        order = solver.solver_configuration.order + 5
        cf = BoundaryFromVolumeCF(cf)

        regions = tuple(solver.mesh.Boundaries(boundary) for boundary in self.boundaries)
        return tuple(Integrate(cf, solver.mesh, definedon=region, order=order) for region in regions)

    def convert_samples_to_dataframe(self, time_index=None) -> pd.DataFrame:

        numpy_2d_arrays = []
        columns = []
        for sample in self._sample_dictionary.values():

            array = np.atleast_3d(sample.values)
            n_samples, n_loc, n_dir = array.shape
            numpy_2d_arrays.append(array.reshape((n_samples, n_loc * n_dir)))

            columns += [(sample.name, loc, dir) for loc in self.boundaries for dir in sample.components[:n_dir]]

        columns = pd.MultiIndex.from_tuples(columns, names=["quantity", "boundary", "dir"])
        df = pd.DataFrame(np.hstack(numpy_2d_arrays), columns=columns)

        if time_index is not None:
            index = pd.Index(time_index, name="time")
            df.set_index(index, inplace=True)

        df.sort_index(level=0, axis=1, inplace=True)

        return df
