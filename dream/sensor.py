from __future__ import annotations

import abc
import dataclasses
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Optional, Callable

from ngsolve import *
from .viscosity import DynamicViscosity

if TYPE_CHECKING:
    from .formulations import Formulation
    from .hdg_solver import CompressibleHDGSolver


@dataclasses.dataclass
class Sample:
    name: str
    func: Callable[[Formulation]]
    values: list = dataclasses.field(default_factory=list)
    components: tuple[str, ...] = ("",)

    def take(self) -> None:
        self.values.append(self.func())

    def __call__(self) -> None:
        self.take()

    def __str__(self) -> str:
        return self.name


class Sensor(abc.ABC):

    def __init__(self, name: Optional[str] = "Sensor") -> None:

        self.name = name
        self._sample_dictionary: dict[str, Sample] = {}
        self._formulation = None

    @property
    def formulation(self) -> Formulation:
        if self._formulation is None:
            raise ValueError("Assign Solver to Sensor!")
        return self._formulation

    def take_single_sample(self):

        for sample in self._sample_dictionary.values():
            sample.take()

    def assign_solver(self, solver: CompressibleHDGSolver):
        self._formulation = solver.formulation

    def sample_density(self, name: str = 'density'):

        def calculate_density():
            return self._evaluate_CF(self.formulation.density())

        sample = Sample(name, calculate_density)
        self._sample_dictionary[name] = sample

    def sample_momentum(self, name: str = 'momentum'):

        def calculate_momentum():
            return self._evaluate_CF(self.formulation.momentum())

        sample = Sample(name, calculate_momentum, components=('x', 'y', 'z'))
        self._sample_dictionary[name] = sample

    def sample_energy(self, name: str = 'energy'):

        def calculate_energy():
            return self._evaluate_CF(self.formulation.energy())

        sample = Sample(name, calculate_energy)
        self._sample_dictionary[name] = sample

    def sample_pressure(self, name: str = 'pressure'):

        def calculate_pressure():
            return self._evaluate_CF(self.formulation.pressure())

        sample = Sample(name, calculate_pressure)
        self._sample_dictionary[name] = sample

    def sample_acoustic_pressure(self, mean_pressure: float, name: str = 'acoustic_pressure'):

        def calculate_acoustic_pressure():
            acoustic_pressure = self.formulation.pressure() - mean_pressure
            return self._evaluate_CF(acoustic_pressure)

        sample = Sample(name, calculate_acoustic_pressure)
        self._sample_dictionary[name] = sample

    def sample_particle_velocity(self, mean_velocity: tuple[float, ...], name: str = 'particle_velocity'):

        def calculate_particle_velocity():
            particle_velocity = self.formulation.velocity() - CF(mean_velocity)
            return self._evaluate_CF(particle_velocity)

        sample = Sample(name, calculate_particle_velocity, components=('x', 'y', 'z'))
        self._sample_dictionary[name] = sample

    def sample_pressure_coefficient(self,
                                    reference_pressure: float,
                                    reference_velocity: float,
                                    reference_density: float,
                                    name: str = 'pressure_coefficient'):

        def calculate_pressure_coefficient():
            pressure = self.formulation.pressure()
            cp = (pressure - reference_pressure)/(reference_density * reference_velocity**2 / 2)
            return self._evaluate_CF(cp)

        sample = Sample(name, calculate_pressure_coefficient)
        self._sample_dictionary[name] = sample

    @abc.abstractmethod
    def _evaluate_CF(self, cf: CF): ...

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

    def _evaluate_CF(self, cf: CF):
        mesh = self.formulation.mesh
        return tuple(cf(mesh(*vertex)) for vertex in self.points)

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

        def calculate_forces():

            dynamic_viscosity = self.formulation.cfg.dynamic_viscosity
            stress = -self.formulation.pressure() * Id(self.formulation.mesh.dim)
            if dynamic_viscosity is not DynamicViscosity.INVISCID:
                stress += self.formulation.deviatoric_stress_tensor()

            normal_stress = scale * stress * self.formulation.normal
            return self._evaluate_CF(normal_stress)

        sample = Sample(name, calculate_forces, components=('x', 'y', 'z'))
        self._sample_dictionary[name] = sample

    def _evaluate_CF(self, cf: CF):
        mesh = self.formulation.mesh
        order = self.formulation.cfg.order + 5

        cf = BoundaryFromVolumeCF(cf)

        regions = tuple(mesh.Boundaries(boundary) for boundary in self.boundaries)
        return tuple(Integrate(cf, mesh, definedon=region, order=order) for region in regions)

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
