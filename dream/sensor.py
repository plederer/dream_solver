from __future__ import annotations

import abc
import dataclasses
import numpy as np
import pandas as pd

from typing import *
from ngsolve import *
from .utils import DreAmLogger

if TYPE_CHECKING:
    from .formulations import _Formulation
    from .solver import CompressibleHDGSolver

logger = DreAmLogger.get_logger('Sensor')


class ScalarComponent(NamedTuple):
    VALUE: bool = True

    @property
    def label(self) -> list[str]:
        return ['']

    def __bool__(self) -> bool:
        return self.VALUE

    def __len__(self) -> int:
        return 1


class VectorComponents(NamedTuple):
    X: bool = False
    Y: bool = False
    Z: bool = False
    MAG: bool = False

    @property
    def indices(self) -> list[int]:
        return [index for index, flag in enumerate([self.X, self.Y, self.Z]) if flag]

    @property
    def label(self) -> list[str]:
        return [label for label, flag in zip(['X', 'Y', 'Z', 'MAG'], self) if flag]

    @classmethod
    def from_pattern(cls, components: Sequence[str]):
        flags = {'X': False, 'Y': False, 'Z': False, 'MAG': False}

        if isinstance(components, str):
            components = components.split("|")

        for component in components:
            component = str(component)
            try:
                if not flags[component.upper()]:
                    flags[component.upper()] = True
            except KeyError:
                logger.warning(f"Component {component} is not valid! Possible alternatives: {list(flags.keys())}")

        return cls(**flags)

    def __bool__(self):
        return self.X | self.Y | self.Z | self.MAG

    def __len__(self):
        return len(self.label)


@dataclasses.dataclass
class Sample:
    name: str
    func: Callable[[Formulation]]
    components: ScalarComponent = ScalarComponent()
    data: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)

    def take(self, key: Optional[float] = None) -> None:
        if key is None:
            key = len(self.data)
        self.data[key] = self.func()

    def __call__(self, time: Optional[float] = None) -> None:
        self.take(time)

    def __str__(self) -> str:
        return self.name


class Sensor(abc.ABC):

    def __init__(self, location: list[Any], name: Optional[str] = "Sensor") -> None:

        self.name = name
        self._location = location
        self._sample_dictionary: dict[str, Sample] = {}
        self._formulation = None

    @property
    def formulation(self) -> _Formulation:
        if self._formulation is None:
            raise ValueError("Assign Solver to Sensor!")
        return self._formulation

    @abc.abstractmethod
    def _evaluate_scalar_CF(self, cf: CF): ...

    def take_single_sample(self, time: Optional[float] = None):

        for sample in self._sample_dictionary.values():
            sample.take(time)

    def assign_solver(self, solver: CompressibleHDGSolver):
        self._formulation = solver.formulation

    def sample_density(self, name: str = 'density'):

        def calculate_density():
            return self._evaluate_scalar_CF(self.formulation.density())

        sample = Sample(name, calculate_density)
        self._sample_dictionary[name] = sample

    def sample_momentum(self, components: Sequence = "x|y", name: str = 'momentum'):

        components = VectorComponents.from_pattern(components)

        if components:
            def calculate_momentum():
                return self._evaluate_vector_CF(self.formulation.momentum(), components)

            sample = Sample(name, calculate_momentum, components)
            self._sample_dictionary[name] = sample
        else:
            logger.warning(f"Can not sample '{name}', since no components have been selected!")

    def sample_velocity(self, components: Sequence = "x|y", name: str = 'velocity'):

        components = VectorComponents.from_pattern(components)

        if components:
            def calculate_velocity():
                return self._evaluate_vector_CF(self.formulation.velocity(), components)

            sample = Sample(name, calculate_velocity, components)
            self._sample_dictionary[name] = sample
        else:
            logger.warning(f"Can not sample '{name}', since no components have been selected!")

    def sample_energy(self, name: str = 'energy'):

        def calculate_energy():
            return self._evaluate_scalar_CF(self.formulation.energy())

        sample = Sample(name, calculate_energy)
        self._sample_dictionary[name] = sample

    def sample_pressure(self, name: str = 'pressure'):

        def calculate_pressure():
            return self._evaluate_scalar_CF(self.formulation.pressure())

        sample = Sample(name, calculate_pressure)
        self._sample_dictionary[name] = sample

    def sample_acoustic_pressure(self, mean_pressure: float, name: str = 'acoustic_pressure'):

        def calculate_acoustic_pressure():
            acoustic_pressure = self.formulation.pressure() - mean_pressure
            return self._evaluate_scalar_CF(acoustic_pressure)

        sample = Sample(name, calculate_acoustic_pressure)
        self._sample_dictionary[name] = sample

    def sample_acoustic_density(self, mean_density: float, name: str = 'acoustic_density'):

        def calculate_acoustic_density():
            acoustic_density = self.formulation.density() - mean_density
            return self._evaluate_scalar_CF(acoustic_density)

        sample = Sample(name, calculate_acoustic_density)
        self._sample_dictionary[name] = sample

    def sample_particle_velocity(self,
                                 mean_velocity: tuple[float, ...],
                                 components: str = 'x|y',
                                 name: str = 'particle_velocity'):

        components = VectorComponents.from_pattern(components)

        if components:
            def calculate_particle_velocity():
                particle_velocity = self.formulation.velocity() - CF(mean_velocity)
                return self._evaluate_vector_CF(particle_velocity, components)

            sample = Sample(name, calculate_particle_velocity, components)
            self._sample_dictionary[name] = sample
        else:
            logger.warning(f"Can not sample '{name}', since no components have been selected!")

    def sample_pressure_coefficient(self,
                                    reference_pressure: float,
                                    reference_velocity: float,
                                    reference_density: float,
                                    name: str = 'pressure_coefficient'):

        def calculate_pressure_coefficient():
            pressure = self.formulation.pressure()
            cp = (pressure - reference_pressure)/(reference_density * reference_velocity**2 / 2)
            return self._evaluate_scalar_CF(cp)

        sample = Sample(name, calculate_pressure_coefficient)
        self._sample_dictionary[name] = sample

    def sample_voritcity(self, name: str = 'vorticity'):

        def calculate_vorticity():
            return self._evaluate_scalar_CF(self.formulation.vorticity())

        sample = Sample(name, calculate_vorticity)
        self._sample_dictionary[name] = sample

    def _evaluate_vector_CF(self, cf: CF, components: VectorComponents):
        values = self._evaluate_scalar_CF(cf)
        output = values[:, components.indices]
        if components.MAG:
            magnitude = np.sqrt(np.sum(np.square(values), axis=1))
            output = np.column_stack([output, magnitude])
        return output.T.flatten()

    def to_dataframe(self, name: str) -> pd.DataFrame:
        sample = self._sample_dictionary[name]
        array = np.array(list(sample.data.values()))
        index = np.array(list(sample.data.keys()))
        columns = [(sample.name, dir, loc) for dir in sample.components.label for loc in self._location]
        columns = pd.MultiIndex.from_tuples(columns, names=["quantity", "component", "location"])
        df = pd.DataFrame(array, columns=columns, index=index)
        df.sort_index(level=0, axis=1, inplace=True)
        return df

    def to_dataframe_all(self) -> pd.DataFrame:
        df = [self.to_dataframe(name) for name in self._sample_dictionary]
        df = pd.concat(df, axis=1)
        df.sort_index(level=0, axis=1, inplace=True)
        return df


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
        super().__init__(points, name)

    def _evaluate_scalar_CF(self, cf: CF) -> np.ndarray:
        mesh = self.formulation.mesh
        return np.array([cf(mesh(*vertex)) for vertex in self._location])


class BoundarySensor(Sensor):

    def __init__(self,
                 boundaries: str,
                 bonus_integration_order: int = 5,
                 name: Optional[str] = "Sensor") -> None:

        if isinstance(boundaries, str):
            boundaries = boundaries.split("|")
        super().__init__(boundaries, name)
        self.bonus_integration_order = bonus_integration_order

    def sample_forces(self, components: Sequence = 'x|y', scale=1, name: str = 'forces'):

        components = VectorComponents.from_pattern(components)

        def calculate_forces():

            mu = self.formulation.cfg.dynamic_viscosity

            stress = -self.formulation.pressure() * Id(self.formulation.mesh.dim)
            if not mu.is_inviscid:
                stress += self.formulation.deviatoric_stress_tensor()

            normal_stress = scale * stress * self.formulation.normal
            return self._evaluate_vector_CF(normal_stress, components)

        sample = Sample(name, calculate_forces, components)
        self._sample_dictionary[name] = sample

    def sample_drag_coefficient(self,
                                reference_density: float,
                                reference_velocity: float,
                                reference_area: float,
                                drag_direction: tuple[float, ...],
                                scale: int = 1,
                                name: str = 'drag_coefficient'):

        def calculate_drag_coefficient():

            mu = self.formulation.cfg.dynamic_viscosity
            stress = -self.formulation.pressure() * Id(self.formulation.mesh.dim)
            if not mu.is_inviscid:
                stress += self.formulation.deviatoric_stress_tensor()

            normal_stress = scale * stress * self.formulation.normal

            drag_coefficient = InnerProduct(normal_stress, drag_direction)
            drag_coefficient *= 2/(reference_density * reference_velocity**2 * reference_area)

            return self._evaluate_scalar_CF(drag_coefficient)

        sample = Sample(name, calculate_drag_coefficient)
        self._sample_dictionary[name] = sample

    def sample_lift_coefficient(self,
                                reference_density: float,
                                reference_velocity: float,
                                reference_area: float,
                                lift_direction: tuple[float, ...],
                                scale: int = 1,
                                name: str = 'lift_coefficient'):

        def calculate_lift_coefficient():

            mu = self.formulation.cfg.dynamic_viscosity

            stress = -self.formulation.pressure() * Id(self.formulation.mesh.dim)
            if not mu.is_inviscid:
                stress += self.formulation.deviatoric_stress_tensor()

            normal_stress = scale * stress * self.formulation.normal

            lift_coefficient = InnerProduct(normal_stress, lift_direction)
            lift_coefficient *= 2/(reference_density * reference_velocity**2 * reference_area)

            return self._evaluate_scalar_CF(lift_coefficient)

        sample = Sample(name, calculate_lift_coefficient)
        self._sample_dictionary[name] = sample

    def _evaluate_scalar_CF(self, cf: CF) -> np.ndarray:
        mesh = self.formulation.mesh
        order = self.formulation.cfg.order + self.bonus_integration_order

        cf = BoundaryFromVolumeCF(cf)

        regions = tuple(mesh.Boundaries(boundary) for boundary in self._location)
        return np.array([Integrate(cf, mesh, definedon=region, order=order) for region in regions])
