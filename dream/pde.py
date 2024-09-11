from __future__ import annotations
import typing
import logging

import ngsolve as ngs

from dream.mesh import DreamMesh
from dream.config import DescriptorConfiguration

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration


class Formulation(DescriptorConfiguration, is_interface=True):

    def set_configuration_and_mesh(self, cfg: SolverConfiguration, mesh: ngs.Mesh | DreamMesh) -> None:

        if isinstance(mesh, ngs.Mesh):
            mesh = DreamMesh(mesh)

        self._cfg = cfg
        self._mesh = mesh

        self._is_linear = True

    def set_finite_element_spaces(self, spaces: dict[str, ngs.FESpace]) -> None:

        if not spaces:
            raise ValueError("Spaces container is empty!")

        fes = list(spaces.values())

        for space in fes[1:]:
            fes[0] *= space

        self.fes = fes[0]
        self.spaces = spaces

    def set_trial_and_test_functions(self) -> None:
        self.TnT = {label: space.TnT() for label, space in self.spaces.items()}

    def set_gridfunctions(self) -> None:
        self.gfu = ngs.GridFunction(self.fes)

        self.gfus = {label: self.gfu for label in self.spaces.keys()}
        for label, gfu in zip(self.spaces.keys(), self.gfu.components):
            self.gfus[label] = gfu

    def set_transient_gridfunctions(self, gfus: dict[str, dict[str, ngs.GridFunction]] = None) -> None:

        if gfus is None:
            gfus = {}

        self.gfus_transient = gfus

    @property
    def dmesh(self) -> DreamMesh:
        return self._mesh

    @property
    def mesh(self) -> ngs.Mesh:
        return self.dmesh.ngsmesh

    @property
    def cfg(self) -> SolverConfiguration:
        return self._cfg

    @property
    def is_linear(self) -> bool:
        return self._is_linear

    def get_system(self, blf: list[ngs.comp.SumOfIntegrals], lf: list[ngs.comp.SumOfIntegrals]):
        raise NotImplementedError()


class PDEConfiguration(DescriptorConfiguration, is_interface=True):

    bcs: BoundaryConditions
    dcs: DomainConditions

    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)

    def set_mesh(self, mesh: ngs.Mesh):
        self.mesh = mesh
        self.bcs = type(self).bcs(mesh.GetBoundaries())

    @property
    def formulation(self) -> Formulation:
        raise NotImplementedError("Overload this property in derived class!")
