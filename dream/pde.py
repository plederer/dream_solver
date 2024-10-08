from __future__ import annotations
import typing
import logging

import ngsolve as ngs

from dream.config import MultipleConfiguration

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration


class Formulation(MultipleConfiguration, is_interface=True):

    def set_configuration(self, cfg: SolverConfiguration) -> None:
        self._cfg = cfg

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
    def mesh(self) -> ngs.Mesh:
        return self.cfg.mesh

    @property
    def cfg(self) -> SolverConfiguration:
        return self._cfg

    def get_system(self, blf: list[ngs.comp.SumOfIntegrals], lf: list[ngs.comp.SumOfIntegrals]):
        raise NotImplementedError()


class PDEConfiguration(MultipleConfiguration, is_interface=True):

    @property
    def mesh(self) -> ngs.Mesh:
        if self._mesh is None:
            raise ValueError("Mesh is not set!")
        return self._mesh

    @property
    def formulation(self) -> Formulation:
        raise NotImplementedError("Overload this property in derived class!")

    def __init__(self, mesh: ngs.Mesh = None, **kwargs) -> None:
        self._mesh = mesh
        super().__init__(**kwargs)
