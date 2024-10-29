from __future__ import annotations
import typing

import ngsolve as ngs
from dream.config import MultipleConfiguration, any

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration


class FiniteElement(MultipleConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @property
    def mesh(self) -> ngs.Mesh:
        return self.cfg.mesh

    @any(default=2)
    def order(self, order):
        return int(order)

    def get_finite_element_spaces(self, spaces: dict[str, ngs.FESpace] = None):

        if spaces is None:
            spaces = {}

        return spaces

    def get_transient_gridfunctions(self, gfus: dict[str, dict[str, ngs.GridFunction]] = None):

        if gfus is None:
            gfus = {}

        return gfus

    def get_discrete_system(self, blf: dict[str, ngs.comp.SumOfIntegrals] = None,
                            lf: dict[str, ngs.comp.SumOfIntegrals] = None):

        if blf is None:
            blf = {}

        if lf is None:
            lf = {}

        return blf, lf

    def set_initial_conditions(self) -> None:
        raise NotImplementedError()

    def set_boundary_conditions(self) -> None:
        raise NotImplementedError()

    order: int


class PDEConfiguration(MultipleConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @property
    def mesh(self) -> ngs.Mesh:
        return self.cfg.mesh

    @property
    def fe(self) -> FiniteElement:
        raise NotImplementedError("Overload this configuration in derived class!")

    def set_finite_element_spaces(self) -> None:
        spaces = self.fe.get_finite_element_spaces()

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

    def set_transient_gridfunctions(self) -> None:
        self.gfus_transient = self.fe.get_transient_gridfunctions()

    def set_discrete_system_tree(self) -> None:
        self.blf, self.lf = self.fe.get_discrete_system()

    def set_boundary_conditions(self) -> None:
        self.fe.set_boundary_conditions()

    def set_initial_conditions(self) -> None:
        self.fe.set_initial_conditions()
