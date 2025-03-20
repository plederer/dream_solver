from __future__ import annotations
import typing
import logging
import ngsolve as ngs

from dream.config import InterfaceConfiguration, configuration, is_notebook, ngsdict
from dream.mesh import BoundaryConditions, DomainConditions, Periodic

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration

logger = logging.getLogger(__name__)


class FiniteElementMethod(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @configuration(default=2)
    def order(self, order):
        return int(order)

    def add_finite_element_spaces(self, spaces: dict[str, ngs.FESpace]):
        raise NotImplementedError("Overload this method in derived class!")

    def add_transient_gridfunctions(self, gfus: dict[str, dict[str, ngs.GridFunction]]):
        raise NotImplementedError("Overload this method in derived class!")

    def add_symbolic_forms(self, 
                           blfi: dict[str, ngs.comp.SumOfIntegrals],
                           blfe: dict[str, ngs.comp.SumOfIntegrals],
                           lf:   dict[str, ngs.comp.SumOfIntegrals]):
        raise NotImplementedError("Overload this method in derived class!")

    def get_fields(self, quantities: dict[str, bool]) -> ngsdict:
        raise NotImplementedError("Overload this method in derived class!")

    def set_initial_conditions(self) -> None:
        raise NotImplementedError("Overload this method in derived class!")

    def set_boundary_conditions(self) -> None:
        raise NotImplementedError("Overload this method in derived class!")

    order: int


class PDEConfiguration(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration
    bcs: BoundaryConditions
    dcs: DomainConditions

    @property
    def fem(self) -> FiniteElementMethod:
        raise NotImplementedError("Overload this configuration in derived class!")

    def initialize_system(self) -> None:
        self.initialize_finite_element_spaces()
        self.initialize_trial_and_test_functions()
        self.initialize_gridfunctions()
        self.initialize_boundary_conditions()

        if not self.cfg.time.is_stationary:
            self.initialize_transient_gridfunctions()
            self.initialize_initial_conditions()

        self.initialize_symbolic_forms()

    def initialize_finite_element_spaces(self) -> None:

        self.spaces = {}
        self.fem.add_finite_element_spaces(self.spaces)

        if not self.spaces:
            raise ValueError("Spaces container is empty!")

        fes = list(self.spaces.values())

        for space in fes[1:]:
            fes[0] *= space

        self.fes: ngs.ProductSpace = fes[0]

    def initialize_trial_and_test_functions(self) -> None:

        if len(self.spaces) > 1:
            self.TnT = {label: (tr, te) for label, tr, te in zip(
                self.spaces, self.fes.TrialFunction(), self.fes.TestFunction())}
        else:
            self.TnT = {label: self.fes.TnT() for label in self.spaces}

    def initialize_gridfunctions(self) -> None:
        self.gfu = ngs.GridFunction(self.fes)

        if len(self.spaces) > 1:
            self.gfus = {label: gfu for label, gfu in zip(self.spaces, self.gfu.components)}
        else:
            self.gfus = {label: self.gfu for label in self.spaces}

    def initialize_transient_gridfunctions(self) -> None:
        self.transient_gfus = {}
        self.fem.add_transient_gridfunctions(self.transient_gfus)

    def initialize_symbolic_forms(self) -> None:
        self.blfi = {}
        self.blfe = {}
        self.lf   = {}
        self.fem.add_symbolic_forms(self.blfi, self.blfe, self.lf)

    def initialize_boundary_conditions(self) -> None:

        if self.mesh.is_periodic and not self.bcs.has_condition(Periodic):
            raise ValueError("Mesh has periodic boundaries, but no periodic boundary conditions are set!")

        self.fem.set_boundary_conditions()

    def initialize_initial_conditions(self) -> None:
        self.fem.set_initial_conditions()

        if not self.cfg.time.is_stationary:

            if not hasattr(self, "transient_gfus"):
                raise ValueError("Transient gridfunctions are not set!")

            self.cfg.time.scheme.set_initial_conditions(self.transient_gfus)

    def get_fields(self, **quantities: bool) -> ngsdict:
        state = self.fem.get_fields(quantities)

        for quantity in quantities:
            logger.info(f"Quantity {quantity} not predefined!")

        return state

    def draw(self, fields: ngsdict, **kwargs):
        if is_notebook():
            from ngsolve.webgui import Draw
        else:
            from ngsolve import Draw

        self.scenes = [Draw(draw, self.mesh, name, **kwargs) for name, draw in fields.items()]

    def redraw(self, blocking: bool = False):

        if hasattr(self, "scenes"):

            for scene in self.scenes:
                if scene is not None:
                    scene.Redraw()
            ngs.Redraw(blocking)
