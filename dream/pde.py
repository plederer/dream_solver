from __future__ import annotations
import typing
import logging
import ngsolve as ngs

from dream.config import MultipleConfiguration, any, State, is_notebook

if typing.TYPE_CHECKING:
    from dream.solver import SolverConfiguration

logger = logging.getLogger(__name__)


class FiniteElement(MultipleConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @property
    def mesh(self) -> ngs.Mesh:
        return self.cfg.mesh

    @any(default=2)
    def order(self, order):
        return int(order)

    def add_finite_element_spaces(self, spaces: dict[str, ngs.FESpace]):
        pass

    def add_transient_gridfunctions(self, gfus: dict[str, dict[str, ngs.GridFunction]]):
        pass

    def add_discrete_system(self, blf: dict[str, ngs.comp.SumOfIntegrals],
                            lf: dict[str, ngs.comp.SumOfIntegrals]):
        pass

    def get_drawing_state(self, quantities: dict[str, bool]) -> State:
        raise NotImplementedError()

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

        self.spaces = {}
        self.fe.add_finite_element_spaces(self.spaces)

        if not self.spaces:
            raise ValueError("Spaces container is empty!")

        fes = list(self.spaces.values())

        for space in fes[1:]:
            fes[0] *= space

        self.fes: ngs.ProductSpace = fes[0]

    def set_trial_and_test_functions(self) -> None:
        self.TnT = {label: (tr, te) for label, tr, te in zip(
            self.spaces, self.fes.TrialFunction(), self.fes.TestFunction())}

    def set_gridfunctions(self) -> None:
        self.gfu = ngs.GridFunction(self.fes)

        self.gfus = {label: self.gfu for label in self.spaces.keys()}
        for label, gfu in zip(self.spaces.keys(), self.gfu.components):
            self.gfus[label] = gfu

    def set_transient_gridfunctions(self) -> None:
        self.transient_gfus = {}
        self.fe.add_transient_gridfunctions(self.transient_gfus)

    def set_discrete_system_tree(self) -> None:
        self.blf = {}
        self.lf = {}
        self.fe.add_discrete_system(self.blf, self.lf)

    def set_boundary_conditions(self) -> None:
        self.fe.set_boundary_conditions()

    def set_initial_conditions(self) -> None:
        self.fe.set_initial_conditions()

    def get_drawing_state(self, **quantities) -> State:
        self.drawings = self.fe.get_drawing_state(quantities)

        for quantity in quantities:
            logger.info(f"Can not draw {quantity}!")

        return self.drawings

    def draw(self, **kwargs):

        if hasattr(self, "drawings"):

            if is_notebook():
                from ngsolve.webgui import Draw
            else:
                from ngsolve import Draw

            self.scenes = [Draw(draw, self.mesh, name, **kwargs) for name, draw in self.drawings.items()]

    def redraw(self, blocking: bool = False):

        if hasattr(self, "scenes"):

            for scene in self.scenes:
                if scene is not None:
                    scene.Redraw()
            ngs.Redraw(blocking)
