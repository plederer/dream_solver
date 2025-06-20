""" Definitions of conservative dg discretizations. """
from __future__ import annotations
import logging
import ngsolve as ngs
import dream.bla as bla

from dream.time import TimeSchemes, TransientRoutine
from dream.config import dream_configuration, Integrals
from dream.compressible.config import ConservativeFiniteElementMethod, Periodic, Initial

from .time import ExplicitEuler, SSPRK3, CRK4

logger = logging.getLogger(__name__)

class ConservativeDG(ConservativeFiniteElementMethod):

    name: str = "conservative_dg"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
        }

        DEFAULT.update(default)

        logger.warning("Conservative DG method is still experimental and may not be fully functional!")

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def scheme(self) -> ExplicitEuler | SSPRK3 | CRK4:
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: TimeSchemes) -> None:

        if not isinstance(self.root.time, TransientRoutine):
            raise TypeError("DG method only supports transient time routines!")

        OPTIONS = [ExplicitEuler, SSPRK3, CRK4]
        self._scheme = self._get_configuration_option(scheme, OPTIONS, TimeSchemes)

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        order = self.root.fem.order
        dim = self.mesh.dim + 2

        U = ngs.L2(self.mesh, order=order, dgjumps=True)

        fes['U'] = U**dim

    def add_convection_form(self, blf: Integrals, lf:  Integrals):

        # Extract the bonus integration order, if specified.
        bonus = self.bonus_int_order['convection']

        # Obtain the relevant test and trial functions. Notice, the solution "U"
        # is assumed to be an unknown in the bilinear form, despite being explicit
        # in time. This works, because we invoke the "Apply" function when solving.
        U, V = self.TnT['U']

        # Get a mask that is nonzero (unity) for only the internal faces.
        mask = self.get_domain_boundary_mask()

        # Current/owned solution.
        Ui = self.get_conservative_fields(U)

        # Neighboring solution.
        Uj = self.get_conservative_fields(U.Other())

        # Compute the flux of the solution on the volume elements.
        F = self.root.get_convective_flux(Ui)

        # Compute the flux on the surface of an element, in the normal direction.
        Fn = self.root.riemann_solver.get_convective_numerical_flux_dg(Ui, Uj, self.mesh.normal)

        # Assemble the explicit bilinear form, keeping in mind this is placed on the RHS.
        blf['U']['convection'] = bla.inner(F, ngs.grad(V)) * ngs.dx(bonus_intorder=bonus['vol'])
        blf['U']['convection'] += -mask*bla.inner(Fn, V) * ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])

    def add_boundary_conditions(self, blf: Integrals, lf:  Integrals):

        bnds = self.root.bcs.to_pattern()

        for bnd, bc in bnds.items():

            logger.debug(f"Adding boundary condition {bc} on boundary {bnd}.")

            # NOTE: for now, we only implement periodic conditions.
            if isinstance(bc, Periodic):
                continue

            else:
                raise TypeError(f"Boundary condition {bc} not implemented in {self}!")

    def add_domain_conditions(self, blf: Integrals, lf:  Integrals):

        doms = self.root.dcs.to_pattern()

        for dom, dc in doms.items():

            logger.debug(f"Adding domain condition {dc} on domain {dom}.")

            if isinstance(dc, Initial):
                continue

            else:
                raise TypeError(f"Domain condition {dc} not implemented in {self}!")
