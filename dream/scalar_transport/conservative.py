from __future__ import annotations
import logging
import ngsolve as ngs
import typing
import dream.bla as bla

from dream.config import Configuration, dream_configuration, Integrals
from dream.mesh import Periodic, Initial
from dream.solver import FiniteElementMethod
from dream.scalar_transport.config import (flowfields)

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from .solver import ScalarTransportSolver



class ScalarTransportFiniteElementMethod(FiniteElementMethod):

    root: ScalarTransportSolver

    def __init__(self, mesh, root=None, **default):
        DEFAULT = {"interior_penalty_coefficient": 1.0}
        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def interior_penalty_coefficient(self) -> float:
        return self._interior_penalty_coefficient

    @interior_penalty_coefficient.setter
    def interior_penalty_coefficient(self, alpha: float) -> None:
        if alpha < 0.0:
            raise ValueError("Interior penalty coefficient must be +ve.")

        self._interior_penalty_coefficient = alpha

    @property
    def TnT(self) -> dict[str, tuple[ngs.comp.ProxyFunction, ...]]:
        return self.root.TnT

    @property
    def gfu(self) -> dict[str, ngs.GridFunction]:
        return self.root.gfus

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]):
        raise NotImplementedError()

    def add_symbolic_spatial_forms(self, blf: Integrals, lf: Integrals):
        raise NotImplementedError()

    def add_symbolic_temporal_forms(self, blf: Integrals, lf: Integrals):
        self.root.time.scheme.add_symbolic_temporal_forms('U', blf, lf)

    def get_temporal_integrators(self):
        return {'U': ngs.dx}

    def set_initial_conditions(self) -> None:
        super().set_initial_conditions()
        U = self.mesh.MaterialCF({dom: self.root.phi(dc.fields) for dom, dc in self.root.dcs.to_pattern(Initial).items()})

        self.gfu['U'].Set(U)

    def get_domain_boundary_mask(self) -> ngs.GridFunction:
        """ 
        Returns a Gridfunction that is 0 on the domain boundaries and 1 on the domain interior.
        """
        fes = ngs.FacetFESpace(self.mesh, order=0)
        mask = ngs.GridFunction(fes, name="mask")
        mask.vec[:] = 0

        bnd_dofs = fes.GetDofs(self.mesh.Boundaries(self.root.bcs.get_domain_boundaries(True)))
        mask.vec[~bnd_dofs] = 1

        return mask

    def get_fields(self, *fields: str, default: bool = True) -> flowfields:

        U = self.get_transported_fields(self.gfu['U'])

        fields_ = flowfields()
        #return fields_
        return U


    def get_transported_fields(self, U: ngs.CoefficientFunction) -> flowfields:

        U_ = flowfields()
        U_.phi = U
  
        if isinstance(U, ngs.comp.ProxyFunction):
            U_.U = U

        return U_

    def add_boundary_conditions(self, blf: Integrals, lf: Integrals):

        bnds = self.root.bcs.to_pattern()

        for bnd, bc in bnds.items():

            logger.debug(f"Adding boundary condition {bc} on boundary {bnd}.")

            if isinstance(bc, Periodic):
                continue

            else:
                raise TypeError(f"Boundary condition {bc} not implemented in {self}!")

    def add_domain_conditions(self, blf: Integrals, lf: Integrals):

        doms = self.root.dcs.to_pattern()

        for dom, dc in doms.items():

            logger.debug(f"Adding domain condition {dc} on domain {dom}.")

            if isinstance(dc, Initial):
                continue

            else:
                raise TypeError(f"Domain condition {dc} not implemented in {self}!")

    def set_boundary_conditions(self) -> None:
        """ Boundary conditions for the scalar transport equation are set weakly. Therefore we do nothing here."""
        pass




class HDG(ScalarTransportFiniteElementMethod):

    name: str = "hdg"

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        order = self.root.fem.order

        U = ngs.L2(self.mesh, order=order)
        Uhat = ngs.FacetFESpace(self.mesh, order=order)

        if self.root.bcs.has_condition(Periodic):
            Uhat = ngs.Periodic(Uhat)

        fes['U'] = U
        fes['Uhat'] = Uhat

    
    def add_symbolic_spatial_forms(self, blf: Integrals, lf: Integrals):

        self.add_convection_form(blf, lf)

        if not self.root.is_inviscid:
            self.add_diffusion_form(blf, lf)

        self.add_boundary_conditions(blf, lf)
        self.add_domain_conditions(blf, lf)


    def add_convection_form(self, blf: Integrals, lf: Integrals):

        bonus = self.root.optimizations.bonus_int_order
        mask = self.get_domain_boundary_mask()

        U, V = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_transported_fields(U)
        Uhat = self.get_transported_fields(Uhat)

        F = self.root.get_convective_flux(U)
        Fn = self.get_convective_numerical_flux(U, Uhat, self.mesh.normal)

        # For context, the terms are defined on the LHS.
        blf['U']['convection']  = -bla.inner(F, ngs.grad(V)) \
                                *  ngs.dx(bonus_intorder=bonus.vol)
        blf['U']['convection'] += bla.inner(Fn, V)           \
                                * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)
        
        # For context, the transmissibility equation is multiplied by a minus sign.
        blf['Uhat']['convection'] = -mask * bla.inner(Fn,Vhat) \
                                  *  ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)


    def get_convective_numerical_flux(self, U: flowfields, Uhat: flowfields, unit_vector: bla.VECTOR):
        unit_vector = bla.as_vector(unit_vector)
        wind = self.root.convection_velocity
        
        tau = self.root.riemann_solver.get_convective_stabilisation_hdg(wind, unit_vector)

        return self.root.get_convective_flux(Uhat) * unit_vector + tau * (U.U - Uhat.U)



    def add_diffusion_form(self, blf: Integrals, lf: Integrals):

        bonus = self.root.optimizations.bonus_int_order
        mask = self.get_domain_boundary_mask()

        U, V = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        # Some abbreviations for readability.
        k = self.root.diffusion_coefficient
        n = self.mesh.normal
        dU = ngs.grad(U)
        dV = ngs.grad(V)
        dUn = dU*n
        dVn = dV*n
        
        # Get the parameters needed for the penalty coefficient: tau.
        N = self.order
        h = self.mesh.meshsize
        alpha = self.interior_penalty_coefficient 
        tau = (alpha*(N+1)**2)/h

        # For context, the terms are defined on the LHS.
        blf['U']['diffusion']  = ngs.InnerProduct(k*dU, dV)          \
                               * ngs.dx(bonus_intorder=bonus.vol)
        blf['U']['diffusion'] -= ngs.InnerProduct(k*(U-Uhat), dVn)   \
                               * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd) 
        blf['U']['diffusion'] -= ngs.InnerProduct(k*dUn, V)          \
                               * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)
        blf['U']['diffusion'] += ngs.InnerProduct(tau*k*(U-Uhat), V) \
                               * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

        # For context, the transmissibility equation is multiplied by a minus sign.
        blf['Uhat']['diffusion']  = mask * ngs.InnerProduct(k*dUn, Vhat)          \
                                  * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)
        blf['Uhat']['diffusion'] -= mask * ngs.InnerProduct(tau*k*(U-Uhat), Vhat) \
                                  * ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

    def set_initial_conditions(self, U: ngs.CF = None):
        super().set_initial_conditions()

        # The volume solution (U) has already been initialized in the base class, use it.
        U = self.gfu['U']

        gfu = self.gfu['Uhat']
        fes = self.gfu['Uhat'].space
        u, v = fes.TnT()

        blf = ngs.BilinearForm(fes)
        blf += u * v * ngs.dx(element_boundary=True)

        f = ngs.LinearForm(fes)
        f += U * v * ngs.dx(element_boundary=True)

        with ngs.TaskManager():
            blf.Assemble()
            f.Assemble()
            gfu.vec.data = blf.mat.Inverse(freedofs=fes.FreeDofs(), inverse="sparsecholesky") * f.vec





