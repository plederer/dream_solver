""" Definitions of conservative dg discretizations. """
from __future__ import annotations
import logging
import ngsolve as ngs
import dream.bla as bla

from dream.time import TimeSchemes, TransientRoutine, MultizoneIMEXTimeRoutine
from dream.config import Configuration, dream_configuration, Integrals
from dream.compressible.config import (flowfields,
                                       ConservativeFiniteElementMethod,
                                       FarField,
                                       Outflow,
                                       InterfaceBC,
                                       Periodic,
                                       Initial)

from .time import ExplicitEuler, SSPRK3, CRK4, RK_ARS22, RK_ARS33, RK_ARS43

logger = logging.getLogger(__name__)


class ViscousTreatment(Configuration, is_interface=True):

    root: CompressibleFlowSolver

    @property
    def fem(self) -> ConservativeFiniteElementMethod:
        return self.root.fem

    @property
    def TnT(self) -> dict[str, tuple[ngs.comp.ProxyFunction, ...]]:
        return self.root.fem.TnT

    def add_diffusion_form(self, blf: Integrals, lf: Integrals) -> None:
        raise NotImplementedError("A method for the viscous treatment of the diffusive forms must be implemented.")


class InteriorPenaltyMethodSDG(ViscousTreatment):
    r""" This is based on the implementation in:

    Hartmann, R. and Houston, P., 2008. An optimal order interior penalty discontinuous Galerkin discretization of the compressible Navier–Stokes equations. Journal of Computational Physics, 227(22), pp.9670-9685.
    """
    name: str = "interior_penalty_method_sdg"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {'interior_penalty_coefficient': 1.0,}

        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def interior_penalty_coefficient(self) -> float:
        r""" Sets the interior penalty constant.

            :getter: Returns the interior penalty constant
            :setter: Sets the interior penalty constant, defaults to 1.0
        """
        return self._interior_penalty_coefficient

    @interior_penalty_coefficient.setter
    def interior_penalty_coefficient(self, alpha: float) -> None:
        if alpha < 0.0:
            raise ValueError("Interior penalty coefficient must be +ve.")
        self._interior_penalty_coefficient = alpha

    def get_scaled_penalty_coefficient(self):
        h = self.mesh.meshsize
        alpha = self.interior_penalty_coefficient 

        # This is taken from Hillewaert's thesis, see Table 3.1 in [1].
        # [1] Hillewaert, Koen. Development of the discontinuous Galerkin method for 
        #     high-resolution, large scale CFD and acoustics in industrial geometries. 
        #     Presses univ. de Louvain, 2013.
        #
        # ... alpha is usually set to unity, higher numbers imply a stiffer system. 
        #     Additionally, this assumes non-curved elements.
        # Perhaps an if condition if this is grid uses triangles/quads? Note, however, 
        # if we use an imex with quads and triangles on both sides, we need to take quads, 
        # as this is a stronger threshold.
        #return alpha * (self.fem.order + 1) * (self.fem.order + 2) / (2*h) # for triangles
        return alpha * (self.fem.order + 1)**2 / h # for quadrilaterals
    
    def add_diffusion_form(self, blf: Integrals, lf: Integrals):

        bonus = self.fem.bonus_int_order['diffusion']
        dV = ngs.dx(bonus_intorder=bonus['vol'])
        U, V = self.TnT['U']

        # Current/owned solution, which includes the gradient.
        U = self.fem.get_conservative_fields(U, with_gradients=True)

        # Compute the flux of the solution, needed for the volume terms.
        G = self.root.get_diffusive_flux(U, U)

        # Add the volume term. NOTE, this must be done before the remaining surface terms.
        blf['U']['diffusion'] = ngs.InnerProduct(G, ngs.grad(V)) * dV

        # Add the remaining surface, symmetrizing and penalty terms.
        self.add_surface_term(blf, lf)
        self.add_symmetrizing_term(blf, lf)
        self.add_penalizing_term(blf, lf)

    def add_surface_term(self, blf: Integrals, lf: Integrals) -> None:

        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.dx(skeleton=True, bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        # Extract local and neighboring solution (with gradients).
        Ui = self.fem.get_conservative_fields(U, with_gradients=True)
        Uj = self.fem.get_conservative_fields(U.Other(), with_gradients=True)

        # Compute the flux of the solution on the current and neighbor element.
        Gi = self.root.get_diffusive_flux(Ui, Ui)
        Gj = self.root.get_diffusive_flux(Uj, Uj)

        # Jump of the test functions.
        jumpV = V - V.Other()

        # Form the surface term.
        surf = (Gi + Gj) * self.mesh.normal / 2 
        blf['U']['diffusion'] -= ngs.InnerProduct(surf, jumpV) * dS

    def add_symmetrizing_term(self, blf: Integrals, lf: Integrals) -> None:

        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.dx(skeleton=True, bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        # Extract the components of the unit normal vector.
        nx = self.mesh.normal[0]
        ny = self.mesh.normal[1]

        # Extract local and neighboring solution (with gradients).
        Ui = self.fem.get_conservative_fields(U, with_gradients=True)
        Uj = self.fem.get_conservative_fields(U.Other(), with_gradients=True)

        # Jump of the conservative solution.
        jumpU = U - U.Other()

        # Get the gradient of the test functions.
        gradVi = ngs.grad(V)
        gradVj = ngs.grad(V.Other())
        
        # Extract the x and y-components of the gradient of the test functions.
        dVidx = gradVi[:, 0]; dVidy = gradVi[:, 1]
        dVjdx = gradVj[:, 0]; dVjdy = gradVj[:, 1]

        # Get the diffusion matrices transposed, based on the conservative gradients.
        KU11Ti, KU12Ti, KU21Ti, KU22Ti = self.get_frozen_diffusion_matrices_conservative_transposed(Ui)
        KU11Tj, KU12Tj, KU21Tj, KU22Tj = self.get_frozen_diffusion_matrices_conservative_transposed(Uj)

        # Form the term: {G^T * grad(V)}, for the current and neighbor solution.
        FTi = nx*(KU11Ti*dVidx + KU21Ti*dVidy) + ny*(KU12Ti*dVidx + KU22Ti*dVidy) 
        FTj = nx*(KU11Tj*dVjdx + KU21Tj*dVjdy) + ny*(KU12Tj*dVjdx + KU22Tj*dVjdy)
  
        # Form the symmetrizing term.
        symm = (FTi + FTj) / 2 
        blf['U']['diffusion'] -= ngs.InnerProduct(symm, jumpU) * dS

    def add_penalizing_term(self, blf: Integrals, lf: Integrals) -> None:

        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.dx(skeleton=True, bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        # Extract the components of the unit normal vector.
        nx = self.mesh.normal[0]
        ny = self.mesh.normal[1]

        # Extract local and neighboring solution (with gradients).
        Ui = self.fem.get_conservative_fields(U, with_gradients=True)
        Uj = self.fem.get_conservative_fields(U.Other(), with_gradients=True)

        # Jump of the conservative solution and test functions.
        jumpU = U - U.Other()
        jumpV = V - V.Other()

        # Interior penalty coefficient.
        tau = self.get_scaled_penalty_coefficient() 

        # Get the diffusion matrices, based on the conservative gradients.
        KU11i, KU12i, KU21i, KU22i = self.get_frozen_diffusion_matrices_conservative(Ui)
        KU11j, KU12j, KU21j, KU22j = self.get_frozen_diffusion_matrices_conservative(Uj)

        # Average the diffusion matrices.
        KU11 = (KU11i + KU11j) / 2
        KU12 = (KU12i + KU12j) / 2
        KU21 = (KU21i + KU21j) / 2
        KU22 = (KU22i + KU22j) / 2

        # Form the penalty term, which is based on the contraction of the diffusion matrices.
        penn = nx * ( KU11*nx + KU12*ny ) + ny * ( KU21*nx + KU22*ny )
        blf['U']['diffusion'] += ngs.InnerProduct(tau*penn * jumpU, jumpV) * dS

    def get_frozen_diffusion_matrices_conservative_transposed(self, U: flowfields):
        
        KU11, KU12, KU21, KU22 = self.get_frozen_diffusion_matrices_conservative(U)
        return KU11.trans, KU12.trans, KU21.trans, KU22.trans

    def get_frozen_diffusion_matrices_conservative(self, U: flowfields):
        
        # Get the relevant nondimensional numbers.
        Re = self.root.scaling.reference_reynolds_number
        Pr = self.root.prandtl_number

        # Extract the velocities.
        u = U.u[0] 
        v = U.u[1]
        
        # Extract the density and energies.
        rho = U.rho 
        ei  = self.root.specific_inner_energy(U)
        ek  = self.root.specific_kinetic_energy(U)

        # Extract the viscosity.
        mu = self.root.viscosity(U)

        # NOTE, we explicitly scale the viscosity with Re, 
        # to obtain the correct nondimensionalization.
        mu /= Re

        # Compute the respective thermal conductivity constant.
        kappa = mu/Pr # NOTE, mu already is divided by Re.
        
        # Get the nondimensionalized specific heat at constant volume.
        cv = self.root.equation_of_state.specific_heat_cv

        # Second viscosity.
        lmb = -2*mu/3

        # Abbreviations.
        ovrho = 1.0/rho
        ovrhocv = ovrho/cv
        lmbp2mu = lmb + 2*mu
       
        # velocity squared. 
        u2 = u*u
        v2 = v*v

        # Form the (frozen) diagonal diffusion matrix: K11.
        KU11_21 = -ovrho*lmbp2mu * u
        KU11_22 =  ovrho*lmbp2mu
        KU11_31 = -ovrho*mu * v
        KU11_33 =  ovrho*mu
        KU11_41 =  ovrhocv*kappa*(ek-ei) - ovrho*( lmbp2mu*u2 + mu*v2 )
        KU11_42 = (ovrho*lmbp2mu - ovrhocv*kappa) * u 
        KU11_43 = (ovrho*mu      - ovrhocv*kappa) * v
        KU11_44 =  ovrhocv*kappa

        KU11 = ngs.CF((0,       0,       0,       0,
                       KU11_21, KU11_22, 0,       0,
                       KU11_31, 0,       KU11_33, 0,
                       KU11_41, KU11_42, KU11_43, KU11_44), dims=(4,4))
        

        # Form the (frozen) diagonal diffusion matrix: K22.
        KU22_21 = -ovrho*mu * u
        KU22_22 =  ovrho*mu
        KU22_31 = -ovrho*lmbp2mu * v
        KU22_33 =  ovrho*lmbp2mu
        KU22_41 =  ovrhocv*kappa*(ek-ei) - ovrho*( mu*u2 + lmbp2mu*v2 )
        KU22_42 = (ovrho*mu      - ovrhocv*kappa) * u 
        KU22_43 = (ovrho*lmbp2mu - ovrhocv*kappa) * v
        KU22_44 =  ovrhocv*kappa

        KU22 = ngs.CF((0,       0,       0,       0,
                       KU22_21, KU22_22, 0,       0,
                       KU22_31, 0,       KU22_33, 0,
                       KU22_41, KU22_42, KU22_43, KU22_44), dims=(4,4))
        

        # Form the (frozen) off-diagonal diffusion matrix: K12.
        KU12_21 = -ovrho*lmb * v
        KU12_23 =  ovrho*lmb
        KU12_31 = -ovrho*mu  * u
        KU12_32 =  ovrho*mu
        KU12_41 = -ovrho*( lmb + mu ) * u*v
        KU12_42 =  ovrho*mu  * v 
        KU12_43 =  ovrho*lmb * u

        KU12 = ngs.CF((0,       0,       0,       0,
                       KU12_21, 0,       KU12_23, 0,
                       KU12_31, KU12_32, 0,       0,
                       KU12_41, KU12_42, KU12_43, 0), dims=(4,4))
        

        # Form the (frozen) off-diagonal diffusion matrix: K21.
        KU21_21 = -ovrho*mu  * v
        KU21_23 =  ovrho*mu
        KU21_31 = -ovrho*lmb * u
        KU21_32 =  ovrho*lmb
        KU21_41 = -ovrho*( lmb + mu ) * u*v
        KU21_42 =  ovrho*lmb * v 
        KU21_43 =  ovrho*mu  * u

        KU21 = ngs.CF((0,       0,       0,       0,
                       KU21_21, 0,       KU21_23, 0,
                       KU21_31, KU21_32, 0,       0,
                       KU21_41, KU21_42, KU21_43, 0), dims=(4,4))

        # We're done.
        return KU11, KU12, KU21, KU22

    def add_viscous_interface_formulation(self, blf: Integrals, lf: Integrals, bc: InterfaceBC, bnd: str):
        
        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        # Form the boundary state, written as conservative variables.
        U_infty = ngs.CF(
            (self.root.density(bc.fields),
             self.root.momentum(bc.fields),
             self.root.energy(bc.fields)))

        # Extract local and neighboring solution (with gradients).
        Ui = self.fem.get_conservative_fields(U, with_gradients=True)
        Uj = bc.fields # NOTE, this uses the Other's relations to compute its quantities.

        # # # 
        # Surface term.
        # # 

        # Compute the flux of the solution on the current and neighbor element.
        Gi = self.root.get_diffusive_flux(Ui, Ui)
        Gj = self.root.get_diffusive_flux(Uj, Uj)

        # Form the surface term.
        surf = (Gi + Gj) * self.mesh.normal / 2 
        blf['U'][f"{bc.name}_{bnd}"] -= ngs.InnerProduct(surf, V) * dS

        # # #
        # Symmetrizing term.
        # # 

        # Extract the components of the unit normal vector.
        nx = self.mesh.normal[0]
        ny = self.mesh.normal[1]

        # Jump of the conservative solution.
        jumpU = U - U_infty

        # Get the gradient of the test functions.
        gradVi = ngs.grad(V)
        
        # Extract the x and y-components of the gradient of the test functions.
        dVidx = gradVi[:, 0]; dVidy = gradVi[:, 1]

        # Get the diffusion matrices transposed, based on the conservative gradients.
        KU11Ti, KU12Ti, KU21Ti, KU22Ti = self.get_frozen_diffusion_matrices_conservative_transposed(Ui)
        KU11Tj, KU12Tj, KU21Tj, KU22Tj = self.get_frozen_diffusion_matrices_conservative_transposed(Uj)

        KU11T = (KU11Ti + KU11Tj) / 2
        KU12T = (KU12Ti + KU12Tj) / 2
        KU21T = (KU21Ti + KU21Tj) / 2
        KU22T = (KU22Ti + KU22Tj) / 2
        
        # Form the term: {G^T * grad(V)}, for the current and neighbor solution.
        FT = nx*(KU11T*dVidx + KU21T*dVidy) + ny*(KU12T*dVidx + KU22T*dVidy) 
  
        # Form the symmetrizing term.
        blf['U'][f"{bc.name}_{bnd}"] -= ngs.InnerProduct(FT, jumpU) * dS

        # # # 
        # Penalty term.
        # # 

        # Interior penalty coefficient.
        tau = self.get_scaled_penalty_coefficient() 

        # Get the diffusion matrices, based on the conservative gradients.
        KU11i, KU12i, KU21i, KU22i = self.get_frozen_diffusion_matrices_conservative(Ui)
        KU11j, KU12j, KU21j, KU22j = self.get_frozen_diffusion_matrices_conservative(Uj)

        # Average the diffusion matrices.
        KU11 = (KU11i + KU11j) / 2
        KU12 = (KU12i + KU12j) / 2
        KU21 = (KU21i + KU21j) / 2
        KU22 = (KU22i + KU22j) / 2

        # Form the penalty term, which is based on the contraction of the diffusion matrices.
        penn = nx * ( KU11*nx + KU12*ny ) + ny * ( KU21*nx + KU22*ny )
        blf['U'][f"{bc.name}_{bnd}"] += ngs.InnerProduct(tau*penn * jumpU, V) * dS

    def add_viscous_farfield_formulation(self, U_infty: ngs.CF, 
                                         blf: Integrals, lf: Integrals, bc: FarField, bnd: str):
        
        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        # Extract local and neighboring solution (with gradients).
        Ui = self.fem.get_conservative_fields(U, with_gradients=True)
        Uj = self.fem.get_conservative_fields(U.Other(bnd=U_infty))

        # # # 
        # Surface term.
        # # 

        # Compute the flux of the solution, which is afunction of the solution and external.
        G = self.root.get_diffusive_flux(Uj, Ui) # NOTE, the gradient is from the local solution.

        # Form the surface term.
        surf = G * self.mesh.normal
        blf['U'][f"{bc.name}_{bnd}"] -= ngs.InnerProduct(surf, V) * dS

        # # #
        # Symmetrizing term.
        # # 

        # Extract the components of the unit normal vector.
        nx = self.mesh.normal[0]
        ny = self.mesh.normal[1]

        # Jump of the conservative solution.
        jumpU = U - U_infty

        # Get the gradient of the test functions.
        gradV = ngs.grad(V)
        
        # Extract the x and y-components of the gradient of the test functions.
        dVdx = gradV[:, 0]; dVdy = gradV[:, 1]

        # Get the diffusion matrices transposed, based on the conservative gradients.
        KU11T, KU12T, KU21T, KU22T = self.get_frozen_diffusion_matrices_conservative_transposed(Ui)

        # Form the term: {G^T * grad(V)}, for the local solution.
        FT = nx*(KU11T*dVdx + KU21T*dVdy) + ny*(KU12T*dVdx + KU22T*dVdy) 
  
        # Form the symmetrizing term.
        blf['U'][f"{bc.name}_{bnd}"] -= ngs.InnerProduct(FT, jumpU) * dS

        # # # 
        # Penalty term.
        # # 

        # Interior penalty coefficient.
        tau = self.get_scaled_penalty_coefficient() 

        # Get the diffusion matrices, based on the conservative gradients.
        KU11, KU12, KU21, KU22 = self.get_frozen_diffusion_matrices_conservative(Ui)

        # Form the penalty term, which is based on the contraction of the diffusion matrices.
        penn = nx * ( KU11*nx + KU12*ny ) + ny * ( KU21*nx + KU22*ny )
        blf['U'][f"{bc.name}_{bnd}"] += ngs.InnerProduct(tau*penn * jumpU, V) * dS

        # NOTE,
        # if we compare this vs an HDG farfield, at least the way it is implemented, we notice:
        # 1) they are not exactly equivalent, as also observed numerically. 
        # 2) the HDG imposes only the convective part, which is still used in the viscous flux 
        #    at the surface (that isn't tested with Vhat). In other words, the penalty term on
        #    the facet is missing, as well as the jump in the viscous flux.
        # 3) We seem to get a similar solution if we comment the symmetrizing and penalty terms 
        #    in the current SDG's viscous farfield treatment. Which means that the surface term
        #    mimics the "internal" viscous fluxes inherent in the HDG formulation, which, to be 
        #    fair, are also a function of the facet solution obtain from the convective treatment.
        #
        # ... All the above suggests we still need to debate which is "better". 
        #     However, they should be equivalent when,
        #     (a): no disturbances impinge on the farfield boundary.
        #     (b): viscosity is negligible, or more generally, Re >> 1.


    def add_viscous_outflow_formulation(self, U_infty: ngs.CF, 
                                        blf: Integrals, lf: Integrals, bc: Outflow, bnd: str):
        
        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        # Extract local and neighboring solution (with gradients).
        Ui = self.fem.get_conservative_fields(U, with_gradients=True)
        Uj = self.fem.get_conservative_fields(U.Other(bnd=U_infty))

        # # # 
        # Surface term.
        # # 

        # Compute the flux of the solution, which is afunction of the solution and external.
        G = self.root.get_diffusive_flux(Uj, Ui) # NOTE, the gradient is from the local solution.

        # Form the surface term.
        surf = G * self.mesh.normal
        blf['U'][f"{bc.name}_{bnd}"] -= ngs.InnerProduct(surf, V) * dS

        # # #
        # Symmetrizing term.
        # # 

        # Extract the components of the unit normal vector.
        nx = self.mesh.normal[0]
        ny = self.mesh.normal[1]

        # Jump of the conservative solution.
        jumpU = U - U_infty

        # Get the gradient of the test functions.
        gradV = ngs.grad(V)
        
        # Extract the x and y-components of the gradient of the test functions.
        dVdx = gradV[:, 0]; dVdy = gradV[:, 1]

        # Get the diffusion matrices transposed, based on the conservative gradients.
        KU11T, KU12T, KU21T, KU22T = self.get_frozen_diffusion_matrices_conservative_transposed(Ui)

        # Form the term: {G^T * grad(V)}, for the local solution.
        FT = nx*(KU11T*dVdx + KU21T*dVdy) + ny*(KU12T*dVdx + KU22T*dVdy) 
  
        # Form the symmetrizing term.
        blf['U'][f"{bc.name}_{bnd}"] -= ngs.InnerProduct(FT, jumpU) * dS

        # # # 
        # Penalty term.
        # # 

        # Interior penalty coefficient.
        tau = self.get_scaled_penalty_coefficient() 

        # Get the diffusion matrices, based on the conservative gradients.
        KU11, KU12, KU21, KU22 = self.get_frozen_diffusion_matrices_conservative(Ui)

        # Form the penalty term, which is based on the contraction of the diffusion matrices.
        penn = nx * ( KU11*nx + KU12*ny ) + ny * ( KU21*nx + KU22*ny )
        blf['U'][f"{bc.name}_{bnd}"] += ngs.InnerProduct(tau*penn * jumpU, V) * dS

        

class ConservativeDG(ConservativeFiniteElementMethod):

    name: str = "conservative_dg"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {'viscous_treatment': None,}

        DEFAULT.update(default)

        logger.warning("Conservative DG method is still experimental and may not be fully functional!")

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def viscous_treatment(self) -> InteriorPenaltyMethodSDG | None:
        return self._viscous_treatment

    @viscous_treatment.setter
    def viscous_treatment(self, value: str | ViscousTreatment | None):
        
        if value is None:
            self._viscous_treatment = None
            return

        OPTIONS = [InteriorPenaltyMethodSDG]
        self._viscous_treatment = self._get_configuration_option(value, OPTIONS, ViscousTreatment)

    @dream_configuration
    def scheme(self) -> ExplicitEuler | SSPRK3 | CRK4 | RK_ARS22 | RK_ARS33 | RK_ARS43:
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: TimeSchemes) -> None:

        if isinstance(self.root.time, TransientRoutine):
            OPTIONS = [ExplicitEuler, SSPRK3, CRK4, RK_ARS22, RK_ARS33, RK_ARS43]

        elif isinstance(self.root.time, MultizoneIMEXTimeRoutine):
            OPTIONS = [ExplicitEuler, RK_ARS22, RK_ARS33, RK_ARS43]

        else:
            raise TypeError("SDG method only supports transient or mutizone time routines!")
        self._scheme = self._get_configuration_option(scheme, OPTIONS, TimeSchemes)

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        order = self.root.fem.order
        dim = self.mesh.dim + 2

        U = ngs.L2(self.mesh, order=order, dgjumps=True)

        fes['U'] = U**dim

    def add_convection_form(self, blf: Integrals, lf:  Integrals):

        bonus = self.bonus_int_order['convection']
        dV = ngs.dx(bonus_intorder=bonus['vol'])
        dS = ngs.dx(skeleton=True, bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        # Jump of the test functions.
        jumpV = V - V.Other()

        # Extract local and neighboring solution.
        Ui = self.get_conservative_fields(U)
        Uj = self.get_conservative_fields(U.Other())

        # Compute the flux of the solution on the volume elements.
        F = self.root.get_convective_flux(Ui)

        # Compute the flux on the surface of an element, in the normal direction.
        Fn = self.root.riemann_solver.get_convective_numerical_flux_dg(Ui, Uj, self.mesh.normal)

        # Assemble the explicit bilinear form, keeping in mind this is placed on the LHS.
        blf['U']['convection'] = -ngs.InnerProduct(F, ngs.grad(V)) * dV
        blf['U']['convection'] += ngs.InnerProduct(Fn, jumpV) * dS 

    def add_diffusion_form(self, blf: Integrals, lf: Integrals):

        if self.viscous_treatment is None:
            raise TypeError(f"Viscous configuration requires a treatment strategy.")

        self.viscous_treatment.add_diffusion_form(blf, lf)

    def add_boundary_conditions(self, blf: Integrals, lf:  Integrals):

        bnds = self.root.bcs.to_pattern()

        for bnd, bc in bnds.items():

            logger.debug(f"Adding boundary condition {bc} on boundary {bnd}.")

            if isinstance(bc, FarField):
                self.add_farfield_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, Outflow):
                self.add_outflow_formulation(blf, lf, bc, bnd)
   
            elif isinstance(bc, InterfaceBC):
                self.add_interface_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, Periodic):
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

    def add_farfield_formulation(self, blf: Integrals, lf: Integrals, bc: FarField, bnd: str):
        
        bonus = self.bonus_int_order['convection']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        U_infty = ngs.CF(
            (self.root.density(bc.fields),
             self.root.momentum(bc.fields),
             self.root.energy(bc.fields)))

        Ui = self.get_conservative_fields(U)
        Uj = self.get_conservative_fields(U.Other(bnd=U_infty))

        # # #
        # Inviscid treatment.
        # # 

        Fn = self.root.riemann_solver.get_convective_numerical_flux_dg(Ui, Uj, self.mesh.normal)
        
        Gamma_infty = ngs.InnerProduct(Fn, V)
        blf['U'][f"{bc.name}_{bnd}"] = Gamma_infty * dS

        # # # 
        # Viscous treatment.
        # # 
        
        if not self.root.dynamic_viscosity.is_inviscid:
            self.viscous_treatment.add_viscous_farfield_formulation(U_infty, blf, lf, bc, bnd)


    def add_outflow_formulation(self, blf: Integrals, lf: Integrals, bc: Outflow, bnd: str):
 
        # NOTE, I took this out of my thesis (Section 5.1.1.3). 
        # If you want I can reference it from somewhere else,
        # ... maybe NASA's FUN3D documentation?

        bonus = self.bonus_int_order['convection']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']
        
        n = bla.as_vector(self.mesh.normal)
        gamma = self.root.equation_of_state.heat_capacity_ratio
        
        # Internal data.
        Ui = self.get_conservative_fields(U)
        p = self.root.pressure(Ui)
        r = self.root.density(Ui)
        s = self.root.equation_of_state.specific_entropy(Ui)
        a = self.root.speed_of_sound(Ui)
        u = self.root.velocity(Ui)
        un = bla.inner(u, n)
        R = un + 2*a /(gamma-1) # Riemann invariant
        
        # Outlet specified values.
        pb = bc.fields.p
        
        # Extrapolate remaining quantities.
        rb = (pb/s)**(1/gamma)
        ab = ngs.sqrt( gamma * pb / rb )
        unb = R - 2*ab/(gamma-1)
        ub = u + (unb - un)*n
        rub = rb*ub
        rEb = pb/(gamma-1) + rb*(ub[0]**2 + ub[1]**2)/2

        # Reconstruct the conservative variables on the boundary.
        U_infty = ngs.CF( (rb, rub, rEb) )
        Uj = self.get_conservative_fields(U.Other(bnd=U_infty))

        # # #
        # Inviscid treatment.
        # # 

        Fn = self.root.riemann_solver.get_convective_numerical_flux_dg(Ui, Uj, self.mesh.normal)
        Gamma_infty = ngs.InnerProduct(Fn, V)
        blf['U'][f"{bc.name}_{bnd}"] = Gamma_infty * dS

        # # # 
        # Viscous treatment.
        # # 
        
        if not self.root.dynamic_viscosity.is_inviscid:
            self.viscous_treatment.add_viscous_outflow_formulation(U_infty, blf, lf, bc, bnd)

    def add_interface_formulation(self, blf: Integrals, lf: Integrals, bc: InterfaceBC, bnd: str):
        
        bonus = self.bonus_int_order['convection']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        # Form the boundary state, written as conservative variables.
        U_infty = ngs.CF(
            (self.root.density(bc.fields),
             self.root.momentum(bc.fields),
             self.root.energy(bc.fields)))

        Ui = self.get_conservative_fields(U)
        Uj = self.get_conservative_fields(U.Other(bnd=U_infty)) # Maybe not needed with bnd=U_infty?

        # # #
        # Inviscid treatment.
        # # 

        Fn = self.root.riemann_solver.get_convective_numerical_flux_dg(Ui, Uj, self.mesh.normal)
        
        Gamma_infty = ngs.InnerProduct(Fn, V)
        blf['U'][f"{bc.name}_{bnd}"] = Gamma_infty * dS
        
        # # # 
        # Viscous treatment.
        # # 
        
        if not self.root.dynamic_viscosity.is_inviscid:
            self.viscous_treatment.add_viscous_interface_formulation(blf, lf, bc, bnd)


