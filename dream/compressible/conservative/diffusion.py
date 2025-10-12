""" Definitions of diffusion formulations. """
from __future__ import annotations
import logging
import ngsolve as ngs
import dream.bla as bla

from dream.config import Configuration, dream_configuration, Integrals
from dream.compressible.config import (flowfields,
                                       ConservativeFiniteElementMethod,
                                       FarField)
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


# # #
# Generic Interior penalty Method   #
# # # # # # # # # # # # # # # # # # #

class InteriorPenalty(ViscousTreatment, is_interface=True):
    
    name: str = "interior_penalty"

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
        return self.interior_penalty_coefficient * (self.fem.order + 1)**2 / self.mesh.meshsize

    def add_viscous_farfield_formulation(self, U_infty: ngs.CF, 
                                         blf: Integrals, lf: Integrals, bc: FarField, bnd: str):
        raise NotImplementedError(f"Function must be implemented in a derived class.")

    def get_frozen_diffusion_matrices_conservative(self, U: flowfields):
        
        # Get the relevant nondimensional numbers.
        Re = self.root.scaling.reference_reynolds_number
        Pr = self.root.prandtl_number

        # Extract the velocities.
        vel = self.root.velocity(U)
        u = vel[0]
        v = vel[1]
        
        # Extract the density and energies.
        rho = self.root.density(U)
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

    def get_frozen_diffusion_matrices_conservative_transposed(self, U: flowfields):
        KU11, KU12, KU21, KU22 = self.get_frozen_diffusion_matrices_conservative(U)
        return KU11.trans, KU12.trans, KU21.trans, KU22.trans

    def get_diffusive_flux_from_conservative_jump(self, U: flowfields, Ujump: ngs.CF, unit_vector: ngs.CF) -> ngs.CF:
        r""" Returns the conservative diffusive flux from given states and jump in the conservative variables along the unit normal vector.

            .. math::
                \bm{G}(\bm{U}, \jump{U} \otimes \bm{n})

            :param U: A dictionary containing the flow quantities
            :type U: flowfields
            :param dU: A CoefficientFunction containing the jump in the conservative variables
            :type dU: CoefficientFunction
        """

        dim = unit_vector.dim
        unit_vector = bla.as_vector(unit_vector)

        U.rho = self.root.density(U)
        U.rho_u = self.root.momentum(U)
        U.rho_E = self.root.energy(U)
        U.u = self.root.velocity(U)
        U.p = self.root.pressure(U)
                
        Ujump = bla.outer(Ujump, unit_vector)
        Ujump = flowfields(grad_rho=Ujump[0, :], grad_rho_u=ngs.CF(tuple(Ujump[i, :] for i in range(1, dim+1)), dims=(dim, dim)), grad_rho_E=Ujump[dim+1, :])
                
        Ujump.grad_u = self.root.velocity_gradient(U, Ujump)
        Ujump.grad_rho_Ek = self.root.kinetic_energy_gradient(U, Ujump)
        Ujump.grad_rho_Ei = self.root.inner_energy_gradient(U, Ujump)
        Ujump.grad_p = self.root.pressure_gradient(U, Ujump)
        Ujump.grad_T = self.root.temperature_gradient(U, Ujump)

        return self.root.get_diffusive_flux(U, Ujump)

    def get_surface_viscous_flux_from_linearized_state(self, Uhat: flowfields, gradU: ngs.CF) -> ngs.CF:
        KU11, KU12, KU21, KU22 = self.get_frozen_diffusion_matrices_conservative(Uhat)
        dUdx = gradU[:, 0]; nx = self.mesh.normal[0] 
        dUdy = gradU[:, 1]; ny = self.mesh.normal[1]

        # This returns:  F_n = n_i * K_{ij} * partial_j U.
        return nx*(KU11*dUdx + KU12*dUdy) + ny*(KU21*dUdx + KU22*dUdy)

    def add_diffusion_form(self, blf: Integrals, lf: Integrals):

        bonus = self.fem.bonus_int_order['diffusion']
        dV = ngs.dx(bonus_intorder=bonus['vol'])
        U, V = self.TnT['U']

        # Local solution, which includes the gradient.
        U = self.fem.get_conservative_fields(U, with_gradients=True)

        # Compute the flux of the solution, needed for the volume terms.
        G = self.root.get_diffusive_flux(U, U)

        # Add the volume term.
        blf['U']['diffusion_vol'] = ngs.InnerProduct(G, ngs.grad(V)) * dV

        # Add the remaining surface, symmetrizing and penalty terms.
        self.add_surface_term(blf, lf)
        self.add_symmetrizing_term(blf, lf)
        self.add_penalizing_term(blf, lf)


# # # 
# Interior penalty for Hybridized Discontinuous Galerkin    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class InteriorPenaltyHDG(InteriorPenalty):

    def add_surface_term(self, blf: Integrals, lf: Integrals) -> None:

        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        
        # Extract a nonzero mask for interior faces.
        mask = self.root.fem.get_domain_boundary_mask()

        # Convert the facet solution to flowfields.
        Uhat = self.fem.get_conservative_fields(Uhat)

        # Form the surface term.
        surf = self.get_surface_viscous_flux_from_linearized_state( Uhat, ngs.grad(U) )
        blf['U']['diffusion_surf'] = -ngs.InnerProduct(surf, V) * dS
        blf['Uhat']['diffusion_surf'] = mask * ngs.InnerProduct(surf, Vhat) * dS

    def add_symmetrizing_term(self, blf: Integrals, lf: Integrals) -> None:

        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']
        Uhat, _ = self.TnT['Uhat']

        # Get the gradient of the test functions and decompose it in x and y-directions.
        gradV = ngs.grad(V)
        dVdx = gradV[:, 0]
        dVdy = gradV[:, 1]

        # Extract the components of the unit normal vector.
        nx = self.mesh.normal[0]
        ny = self.mesh.normal[1]

        # Define the jump of the solution.
        jumpU = U - Uhat
        
        # Convert the facet solution to flowfields.
        Uhat = self.fem.get_conservative_fields(Uhat)

        # Form the term: G^T * grad(V). Notice, G is evaluated at Uhat, ie. G=G(Uhat).
        KU11T, KU12T, KU21T, KU22T = self.get_frozen_diffusion_matrices_conservative_transposed(Uhat)

        # Form the symmetrizing term.
        symm = nx * (KU11T*dVdx + KU21T*dVdy) + ny * (KU12T*dVdx + KU22T*dVdy) 
        blf['U']['diffusion_symm'] = -ngs.InnerProduct(symm, jumpU) * dS

    def add_penalizing_term(self, blf: Integrals, lf: Integrals) -> None:

        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        
        # Extract a nonzero mask for interior faces.
        mask = self.root.fem.get_domain_boundary_mask()

        # Extract the components of the unit normal vector.
        nx = self.mesh.normal[0]
        ny = self.mesh.normal[1]
        
        # Define the jump of the solution.
        jumpU = U - Uhat
        
        # Convert the solutions to flowfields.
        U = self.fem.get_conservative_fields(U)
        Uhat = self.fem.get_conservative_fields(Uhat)

        # Interior penalty coefficient.
        tau = self.get_scaled_penalty_coefficient() 

        # Get the diffusion matrices, based on the conservative gradients.
        KU11, KU12, KU21, KU22 = self.get_frozen_diffusion_matrices_conservative(Uhat)

        # Form the penalty term, which is based on the contraction of the diffusion matrices.
        KUij = nx * (KU11*nx + KU12*ny) + ny * (KU21*nx + KU22*ny)
        penn = tau * KUij * jumpU
        blf['U']['diffusion_penn'] = ngs.InnerProduct(penn, V) * dS
        blf['Uhat']['diffusion_penn'] = -mask * ngs.InnerProduct(penn, Vhat) * dS

    def add_viscous_farfield_formulation(self, U_infty: ngs.CF, 
                                         blf: Integrals, lf: Integrals, bc: FarField, bnd: str):
        
        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])
        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        
        # Choose the linearization state, needed for: K_{ij} = K_{ij}(Uhat).
        Uhat = self.fem.get_conservative_fields(Uhat)

        # # # 
        # Surface term.
        # # 

        # Compute the flux of the solution, which is a function of the solution and external.
        surf = self.get_surface_viscous_flux_from_linearized_state( Uhat, ngs.grad(U) )
        blf['Uhat'][f"{bc.name}_{bnd}_surf"] = ngs.InnerProduct(surf, Vhat) * dS

        # # # 
        # Penalty term.
        # # 

        # Interior penalty coefficient.
        tau = self.get_scaled_penalty_coefficient() 

        # Get the diffusion matrices, based on the conservative gradients.
        KU11, KU12, KU21, KU22 = self.get_frozen_diffusion_matrices_conservative(Uhat)

        # Form the penalty term, which is based on the contraction of the diffusion matrices.
        KUij = nx * (KU11*nx + KU12*ny) + ny * (KU21*nx + KU22*ny)
        penn = tau * KUij * jumpU
        blf['Uhat'][f"{bc.name}_{bnd}_penn"] = -ngs.InnerProduct(penn, Vhat) * dS


# # #
# Interior penalty for Standard Discontinuous Galerkin      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class InteriorPenaltySDG(InteriorPenalty):
    r""" This is based on the implementation in:

    Hartmann, R. and Houston, P., 2008. An optimal order interior penalty discontinuous Galerkin discretization of the compressible Navier–Stokes equations. Journal of Computational Physics, 227(22), pp.9670-9685.
    """
    
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
        blf['U']['diffusion_surf'] = -ngs.InnerProduct(surf, jumpV) * dS

    def add_symmetrizing_term(self, blf: Integrals, lf: Integrals) -> None:

        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.dx(skeleton=True, bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        # Extract the components of the unit normal vector.
        nx = self.mesh.normal[0]
        ny = self.mesh.normal[1]

        # Extract local and neighboring solution (with gradients).
        Ui = self.fem.get_conservative_fields(U)
        Uj = self.fem.get_conservative_fields(U.Other())

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
        FTi = nx * (KU11Ti*dVidx + KU21Ti*dVidy) + ny * (KU12Ti*dVidx + KU22Ti*dVidy) 
        FTj = nx * (KU11Tj*dVjdx + KU21Tj*dVjdy) + ny * (KU12Tj*dVjdx + KU22Tj*dVjdy)
  
        # Form the symmetrizing term.
        symm = (FTi + FTj) / 2 
        blf['U']['diffusion_symm'] = -ngs.InnerProduct(symm, jumpU) * dS

    def add_penalizing_term(self, blf: Integrals, lf: Integrals) -> None:

        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.dx(skeleton=True, bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        # Extract the components of the unit normal vector.
        nx = self.mesh.normal[0]
        ny = self.mesh.normal[1]

        # Extract local and neighboring solution (with gradients).
        Ui = self.fem.get_conservative_fields(U)
        Uj = self.fem.get_conservative_fields(U.Other())

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
        KUij = nx * (KU11*nx + KU12*ny) + ny * (KU21*nx + KU22*ny)
        penn = tau * KUij * jumpU
        blf['U']['diffusion_penn'] = ngs.InnerProduct(penn, jumpV) * dS
    
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
        blf['U'][f"{bc.name}_{bnd}_surf"] = -ngs.InnerProduct(surf, V) * dS

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
        symm = nx * (KU11T*dVidx + KU21T*dVidy) + ny * (KU12T*dVidx + KU22T*dVidy) 
  
        # Form the symmetrizing term.
        blf['U'][f"{bc.name}_{bnd}_symm"] = -ngs.InnerProduct(symm, jumpU) * dS

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
        KUij = nx * (KU11*nx + KU12*ny) + ny * (KU21*nx + KU22*ny)
        penn = tau * KUij * jumpU
        blf['U'][f"{bc.name}_{bnd}_penn"] = ngs.InnerProduct(penn, V) * dS

    def add_viscous_farfield_formulation(self, U_infty: ngs.CF, 
                                         blf: Integrals, lf: Integrals, bc: FarField, bnd: str):
        
        bonus = self.fem.bonus_int_order['diffusion']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']
        gradU = ngs.grad(U)

        # Choose the linearization state, needed for: K_{ij} = K_{ij}(Uhat).
        Uhat = self.fem.get_conservative_fields(U_infty)

        # # # 
        # Surface term.
        # # 

        # Form the surface term.
        surf = self.get_surface_viscous_flux_from_linearized_state(Uhat, gradU)
        blf['U'][f"{bc.name}_{bnd}_surf"] = -ngs.InnerProduct(surf, V) * dS

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
        KU11T, KU12T, KU21T, KU22T = self.get_frozen_diffusion_matrices_conservative_transposed(Uhat)

        # Form the term: {G^T * grad(V)}, for the local solution.
        symm = nx * (KU11T*dVdx + KU21T*dVdy) + ny * (KU12T*dVdx + KU22T*dVdy) 
  
        # Form the symmetrizing term.
        blf['U'][f"{bc.name}_{bnd}_symm"] = -ngs.InnerProduct(symm, jumpU) * dS

        # # # 
        # Penalty term.
        # # 

        # Interior penalty coefficient.
        tau = self.get_scaled_penalty_coefficient() 

        # Get the diffusion matrices, based on the conservative gradients.
        KU11, KU12, KU21, KU22 = self.get_frozen_diffusion_matrices_conservative(Uhat)

        # Form the penalty term, which is based on the contraction of the diffusion matrices.
        KUij = nx * (KU11*nx + KU12*ny) + ny * (KU21*nx + KU22*ny)
        penn = tau * KUij * jumpU
        blf['U'][f"{bc.name}_{bnd}_penn"] = ngs.InnerProduct(penn, V) * dS


