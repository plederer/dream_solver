""" Definitions of conservative dg discretizations. """
from __future__ import annotations
import logging
import ngsolve as ngs
import dream.bla as bla

from dream.time import (TimeSchemes,
                        TransientRoutine,
                        PseudoTimeSteppingRoutine,
                        MultizoneIMEXTimeRoutine)
from dream.config import Configuration, dream_configuration, Integrals
from dream.compressible.config import (flowfields,
                                       ConservativeFiniteElementMethod,
                                       FarField,
                                       Dirichlet,
                                       Outflow,
                                       Inflow,
                                       InviscidWall,
                                       Symmetry,
                                       AdiabaticWall,
                                       IsothermalWall,
                                       InterfaceBC,
                                       Periodic,
                                       Force,
                                       Initial)

from .diffusion import ViscousTreatment, InteriorPenaltySDG
from .time import ExplicitEuler, SSPRK3, CRK4, RK_ARS22, RK_ARS33, RK_ARS43, ImplicitEuler, BDF2

logger = logging.getLogger(__name__)


class ConservativeDG(ConservativeFiniteElementMethod):

    name: str = "conservative_dg"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {'viscous_treatment': None,}

        DEFAULT.update(default)

        logger.warning("Conservative DG method is still experimental and may not be fully functional!")

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def viscous_treatment(self) -> InteriorPenaltySDG | None:
        return self._viscous_treatment

    @viscous_treatment.setter
    def viscous_treatment(self, value: str | ViscousTreatment | None):
        
        if value is None:
            self._viscous_treatment = None
            return

        OPTIONS = [InteriorPenaltySDG]
        self._viscous_treatment = self._get_configuration_option(value, OPTIONS, ViscousTreatment)

    @dream_configuration
    def scheme(self) -> ExplicitEuler | SSPRK3 | CRK4 | RK_ARS22 | RK_ARS33 | RK_ARS43 | ImplicitEuler | BDF2:
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: TimeSchemes) -> None:

        if isinstance(self.root.time, TransientRoutine):
            OPTIONS = [ExplicitEuler, SSPRK3, CRK4, RK_ARS22, RK_ARS33, RK_ARS43, ImplicitEuler, BDF2]
        elif isinstance(self.root.time, PseudoTimeSteppingRoutine):
            OPTIONS = [ImplicitEuler, BDF2]
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

        for bnd, bc in self.root.bcs.items():

            logger.debug(f"Adding boundary condition {bc} on boundary {bnd}.")

            if isinstance(bc, (FarField, Dirichlet)):
                self.add_farfield_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, Outflow):
                self.add_outflow_formulation(blf, lf, bc, bnd)
             
            elif isinstance(bc, Inflow):
                self.add_inflow_formulation(blf, lf, bc, bnd)  

            elif isinstance(bc, InterfaceBC):
                self.add_interface_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, (InviscidWall, Symmetry)):
                self.add_inviscid_wall_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, AdiabaticWall):
                self.add_adiabatic_wall_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, IsothermalWall):
                self.add_isothermal_wall_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, Periodic):
                continue
         
            else:
                raise TypeError(f"Boundary condition {bc} not implemented in {self}!")

    def add_domain_conditions(self, blf: Integrals, lf:  Integrals):

        for dom, dc in self.root.dcs.items():

            logger.debug(f"Adding domain condition {dc} on domain {dom}.")

            if isinstance(dc, Initial):
                continue

            elif isinstance(dc, Force):
                self.add_forcing_formulation(blf, lf, dc, dom)

            else:
                raise TypeError(f"Domain condition {dc} not implemented in {self}!")

    def get_unique_farfield_state(self, Ui: flowfields, Uinf: flowfields):

        # Extract the normal and specific heat ratio.
        gamma = self.root.equation_of_state.heat_capacity_ratio
        normal = bla.as_vector(self.mesh.normal)
        nx = normal[0]
        ny = normal[1]

        # Compute local data variables.
        rho = self.root.density(Ui)
        vel = self.root.velocity(Ui)
        a = self.root.speed_of_sound(Ui)
        p = self.root.pressure(Ui)
        un = bla.inner(vel, normal)
        u = vel[0]
        v = vel[1]

        # Compute external data variables.
        rho_inf = self.root.density(Uinf)
        vel_inf = self.root.momentum(Uinf)
        p_inf = self.root.pressure(Uinf)
        u_inf = vel_inf[0]
        v_inf = vel_inf[1]

        # Compute the 3 distinct eigenvalues.
        lmb1 = un - a
        lmb2 = un
        lmb3 = un + a

        # Compute a diagonal entries: unity for outgoing characteristics, otherwise zero.
        d1 = ngs.IfPos(lmb1, 1, 0)
        d2 = ngs.IfPos(lmb2, 1, 0)
        d3 = d2
        d4 = ngs.IfPos(lmb3, 1, 0)
        
        # Jump in the primitive variables.
        drho = rho - rho_inf
        du   =   u - u_inf
        dv   =   v - v_inf
        dp   =   p - p_inf
        
        # Some abreviations.
        dun = nx*du + ny*dv
        nx2 = nx*nx
        ny2 = ny*ny
        nxny = nx*ny
        d1md4 = d1 - d4 
        d1pd4 = d1 + d4 
        rhoa  = rho*a
        ov2rhoa = 1/(2*rhoa)
        rov2a = rho/(2*a)
        a2    = a*a
        ova2  = 1/a2
        
        # Assemble the primitive boundary state.
        rho_b = rho_inf - rov2a*d1md4 * dun + d2 * drho + ova2*(d1pd4/2-d2) * dp 
        u_b   = u_inf + (nx2*d1pd4/2 + ny2*d3) * du + nxny*(d1pd4/2-d3) * dv - (nx*ov2rhoa*d1md4) * dp 
        v_b   = v_inf + (ny2*d1pd4/2 + nx2*d3) * dv + nxny*(d1pd4/2-d3) * du - (ny*ov2rhoa*d1md4) * dp 
        p_b   = p_inf + (d1pd4/2) * dp - (rhoa/2)*d1md4 * dun
        
        # Compute the conservative variables from the primitive ones on the boundary state.
        ru_b = rho_b*u_b
        rv_b = rho_b*v_b
        ek_b = (u_b*u_b + v_b*v_b)/2
        rE_b = p_b/(gamma-1) + rho_b*ek_b
        
        # Return the conservative boundary state.
        return ngs.CF( (rho_b, ru_b, rv_b, rE_b) )

    def add_farfield_formulation(self, blf: Integrals, lf: Integrals, bc: FarField, bnd: str):
        
        bonus = self.bonus_int_order['convection']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        U_infty = ngs.CF(
            (self.root.density(bc.fields),
             self.root.momentum(bc.fields),
             self.root.energy(bc.fields)))

        
        # # #
        # Inviscid treatment.
        # # 

        n = bla.as_vector(self.mesh.normal)
        Ui = self.get_conservative_fields(U)

        # Extract the characteristic matrix and assemble the boundary state from it.
        Dn = self.root.get_conservative_convective_identity(Ui, self.mesh.normal, None)
        U_farfield = 0.5 * Dn * (U - U_infty) + 0.5 * (U + U_infty)

        # Compute the farfield state via a flux-vector splitting strategy.
        #U_farfield = self.get_unique_farfield_state(Ui, bc.fields)
        Ubc = self.get_conservative_fields(U_farfield)
        Ubc.U = U_farfield

        # Evaluate the convective flux at the boundary, and accumulate its residual.
        F = self.root.get_convective_flux(Ubc)
        Gamma_infty = ngs.InnerProduct(F*n, V)
        blf['U'][f"{bc.name}_{bnd}"] = Gamma_infty * dS

        # # # 
        # Viscous treatment.
        # # 
        
        if not self.root.dynamic_viscosity.is_inviscid:
            self.viscous_treatment.add_standard_viscous_formulation(U_infty, blf, lf, bc, bnd)

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
        Uj = self.get_conservative_fields(U_infty)
        Uj.U = U_infty

        # # #
        # Inviscid treatment.
        # # 

        # NOTE, check if avoiding a numerical flux is better.
        Fn = self.root.riemann_solver.get_convective_numerical_flux_dg(Ui, Uj, self.mesh.normal)
        Gamma_infty = ngs.InnerProduct(Fn, V)
        blf['U'][f"{bc.name}_{bnd}"] = Gamma_infty * dS

        # # # 
        # Viscous treatment.
        # # 
        
        # NOTE, check if we even need to impose viscous conditions.
        if not self.root.dynamic_viscosity.is_inviscid:
            self.viscous_treatment.add_standard_viscous_formulation(U_infty, blf, lf, bc, bnd)


    def add_inflow_formulation(self, blf: Integrals, lf: Integrals, bc: Inflow, bnd: str):
 
        # NOTE, I took this out of my thesis (Section 5.1.1.1). 
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
        a = self.root.speed_of_sound(Ui)
        u = self.root.velocity(Ui)
        un = bla.inner(u, n)
        R = un + 2*a /(gamma-1) # Riemann invariant
        
        # Outlet specified values.
        rb = bc.density
        ub = bc.velocity
        ubn = bla.inner(ub, n)

        # Extrapolate remaining quantities.
        ab = (gamma-1) * (R - ubn)/2
        pb = (rb*ab*ab ) / gamma
        rub = rb*ub
        rEb = pb/(gamma-1) + rb*(ub[0]**2 + ub[1]**2)/2

        # Reconstruct the conservative variables on the boundary.
        U_infty = ngs.CF( (rb, rub, rEb) )
        Uj = self.get_conservative_fields(U_infty)
        Uj.U = U_infty

        # # #
        # Inviscid treatment.
        # # 

        # NOTE, check if avoiding a numerical flux is better.
        Fn = self.root.riemann_solver.get_convective_numerical_flux_dg(Ui, Uj, self.mesh.normal)
        Gamma_infty = ngs.InnerProduct(Fn, V)
        blf['U'][f"{bc.name}_{bnd}"] = Gamma_infty * dS

        # # # 
        # Viscous treatment.
        # # 
        
        # NOTE, check if we even need to impose viscous conditions.
        if not self.root.dynamic_viscosity.is_inviscid:
            self.viscous_treatment.add_standard_viscous_formulation(U_infty, blf, lf, bc, bnd)

    def add_inviscid_wall_formulation(self, blf: Integrals, lf: Integrals, bc: InviscidWall, bnd: str):
        if not self.root.dynamic_viscosity.is_inviscid:
            raise TypeError(f"Inviscid wall requires an inviscid formulation.")

        bonus = self.bonus_int_order['convection']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']
        n = self.mesh.normal

        Ui = self.get_conservative_fields(U)
        rho = self.root.density(Ui)
        rho_u = self.root.momentum(Ui)
        rho_E = self.root.energy(Ui)
        Ei = self.root.inner_energy(Ui)
        u = self.root.velocity(Ui)
        un = ngs.InnerProduct(u, n)
        
        # Get the wall values.
        u_wall =  u - un*n
        Ek_wall = rho * ngs.InnerProduct(u_wall, u_wall) / 2
        rhoE_wall = Ei + Ek_wall
        rhou_wall = rho*u_wall

        # Form the conservative values, based on the inviscid wall condition.
        Uw = ngs.CF((rho, rhou_wall, rhoE_wall))
        
        # Based on Hartmann's analysis when used with a Riemann solver, See Eq (12) and the paragraph below in:
        #  Hartmann, Ralf, and Tobias Leicht. 
        #  "Generalized adjoint consistent treatment of wall boundary conditions for compressible flows." 
        #  Journal of Computational Physics 300 (2015): 754-778. 
        Uw = 2*Uw -  U 
        Uj = self.get_conservative_fields(Uw)

        Fn = self.root.riemann_solver.get_convective_numerical_flux_dg(Ui, Uj, self.mesh.normal)
        Gamma_inv = ngs.InnerProduct(Fn, V)
        blf['U'][f"{bc.name}_{bnd}"] = Gamma_inv * dS

    def add_adiabatic_wall_formulation(self, blf: Integrals, lf: Integrals, bc: AdiabaticWall, bnd: str):
 
        if self.viscous_treatment is None:
            raise TypeError(f"Adiabatic wall requires viscous treatment.")

        bonus = self.bonus_int_order['convection']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        # Get the flowfields from the local (ith) and boundary (jth) states.
        Ui = self.get_conservative_fields(U)
        U_infty = self.viscous_treatment.get_adiabatic_boundary_state(Ui)
        Uj = self.get_conservative_fields(U_infty)
        
        # We use the Riemann solver, to impose the BC weakly.
        Fn = self.root.riemann_solver.get_convective_numerical_flux_dg(Ui, Uj, self.mesh.normal)
        blf['U'][f"{bc.name}_{bnd}_conv"] = ngs.InnerProduct(Fn, V) * dS
        
        # Proceed with the viscous treatment, based on its respective class.
        self.viscous_treatment.add_adiabatic_wall_formulation(blf, lf, bc, bnd)

    def add_isothermal_wall_formulation(self, blf: Integrals, lf: Integrals, bc: IsothermalWall, bnd: str):
        
        if self.viscous_treatment is None:
            raise TypeError(f"Isothermal wall requires viscous treatment.")

        bonus = self.bonus_int_order['convection']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])
        U, V = self.TnT['U']

        # Get the flowfields from the local (ith) and boundary (jth) states.
        Ui = self.get_conservative_fields(U)
        T_infty = bc.fields.T
        U_infty = self.viscous_treatment.get_isothermal_boundary_state(Ui, T_infty)
        Uj = self.get_conservative_fields(U_infty)
        
        # We use the Riemann solver, to impose the BC weakly.
        Fn = self.root.riemann_solver.get_convective_numerical_flux_dg(Ui, Uj, self.mesh.normal)
        blf['U'][f"{bc.name}_{bnd}_conv"] = ngs.InnerProduct(Fn, V) * dS
        
        # Proceed with the viscous treatment, based on its respective class.
        self.viscous_treatment.add_isothermal_wall_formulation(blf, lf, bc, bnd)

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

    def add_forcing_formulation(self, blf: Integrals, lf: Integrals, dc: Force, dom: str):

        dX = ngs.dx(definedon=self.mesh.Materials(dom), bonus_intorder=dc.order)
        dXe = ngs.dx(element_boundary=True, definedon=self.mesh.Materials(dom), bonus_intorder=dc.order)
        n = self.mesh.normal

        _, V = self.TnT['U']

        if any([dc._continuum, dc._momentum, dc._energy]):
            F = dc.get_force_vector(self.mesh.dim)
            lf['U'][f"{dc.name}_{dom}"] = F * V * dX
            
        elif dc._flux is not None:
            lf['U'][f"{dc.name}_{dom}"] = -ngs.InnerProduct(dc._flux,  ngs.grad(V)) * dX
            lf['U'][f"{dc.name}_{dom}"] += ngs.InnerProduct(dc._flux * n,  V) * dXe

