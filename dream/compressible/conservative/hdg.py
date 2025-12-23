""" Definitions of conservative hdg discretizations. """
from __future__ import annotations
import logging
import ngsolve as ngs
import typing
import dream.bla as bla

from dream.time import (TimeSchemes,
                        TransientRoutine,
                        PseudoTimeSteppingRoutine,
                        MultizoneIMEXTimeRoutine,
                        PredictorCorrectorIMEXRoutine)
from dream.config import dream_configuration, Integrals
from dream.mesh import SpongeLayer, PSpongeLayer, Periodic, Initial
from dream.compressible.config import (flowfields,
                                       ConservativeFiniteElementMethod,
                                       FarField,
                                       Outflow,
                                       InviscidWall,
                                       Symmetry,
                                       IsothermalWall,
                                       AdiabaticWall,
                                       InterfaceBC,
                                       Dirichlet,
                                       Force,
                                       CBC)

from .diffusion import ViscousTreatment, InteriorPenaltyHDG
from .time import ImplicitEuler, BDF2, BDF3, SDIRK22, SDIRK33, SDIRK43, SDIRK54, DIRK34_LDD, DIRK43_WSO2, IMEXRK_ARS443
from .diffusion import ViscousTreatment, StrainHeat, Gradient

logger = logging.getLogger(__name__)

# --- Finite Element Methods --- #


class ConservativeHDG(ConservativeFiniteElementMethod):
    r""" Conservative hybridizable Discontinuous Galerkin method for compressible flow.

    Find :math:`\left(\vec{U}_h,\hat{\vec{U}}_h, \vec{Q}_h \right) \in U_h \times \hat{U}_h \times Q_h` such that

    .. math::

        \sum_{T \in \mesh} \int_{T} \frac{\partial \vec{U}_h}{\partial t} \cdot \vec{V}_h \, d\bm{x} - \int_{T} \left(\vec{F}(\vec{U}_h) - \vec{G}(\vec{U}_h, \vec{Q}_h)\right)  : \grad{\vec{V}_h} \, d\bm{x}+ \int_{\partial T} (\hat{\vec{F}}_h - \hat{\vec{G}}_h) \vec{n} \cdot \vec{V}_h   \, d\bm{s}   & = 0, \\
        - \sum_{F \in \facets^{\text{int}}} \int_{F} \jump{(\hat{\vec{F}}_h - \hat{\vec{G}}_h) \vec{n}} \cdot \hat{\vec{V}}_h \, d\bm{s} + \sum_{F \in \facets^{\text{ext}}} \int_{F} \hat{\vec{\Gamma}}_h \cdot \hat{\vec{V}}_h  \, d\bm{s} & = 0, 

    for all :math:`\left(\vec{V}_h,\hat{\vec{V}}_h \right) \in U_h \times \hat{U}_h`. With the discrete spaces choosen as

    .. math::

        U_h       & := L^2\left( (0, t_{end}] ; \mathbb{P}^k(\mesh, \mathbb{R}^{d+2}) \right),   \\
        \hat{U}_h & := L^2\left( (0, t_{end}] ; \mathbb{P}^k(\facets, \mathbb{R}^{d+2}) \right). 

    In the formulation, :math:`\hat{\vec{\Gamma}}_h` represents the boundary operator. 

    :note: See :class:`ViscousTreatment` for the definition of :math:`\vec{Q}_h` and :math:`Q_h`.

    """

    name: str = "conservative_hdg"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            'viscous_treatment': None,
            'static_condensation': True,
        }

        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def scheme(self) -> ImplicitEuler | BDF2 | BDF3 | IMEXRK_ARS443 | SDIRK22 | SDIRK33 | SDIRK54 | DIRK34_LDD | DIRK43_WSO2:
        """ Time scheme for the HDG method depending on the choosen time routine.

            :getter: Returns the time scheme
            :setter: Sets the time scheme
        """
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: TimeSchemes) -> None:
        if isinstance(self.root.time, TransientRoutine):
            OPTIONS = [ImplicitEuler, BDF2, BDF3, IMEXRK_ARS443, SDIRK22, SDIRK33, SDIRK43, SDIRK54, DIRK34_LDD, DIRK43_WSO2]
        elif isinstance(self.root.time, PseudoTimeSteppingRoutine):
            OPTIONS = [ImplicitEuler, BDF2]
        elif isinstance(self.root.time, MultizoneIMEXTimeRoutine):
            OPTIONS = [ImplicitEuler, SDIRK22, SDIRK33, SDIRK43]
        elif isinstance(self.root.time, PredictorCorrectorIMEXRoutine):
            OPTIONS = [ImplicitEuler]
        else:
            raise TypeError("HDG method only supports transient, pseudo time stepping or multizone time routines!")
        self._scheme = self._get_configuration_option(scheme, OPTIONS, TimeSchemes)

    @dream_configuration
    def viscous_treatment(self) -> None | StrainHeat | Gradient | InteriorPenaltyHDG:
        """
        The viscous treatment to be used for the conservative HDG method.
        """
        return self._viscous_treatment

    @viscous_treatment.setter
    def viscous_treatment(self, value: str | ViscousTreatment):

        if value is None:
            self._viscous_treatment = None
            return

        OPTIONS = [StrainHeat, Gradient, InteriorPenaltyHDG]
        self._viscous_treatment = self._get_configuration_option(value, OPTIONS, ViscousTreatment)

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        order = self.order
        dim = self.mesh.dim + 2

        U = ngs.L2(self.mesh, order=order)
        Uhat = ngs.FacetFESpace(self.mesh, order=order)

        psponge_layers = self.root.dcs.get_psponge_layers()
        if psponge_layers:
            U = self.root.dcs.reduce_psponge_layers_order_elementwise(U, psponge_layers)
            Uhat = self.root.dcs.reduce_psponge_layers_order_facetwise(Uhat, psponge_layers)

        if Periodic in self.root.bcs:
            Uhat = ngs.Periodic(Uhat)

        fes['U'] = U**dim
        fes['Uhat'] = Uhat**dim

        if not self.root.dynamic_viscosity.is_inviscid:
            self.viscous_treatment.add_finite_element_spaces(fes)

    def add_convection_form(self, blf: Integrals, lf: Integrals):

        bonus = self.bonus_int_order['convection']
        dX = ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])

        mask = self.get_domain_boundary_mask()

        U, V = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_conservative_fields(U)
        Uhat = self.get_conservative_fields(Uhat)

        F = self.root.get_convective_flux(U)
        Fn = self.get_convective_numerical_flux(U, Uhat, self.mesh.normal)

        blf['U']['convection'] = -bla.inner(F, ngs.grad(V)) * ngs.dx(bonus_intorder=bonus['vol'])
        blf['U']['convection'] += bla.inner(Fn, V) * dX

        if self.root.dynamic_viscosity.is_inviscid:
            tau_cs = self.root.riemann_solver.get_simplified_convective_stabilisation_matrix_hdg(Uhat, self.mesh.normal)
            blf['Uhat']['convection'] = -mask * (tau_cs*U.U - Uhat.U) * Vhat * dX
        else:
            blf['Uhat']['convection'] = -mask * bla.inner(Fn, Vhat) * dX

    def add_diffusion_form(self, blf: Integrals, lf: Integrals):

        if self.viscous_treatment is None:
            raise TypeError(f"Viscous configuration requires a treatment strategy.")

        self.viscous_treatment.add_diffusion_form(blf, lf)

    def add_boundary_conditions(self, blf: Integrals, lf: Integrals):

        for bnd, bc in self.root.bcs.items():

            logger.debug(f"Adding boundary condition {bc} on boundary {bnd}.")

            if isinstance(bc, FarField):
                self.add_farfield_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, CBC):
                self.add_cbc_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, Outflow):
                self.add_outflow_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, (InviscidWall, Symmetry)):
                self.add_inviscid_wall_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, IsothermalWall):
                self.add_isothermal_wall_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, AdiabaticWall):
                self.add_adiabatic_wall_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, InterfaceBC):
                self.add_interface_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, Dirichlet):
                self.add_dirichlet_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, Periodic):
                continue

            else:
                raise TypeError(f"Boundary condition {bc} not implemented in {self}!")

    def add_domain_conditions(self, blf: Integrals, lf: Integrals):

        for dom, dc in self.root.dcs.items():

            logger.debug(f"Adding domain condition {dc} on domain {dom}.")

            if isinstance(dc, SpongeLayer):
                self.add_sponge_layer_formulation(blf, lf, dc, dom)

            elif isinstance(dc, PSpongeLayer):
                self.add_psponge_layer_formulation(blf, lf, dc, dom)

            elif isinstance(dc, Force):
                self.add_forcing_formulation(blf, lf, dc, dom)

            elif isinstance(dc, Initial):
                continue

            else:
                raise TypeError(f"Domain condition {dc} not implemented in {self}!")

    def add_farfield_formulation(self, blf: Integrals, lf: Integrals, bc: FarField, bnd: str):
        r""" Implementation of the farfield boundary condition :class:`~dream.compressible.config.FarField`.

        On the boundary :math:`\Gamma` we solve :cite:`peraireHybridizableDiscontinuousGalerkin2010, vila-perezHybridisableDiscontinuousGalerkin2021`

        .. math::
            \int_{\Gamma} \left[ \widehat{\mat{A}}^+_n (\widehat{\vec{U}}_h - \vec{U}_h) - \widehat{\mat{A}}^-_n(\widehat{\vec{U}}_h - \vec{U}_\infty) \right] \cdot \widehat{\vec{V}}_h = \vec{0},

        where :math:`\widehat{\mat{A}}^\pm_n` are  the convective Jacobians in normal direction :math:`\vec{n}`.

        To increse the stability of the farfield condition on boundaries which are aligned with the flow,
        the identity Jacobian can be used instead of the convective Jacobian :cite:`PellmenreichCharacteristicBoundaryConditions2025`

        .. math::
            \int_{\Gamma} \left[\widehat{\vec{U}}_h - \frac{\vec{U}_h + \vec{U}_\infty}{2} - \widehat{\mat{Q}}_n \frac{\vec{U}_h - \vec{U}_\infty}{2} \right] \cdot \widehat{\vec{V}}_h  = \vec{0}.
        """

        bonus = self.bonus_int_order['convection']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        Uhat = self.get_conservative_fields(Uhat)

        U_infty = ngs.CF(
            (self.root.density(bc.fields),
             self.root.momentum(bc.fields),
             self.root.energy(bc.fields)))

        if bc.use_identity_jacobian:
            Qn = self.root.get_conservative_convective_identity(Uhat, self.mesh.normal, None)
            Gamma_infty = ngs.InnerProduct(Uhat.U - 0.5 * Qn * (U - U_infty) - 0.5 * (U + U_infty), Vhat)
        else:
            An_in = self.root.get_conservative_convective_jacobian(Uhat, self.mesh.normal, 'incoming')
            An_out = self.root.get_conservative_convective_jacobian(Uhat, self.mesh.normal, 'outgoing')
            Gamma_infty = ngs.InnerProduct(An_out * (Uhat.U - U) - An_in * (Uhat.U - U_infty), Vhat)

        blf['Uhat'][f"{bc.name}_{bnd}"] = Gamma_infty * dS

    def add_outflow_formulation(self, blf: Integrals, lf: Integrals, bc: Outflow, bnd: str):

        bonus = self.bonus_int_order['convection']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_conservative_fields(U)
        U_bc = flowfields(rho=U.rho, rho_u=U.rho_u, rho_Ek=U.rho_Ek, p=bc.fields.p)
        U_bc = ngs.CF((self.root.density(U_bc), self.root.momentum(U_bc), self.root.energy(U_bc)))

        Gamma_out = ngs.InnerProduct(Uhat - U_bc, Vhat)
        blf['Uhat'][f"{bc.name}_{bnd}"] = Gamma_out * dS

    def add_cbc_formulation(self, blf: Integrals, lf: Integrals, bc: CBC, bnd: str):

        bonus = self.bonus_int_order['convection']
        label = f"{bc.name}_{bnd}"
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_conservative_fields(U)
        Uhat = self.get_conservative_fields(Uhat)

        if bc.target == "farfield":
            U_bc = ngs.CF(
                (self.root.density(bc.fields),
                 self.root.momentum(bc.fields),
                 self.root.energy(bc.fields)))

        elif bc.target == "outflow":
            U_bc = flowfields(rho=U.rho, rho_u=U.rho_u, rho_Ek=U.rho_Ek, p=bc.fields.p)
            U_bc = ngs.CF((self.root.density(U_bc), self.root.momentum(U_bc), self.root.energy(U_bc)))

        elif bc.target == "mass_inflow":
            U_bc = flowfields(rho=bc.fields.rho, u=bc.fields.u, rho_Ek=bc.fields.rho_Ek, p=U.p)
            U_bc = ngs.CF((self.root.density(U_bc), self.root.momentum(U_bc), self.root.energy(U_bc)))

        elif bc.target == "temperature_inflow":
            rho_ = self.root.isentropic_density(U, bc.fields)
            U_bc = flowfields(rho=rho_, u=bc.fields.u, T=U.T)
            U_bc.Ek = self.root.specific_kinetic_energy(U_bc)
            U_bc = ngs.CF((self.root.density(U_bc), self.root.momentum(U_bc), self.root.energy(U_bc)))

        D = bc.get_relaxation_matrix(self.mesh.dim,
                                     dt=self.root.time.timer.step, c=self.root.speed_of_sound(Uhat),
                                     M=self.root.mach_number)
        D = self.root.transform_characteristic_to_conservative(D, Uhat, self.mesh.normal)

        beta = bc.tangential_relaxation
        Qn = self.root.get_conservative_convective_identity(Uhat, self.mesh.normal, None)
        Qin = self.root.get_conservative_convective_identity(Uhat, self.mesh.normal, "incoming")
        B = self.root.get_conservative_convective_jacobian(Uhat, self.mesh.tangential)

        dt = self.scheme.get_time_step(True)
        Uhat_n = self.scheme.get_current_level('Uhat', True)

        blf['Uhat'][label] = (Uhat.U - 0.5 * Qn * (U.U - Uhat_n) - 0.5 * (U.U + Uhat_n)) * Vhat * dS
        blf['Uhat'][label] -= dt * Qin * D * (U_bc - Uhat.U) * Vhat * dS
        blf['Uhat'][label] += dt * beta * Qin * B * (ngs.grad(Uhat.U) * self.mesh.tangential) * Vhat * dS

        if bc.is_viscous_fluxes:
            self.viscous_treatment.add_cbc_formulation(blf, lf, bc, bnd)

    def add_inviscid_wall_formulation(self, blf: Integrals, lf: Integrals, bc: InviscidWall, bnd: str):

        n = self.mesh.normal
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd))

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_conservative_fields(U)

        rho = self.root.density(U)
        rho_u = self.root.momentum(U)
        rho_E = self.root.energy(U)
        U_bc = ngs.CF((rho, rho_u - ngs.InnerProduct(rho_u, n)*n, rho_E))

        Gamma_inv = ngs.InnerProduct(Uhat - U_bc, Vhat)
        blf['Uhat'][f"{bc.name}_{bnd}"] = Gamma_inv * dS

    def add_isothermal_wall_formulation(self, blf: Integrals, lf: Integrals, bc: IsothermalWall, bnd: str):

        bonus = self.bonus_int_order['convection']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])

        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_conservative_fields(U)
        U_bc = flowfields(rho=U.rho, rho_u=tuple(0 for _ in range(self.mesh.dim)), rho_Ek=0, T=bc.fields.T)
        U_bc = ngs.CF((self.root.density(U_bc), self.root.momentum(U_bc), self.root.inner_energy(U_bc)))

        Gamma_iso = ngs.InnerProduct(Uhat - U_bc, Vhat)
        blf['Uhat'][f"{bc.name}_{bnd}"] = Gamma_iso * dS

    def add_adiabatic_wall_formulation(self, blf: Integrals, lf: Integrals, bc: AdiabaticWall, bnd: str):

        if self.viscous_treatment is None:
            raise TypeError(f"Adiabatic wall requires viscous treatment.")

        self.viscous_treatment.add_adiabatic_wall_formulation(blf, lf, bc, bnd)

    def add_interface_formulation(self, blf: Integrals, lf: Integrals, bc: InterfaceBC, bnd: str):
        bonus = self.bonus_int_order['convection']
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus['bnd'])

        Uhat, Vhat = self.TnT['Uhat']
        Uhat = self.get_conservative_fields(Uhat)

        U_infty = ngs.CF(
            (self.root.density(bc.fields),
             self.root.momentum(bc.fields),
             self.root.energy(bc.fields)))

        # This presumes an interface between an implicit (current) and an explicit (other) region.
        # Hence, it makes sense to utilize the solution obtained weakly in the explicit step as the BCs.
        # NOTE, even though that this is being imposed "strongly", it is still obtained in a weak manner.
        blf['Uhat'][f"{bc.name}_{bnd}"] = ngs.InnerProduct(Uhat.U - U_infty, Vhat) * dS

    def add_dirichlet_formulation(self, blf: Integrals, lf: Integrals, bc: Dirichlet, bnd: str):

        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd))
        Uhat, Vhat = self.TnT['Uhat']
        Ud = ngs.CF((self.root.density(bc.fields), self.root.momentum(bc.fields), self.root.energy(bc.fields)))

        Gamma_out = ngs.InnerProduct(Uhat - Ud, Vhat)
        blf['Uhat'][f"{bc.name}_{bnd}"] = Gamma_out * dS

    def add_sponge_layer_formulation(self, blf: Integrals, lf: Integrals, dc: SpongeLayer, dom: str):

        dX = ngs.dx(definedon=self.mesh.Materials(dom), bonus_intorder=dc.order)

        U, V = self.TnT['U']
        U_target = ngs.CF(
            (self.root.density(dc.target_state),
             self.root.momentum(dc.target_state),
             self.root.energy(dc.target_state)))

        blf['U'][f"{dc.name}_{dom}"] = dc.function * (U - U_target) * V * dX

    def add_psponge_layer_formulation(self, blf: Integrals, lf: Integrals, dc: PSpongeLayer, dom: str):

        dX = ngs.dx(definedon=self.mesh.Materials(dom), bonus_intorder=dc.order)

        U, V = self.TnT['U']

        if dc.is_equal_order:

            U_target = ngs.CF(
                (self.root.density(dc.target_state),
                 self.root.momentum(dc.target_state),
                 self.root.energy(dc.target_state)))

            Delta_U = U - U_target

        else:

            low_order_space = ngs.L2(self.mesh, order=dc.low_order)
            U_low = ngs.CF(tuple(ngs.Interpolate(proxy, low_order_space) for proxy in U))
            Delta_U = U - U_low

        blf['U'][f"{dc.name}_{dom}"] = dc.function * Delta_U * V * dX

    def add_forcing_formulation(self, blf: Integrals, lf: Integrals, dc: Force, dom: str):

        dX = ngs.dx(definedon=self.mesh.Materials(dom), bonus_intorder=dc.order)
        _, V = self.TnT['U']

        F = dc.get_force_vector(self.mesh.dim)
        if dc.is_constant:
            lf['U'][f"{dc.name}_{dom}"] = F * V * dX
        else:
            blf['U'][f"{dc.name}_{dom}"] = -F * V * dX

    def get_convective_numerical_flux(self, U: flowfields, Uhat: flowfields, unit_vector: ngs.CF):
        r"""
        Convective numerical flux

        .. math::

            \hat{\vec{F}}_h  \vec{n}^\pm  := \vec{F}(\hat{\vec{U}}_h) \vec{n}^\pm + \mat{\tau}_c(\hat{\vec{U}}_h) (\vec{U}_h - \hat{\vec{U}}_h)

        :note: See equation :math:`(E22a)` in :cite:`vila-perezHybridisableDiscontinuousGalerkin2021`.
        :note: See :class:`~dream.compressible.riemann_solver` for more details on the definition of :math:`\mat{\tau}_c`.
        """
        unit_vector = bla.as_vector(unit_vector)

        tau_c = self.root.riemann_solver.get_convective_stabilisation_matrix_hdg(Uhat, unit_vector)

        return self.root.get_convective_flux(Uhat) * unit_vector + tau_c * (U.U - Uhat.U)

    def get_solution_fields(self) -> flowfields:
        U = super().get_solution_fields()
        if self.viscous_treatment is not None:
            U.update(self.viscous_treatment.get_solution_fields())
        return U

    def initialize_time_scheme_gridfunctions(self, *spaces: str):

        SPACES = []
        if CBC in self.root.bcs:
            SPACES.append('Uhat')
        SPACES.extend(spaces)

        super().initialize_time_scheme_gridfunctions(*SPACES)

    def set_initial_conditions(self):

        U0 = self.mesh.MaterialCF({dom: ngs.CF(
            (self.root.density(dc.fields),
                self.root.momentum(dc.fields),
                self.root.energy(dc.fields))) for dom, dc in self.root.dcs.items(Initial)})
        bonus_int_order = max([dc.bonus_int_order for _, dc in self.root.dcs.items(Initial)])

        gfu = self.gfus['Uhat']
        fes = self.gfus['Uhat'].space
        u, v = fes.TnT()

        blf = ngs.BilinearForm(fes)
        blf += u * v * ngs.dx(element_boundary=True)

        f = ngs.LinearForm(fes)
        f += U0 * v * ngs.dx(element_boundary=True, bonus_intorder=bonus_int_order)

        with ngs.TaskManager():
            blf.Assemble()
            f.Assemble()
            gfu.vec.data = blf.mat.Inverse(freedofs=fes.FreeDofs(), inverse="sparsecholesky") * f.vec

        if self.viscous_treatment is not None:
            self.viscous_treatment.set_initial_conditions()

        super().set_initial_conditions()


class ConservativeDG_HDG(ConservativeHDG):

    name: str = "conservative_dg_hdg"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            'static_condensation': True,
        }

        DEFAULT.update(default)

        logger.warning("Conservative DG-HDG method is still experimental and may not be fully functional!")

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def scheme(self) -> IMEXRK_ARS443:
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: TimeSchemes) -> None:

        if not isinstance(self.root.time, TransientRoutine):
            raise TypeError("DG-HDG method only supports transient time routines!")

        OPTIONS = [IMEXRK_ARS443]
        self._scheme = self._get_configuration_option(scheme, OPTIONS, TimeSchemes)

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:

        order = self.root.fem.order
        dim = self.mesh.dim + 2

        super().add_finite_element_spaces(fes)
        fes['U'] = ngs.L2(self.mesh, order=order, dgjumps=True)**dim

    # In this (IMEX-)specialized class, the inviscid terms are handled via a standard DG.
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

        # Assemble the explicit bilinear form, keeping in mind this is also placed on the LHS.
        blf['U']['convection'] = -bla.inner(F, ngs.grad(V)) * ngs.dx(bonus_intorder=bonus['vol'])
        blf['U']['convection'] += mask*bla.inner(Fn, V) * ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])

    # In this (IMEX-)specialized class, the elliptic terms are handled via an HDG.
    def add_diffusion_form(self, blf: Integrals, lf: Integrals):

        bonus = self.bonus_int_order['diffusion']

        mask = self.get_domain_boundary_mask()

        U, V = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']
        Q, _ = self.viscous_treatment.TnT['Q']

        U = self.get_conservative_fields(U)
        Uhat = self.get_conservative_fields(Uhat)
        Q = self.root.fem.viscous_treatment.get_mixed_fields(Q)

        G = self.root.get_diffusive_flux(U, Q)
        Gn = self.viscous_treatment.get_diffusive_numerical_flux(U, Uhat, Q, self.mesh.normal)

        blf['U']['diffusion'] = ngs.InnerProduct(G, ngs.grad(V)) * ngs.dx(bonus_intorder=bonus['vol'])
        blf['U']['diffusion'] -= ngs.InnerProduct(Gn, V) * ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])
        blf['Uhat']['diffusion'] = mask * ngs.InnerProduct(Gn,
                                                           Vhat) * ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])

        # NOTE, to obtain a well-posed formulation, we require a value for rho_hat, since we need it on the facets.
        # To this end, we estimate its value as the average of the density on the surface (w.r.t. neighboring elements).
        # Recall, we solve for a Uhat implicitly, but use it explicitly in the next time step -- also note, rho is
        # solved for explicitly, as it's governed by a pure hyperbolic equation (continuity eq).
        rho = self.root.density(U)
        rhoHat = self.root.density(Uhat)
        rho_avg = rho - rhoHat
        eq = ngs.CF((rho_avg, 0, 0, 0))

        blf['Uhat']['test'] = mask * ngs.InnerProduct(eq,
                                                      Vhat) * ngs.dx(element_boundary=True, bonus_intorder=bonus['bnd'])

