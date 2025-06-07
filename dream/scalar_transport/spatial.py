""" Definitions of the spatial discretizations for the scalar transport equation """
from __future__ import annotations
import logging
import ngsolve as ngs
import typing
import dream.bla as bla

from dream.time import TimeSchemes, TransientRoutine
from dream.config import Configuration, dream_configuration, Integrals
from dream.mesh import Periodic, Initial
from dream.solver import FiniteElementMethod
from dream.scalar_transport.config import (transportfields,
                                           FarField)

from .time import ImplicitEuler, BDF2, SDIRK22, SDIRK33, IMEXRK_ARS443, ExplicitEuler, SSPRK3, CRK4

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

    def get_time_scheme_spaces(self) -> dict[str, ngs.FESpace]:
        return {'U': self.root.fem.spaces['U']}

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]):
        raise NotImplementedError()

    def initialize_time_scheme_gridfunctions(self):
        spaces = self.get_time_scheme_spaces()
        super().initialize_time_scheme_gridfunctions(*spaces)
    
    def set_initial_conditions(self) -> None:
        U = self.mesh.MaterialCF({dom: self.root.phi(dc.fields) for dom, dc in self.root.dcs.to_pattern(Initial).items()})

        self.gfus['U'].Set(U)
        super().set_initial_conditions()

    def get_domain_boundary_mask(self) -> ngs.GridFunction:
        """ Returns a Gridfunction that is 0 on the domain boundaries and 1 on the domain interior.
        """
        fes = ngs.FacetFESpace(self.mesh, order=0)
        mask = ngs.GridFunction(fes, name="mask")
        mask.vec[:] = 0

        bnd_dofs = fes.GetDofs(self.mesh.Boundaries(self.root.bcs.get_domain_boundaries(True)))
        mask.vec[~bnd_dofs] = 1

        return mask

    def get_penalty_coefficient(self) -> ngs.CF:
        r""" Returns the (dimensionally-consistent) interior penalty coefficient. Its definition is

        :math:`\tau = \alpha \frac{(N+1)^2}{h}`,
        
        where N denotes the polynomial order and :math:`\alpha` is (user-specified) penalty constant from :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.interior_penalty_coefficient`

        """
        N = self.order
        h = self.mesh.meshsize
        alpha = self.interior_penalty_coefficient 
        tau = (alpha*(N+1)**2)/h
        return tau

    def get_fields(self, *fields: str, default: bool = True) -> transportfields:

        U = self.get_transported_fields(self.gfus['U'])
        fields_ = transportfields()
        return U


    def get_transported_fields(self, U: ngs.CoefficientFunction) -> transportfields:

        U_ = transportfields()
        U_.phi = U
  
        if isinstance(U, ngs.comp.ProxyFunction):
            U_.U = U

        return U_
    
    def add_symbolic_spatial_forms(self, blf: Integrals, lf: Integrals):
        r""" Defines the bilinear forms and linear functionals associated with an spatial formulation. 

        - See :func:`~dream.scalar_transport.spatial.HDG.add_convection_form` and :func:`~dream.scalar_transport.spatial.DG.add_convection_form` for the definition of the convection terms for the HDG and DG formulation, respectively. 
        - See :func:`~dream.scalar_transport.spatial.HDG.add_diffusion_form` and :func:`~dream.scalar_transport.spatial.DG.add_diffusion_form` for the implementation of the diffusion terms for the HDG and DG formulation, respectively. Note, :func:`~dream.scalar_transport.solver.ScalarTransportSolver.is_inviscid` must be False. 
        - See :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.add_boundary_conditions` for physical boundary condition imposition.
        - See :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.add_domain_conditions` for imposing domain conditions (e.g. initial conditions).
        """
        self.add_convection_form(blf, lf)

        if not self.root.is_inviscid:
            self.add_diffusion_form(blf, lf)

        self.add_boundary_conditions(blf, lf)
        self.add_domain_conditions(blf, lf)

    def add_boundary_conditions(self, blf: Integrals, lf: Integrals):
        r""" Adds specific boundary conditions to the blinear/linear forms.

        The available options are: :class:`FarField <dream.scalar_transport.config.FarField>` and :class:`Periodic`
        """
        bnds = self.root.bcs.to_pattern()

        for bnd, bc in bnds.items():

            logger.debug(f"Adding boundary condition {bc} on boundary {bnd}.")

            if isinstance(bc, FarField):
                self.add_farfield_formulation(blf, lf, bc, bnd)

            elif isinstance(bc, Periodic):
                continue

            else:
                raise TypeError(f"Boundary condition {bc} not implemented in {self}!")

    def add_domain_conditions(self, blf: Integrals, lf: Integrals):
        r""" Adds domain conditions. 
        """
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
    r""" Class containing the tools that define a hybridized discontinuous Galerkin (HDG) finite element method for the scalar transport equation.
    """
    name: str = "hdg"

    @dream_configuration
    def scheme(self) -> ImplicitEuler | BDF2 | SDIRK22 | SDIRK33:
        """ Time scheme for the HDG method depending on the choosen time routine.

            :getter: Returns the time scheme
            :setter: Sets the time scheme
        """
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: TimeSchemes) -> None:
        if isinstance(self.root.time, TransientRoutine):
            OPTIONS = [ImplicitEuler, BDF2, SDIRK22, SDIRK33]
        else:
            raise TypeError("HDG method only supports transient time stepping routines!")
        self._scheme = self._get_configuration_option(scheme, OPTIONS, TimeSchemes)

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:
        r""" Defines the spaces for the test and trial functions based on an HDG discretization. This also sets periodic boundaries on the facet space, in case periodic boundary conditions are prescribed.
        """
        order = self.root.fem.order

        U = ngs.L2(self.mesh, order=order)
        if self.mesh.dim == 1:
            Uhat = ngs.H1(self.mesh, order=1)
        else:
            Uhat = ngs.FacetFESpace(self.mesh, order=order)

        if self.root.bcs.has_condition(Periodic):
            Uhat = ngs.Periodic(Uhat)

        fes['U'] = U
        fes['Uhat'] = Uhat


    def add_convection_form(self, blf: Integrals, lf: Integrals):
        r""" Discretization (in the internal domain) of the convection terms using a standard Riemann solver. The `convective` bilinear form over an internal element is:
        
        .. math::
            B_c(\{u,\hat{u}\},v)       &= -\int\limits_{D} \bm{f}(u) \cdot \nabla v\, d\bm{x}\, 
                                        +  \hspace{-4mm} \int\limits_{\partial D \backslash \partial \Omega} f^*(u,\hat{u})\, v\, d\bm{s},\\
            B_c(\{u,\hat{u}\},\hat{v}) &= -\hspace{-4mm} \int\limits_{\partial D \backslash \partial \Omega} f^*(u, \hat{u})\, \hat{v}\, d\bm{s},

        where the physical (inviscid) flux is :math:`\bm{f}(u) = \bm{b} u` and the numerical flux :math:`f^* = f^*(u,\hat{u})`, over an element boundary, is defined based on the local solution :math:`u` and its neighboring facet variable :math:`\hat{u}`. See :func:`~dream.scalar_transport.riemann_solver.RiemannSolver.get_convective_numerical_flux_hdg` for details.
        """
        bonus = self.root.optimizations.bonus_int_order
        mask = self.get_domain_boundary_mask()

        U, V = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        U = self.get_transported_fields(U)
        Uhat = self.get_transported_fields(Uhat)

        F = self.root.get_convective_flux(U)
        Fn = self.root.riemann_solver.get_convective_numerical_flux_hdg(U, Uhat, self.mesh.normal)

        # For context, the terms are defined on the LHS.
        blf['U']['convection']  = -bla.inner(F, ngs.grad(V)) \
                                *  ngs.dx(bonus_intorder=bonus.vol)
        blf['U']['convection'] +=  bla.inner(Fn, V)          \
                                *  ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)
        
        # For context, the transmissibility equation is multiplied by a minus sign.
        blf['Uhat']['convection'] = -mask * bla.inner(Fn,Vhat) \
                                  *  ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)


    def add_diffusion_form(self, blf: Integrals, lf: Integrals):
        r""" HDG discretization (in the internal domain) of the elliptic terms using a symmetric interior penalty method. The `diffusion` bilinear form over an internal element is:
        
        .. math::
            \begin{aligned}
                B_d\big(\{u,\hat{u}\},v\big)       &=
                                                \int\limits_D \kappa \nabla u \cdot \nabla v \, d\bm{x}\,
                                               -\hspace{-4mm} \int\limits_{\partial D \setminus \partial \Omega} \kappa (\nabla u \cdot \bm{n}) v\, d\bm{s}\\
                                       &\qquad\qquad -\hspace{-4mm} \int\limits_{\partial D \setminus \partial \Omega} \kappa (u - \hat{u}) (\bm{n} \cdot \nabla v)\, d\bm{s}\,
                                               +\hspace{-4mm} \int\limits_{\partial D \setminus \partial \Omega} \kappa \tau (u - \hat{u}) v \, d\bm{s}, \\[1ex]
                B_d\big(\{u,\hat{u}\},\hat{v}\big) &=
                                                \hspace{-4mm} \int\limits_{\partial D \setminus \partial \Omega} \kappa (\nabla u \cdot \bm{n}) \hat{v}\, d\bm{s}\,
                                               -\hspace{-4mm} \int\limits_{\partial D \setminus \partial \Omega} \kappa \tau (u - \hat{u}) \hat{v}\, d\bm{s},
            \end{aligned}

        where :math:`\tau` is the interior penalty coefficient, see :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.get_penalty_coefficient`.
        """
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
        
        # Get the parameters needed for the penalty coefficient.
        tau = self.get_penalty_coefficient()

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
        r""" Initializes the solution by projecting the initial condition on both the volume elements and their facets.
        """
        super().set_initial_conditions()

        # The volume solution (U) has already been initialized in the base class, use it.
        U = self.gfus['U']

        gfu = self.gfus['Uhat']
        fes = self.gfus['Uhat'].space
        u, v = fes.TnT()

        blf = ngs.BilinearForm(fes)
        blf += u * v * ngs.dx(element_boundary=True)

        f = ngs.LinearForm(fes)
        f += U * v * ngs.dx(element_boundary=True)

        with ngs.TaskManager():
            blf.Assemble()
            f.Assemble()
            gfu.vec.data = blf.mat.Inverse(freedofs=fes.FreeDofs(), inverse="sparsecholesky") * f.vec

    def add_farfield_formulation(self, blf: Integrals, lf: Integrals, bc: FarField, bnd: str):
        r""" Imposes a farfield boundary condition using both convective and diffusive (if turned on) fluxes in an HDG formulation. The bilinear and linear forms associated with a farfield BC are:

        .. math::
            B^{\partial} = B_c^{\partial} + B_d^{\partial}, \qquad
            l^{\partial} = l_c^{\partial} + l_d^{\partial}.
       
        - The convection forms are defined as
        
        .. math:: 
            B_c^{\partial}\Big(\{u,\hat{u}\}, \hat{v}\Big) &= -\hspace{-4mm}\int\limits_{\partial D \cap \partial \Omega} \Big( (b_n^{+} + b_n^{-}) \hat{u} + b_n^{+} u\Big)\, \hat{v}\, d\bm{s},\\
            l_c^{\partial}(\hat{v})                        &= -\hspace{-4mm}\int\limits_{\partial D \cap \partial \Omega} b_n^{-} \bar{u}\, \hat{v}\, d\bm{s},
        
        with :math:`b_n^{+} = \max(b_n, 0)`, :math:`b_n^{-} = \min(b_n, 0)` and :math:`b_n = \bm{b} \cdot \bm{n}`.
        

        - The diffusion forms are defined as
        
        .. math::
            B_d^{\partial}\Big(\{u,\hat{u}\}, \hat{v}\Big) &= -\hspace{-4mm}\int\limits_{\partial D \cap \partial \Omega} \tau \kappa \hat{u}\, \hat{v}\, d\bm{s},\\
            l_d^{\partial}(\hat{v})                        &= -\hspace{-4mm}\int\limits_{\partial D \cap \partial \Omega} \tau \kappa \bar{u}\, \hat{v}\, d\bm{s},

        with :math:`\bar{u}` denoting the farfield boundary value we impose.

        """
        bonus = self.root.optimizations.bonus_int_order
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)
        
        U, _ = self.TnT['U']
        Uhat, Vhat = self.TnT['Uhat']

        # Extract the boundary value imposed.
        uin = bc.fields.phi
        
        # User must specify an inlet value.
        if uin is None:
            raise ValueError("Must impose a value for the solution!")
        
        # Convection term.
        bn = ngs.InnerProduct( self.root.convection_velocity, self.mesh.normal)
        bp = ngs.IfPos( bn, bn, 0.0 )
        bm = ngs.IfPos( bn, 0.0, bn )

        blf['Uhat']['convection'] += ngs.InnerProduct( -(bp+bm)*Uhat + bp*U, Vhat ) * dS
        lf['Uhat']['convection'] = -bm*uin * Vhat * dS 
        
        # Diffusion term (if diffusion enabled).
        if not self.root.is_inviscid: 
            k = self.root.diffusion_coefficient
            tau = self.get_penalty_coefficient()

            blf['Uhat']['diffusion'] -= ngs.InnerProduct( tau*k*Uhat, Vhat ) * dS
            lf['Uhat']['diffusion'] = -tau*k*uin * Vhat * dS 



class DG(ScalarTransportFiniteElementMethod):
    r""" Class containing the tools that define a standard discontinuous Galerkin (DG) finite element method for the scalar transport equation.
    """
    name: str = "dg"

    @dream_configuration
    def scheme(self) -> ImplicitEuler | BDF2 | SDIRK22 | SDIRK33 | IMEXRK_ARS443 | ExplicitEuler | SSPRK3 | CRK4:
        """ Time scheme for the DG method depending on the choosen time routine.

            :getter: Returns the time scheme
            :setter: Sets the time scheme
        """
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: TimeSchemes) -> None:
        if isinstance(self.root.time, TransientRoutine):
            OPTIONS = [ImplicitEuler, BDF2, SDIRK22, SDIRK33, IMEXRK_ARS443, ExplicitEuler, SSPRK3, CRK4]
        else:
            raise TypeError("DG method only supports transient time stepping routines!")
        self._scheme = self._get_configuration_option(scheme, OPTIONS, TimeSchemes)

    def add_finite_element_spaces(self, fes: dict[str, ngs.FESpace]) -> None:
        r""" Defines the spaces for the test and trial functions based on a DG discretization.
        """
        order = self.root.fem.order
        U = ngs.L2(self.mesh, order=order, dgjumps=True)
        fes['U'] = U
        
        # Issue an error, if static condensation is turned on.
        if self.root.optimizations.static_condensation is True:
            raise ValueError("Cannot have static condensation with a standard DG implementation!")
    
    def add_convection_form(self, blf: Integrals, lf: Integrals):
        r""" Implementation of the spatial discretization, in the internal domain, of the convective terms using a standard Riemann solver.
        
        .. math::
            B_c\big(u,v\big) = -\int\limits_{D} \bm{f}(u) \cdot \nabla v\, d\bm{x}\, 
                     + \hspace{-4mm} \int\limits_{\partial D \backslash \partial \Omega} f^*(u,\hat{u})\, v\, d\bm{s},

        where the numerical flux :math:`f^*` is defined based on a Riemann solver, see :func:`~dream.scalar_transport.riemann_solver.RiemannSolver.get_convective_numerical_flux_dg`.

        """
        bonus = self.root.optimizations.bonus_int_order
        mask = self.get_domain_boundary_mask()
                
        U, V = self.TnT['U']

        wind = self.root.convection_velocity

        Ui = self.get_transported_fields(U)
        Uj = self.get_transported_fields(U.Other())

        F = self.root.get_convective_flux(Ui)
        Fn = self.root.riemann_solver.get_convective_numerical_flux_dg(Ui, Uj, wind, self.mesh.normal)

        # For context, the terms are defined on the LHS.
        blf['U']['convection']  = -bla.inner(F, ngs.grad(V)) \
                                *  ngs.dx(bonus_intorder=bonus.vol)
        blf['U']['convection'] +=  mask*bla.inner(Fn, V)     \
                                *  ngs.dx(element_boundary=True, bonus_intorder=bonus.bnd)

    def add_diffusion_form(self, blf: Integrals, lf: Integrals):
        r""" DG discretization (in the internal domain) of the elliptic terms using a symmetric interior penalty method. The `diffusion` bilinear form over an internal element is:
        
        .. math::
            \begin{aligned}
                B_d\big(u,v\big) &= 
                                     \int\limits_D \kappa \nabla u \cdot \nabla v\, d\bm{x}\,
                                    -\hspace{-4mm} \int\limits_{\partial D \setminus \partial \Omega} \frac{\kappa}{2} (\nabla u_i + \nabla u_j) \cdot \bm{n}\, v\, d\bm{s}\\
                                 &\qquad\qquad 
                                    -\hspace{-4mm} \int\limits_{\partial D \setminus \partial \Omega} \frac{\kappa}{2} (u_i - u_j) (\nabla v_i + \nabla v_j) \cdot \bm{n}\, d\bm{s}\,
                                    +\hspace{-4mm} \int\limits_{\partial D \setminus \partial \Omega} \kappa \tau (u_i - u_j)(v_i - v_j)\, d\bm{s},
            \end{aligned}

        where the *ith* and *jth* subscripts correspond to the indices of the local solution and its neighobring solution, respectively. Additionally, :math:`\tau` is the interior penalty coefficient, see :func:`~dream.scalar_transport.spatial.ScalarTransportFiniteElementMethod.get_penalty_coefficient`.
        """

        bonus = self.root.optimizations.bonus_int_order
        mask = self.get_domain_boundary_mask()

        U, V = self.TnT['U']

        # Some abbreviations for readability.
        k = self.root.diffusion_coefficient
        n = self.mesh.normal
        dUi = ngs.grad(U)
        dUj = ngs.grad(U.Other())
        dVi = ngs.grad(V)
        dVj = ngs.grad(V.Other())
       
        # Get the parameters needed for the penalty coefficient.
        tau = self.get_penalty_coefficient()

        surf = 0.5*k*( dUi + dUj )*n
        symm = 0.5*k*( dVi + dVj )*n
        ujmp = U - U.Other()
        vjmp = V - V.Other()
        ipen = tau*k*ujmp

        # For context, the terms are defined on the LHS.
        blf['U']['diffusion']  = ngs.InnerProduct(k*dUi, dVi)      \
                               * ngs.dx(bonus_intorder=bonus.vol)
        blf['U']['diffusion'] -= mask*ngs.InnerProduct(symm, ujmp) \
                               * ngs.dx(skeleton=True, bonus_intorder=bonus.bnd) 
        blf['U']['diffusion'] -= mask*ngs.InnerProduct(surf, vjmp) \
                               * ngs.dx(skeleton=True, bonus_intorder=bonus.bnd)
        blf['U']['diffusion'] += mask*ngs.InnerProduct(ipen, vjmp) \
                               * ngs.dx(skeleton=True, bonus_intorder=bonus.bnd)

    def add_farfield_formulation(self, blf: Integrals, lf: Integrals, bc: FarField, bnd: str):
        r""" Imposes a farfield boundary condition using both convective and diffusive (if turned on) fluxes in a DG formulation. The bilinear and linear forms associated with a farfield BC are:

        .. math::
            B^{\partial} = B_c^{\partial} + B_d^{\partial}, \qquad
            l^{\partial} = l_c^{\partial} + l_d^{\partial}.
       
        - The convection forms are defined as
        
        .. math:: 
            B_c^{\partial}\big(u, v\big) &=  \hspace{-4mm}\int\limits_{\partial D \cap \partial \Omega} \frac{1}{2} (b_n + |b_n|) u\, v\, d\bm{s},\\
            l_c^{\partial}(v)            &= -\hspace{-4mm}\int\limits_{\partial D \cap \partial \Omega} \frac{1}{2} (b_n - |b_n|) \hat{u}\, v\, d\bm{s}.
        

        - The diffusion forms are defined as
        
        .. math::
            B_d^{\partial}\big(u, v\big) &= \hspace{-4mm}\int\limits_{\partial D \cap \partial \Omega} \tau \kappa u\, v\, d\bm{s},\\
            l_d^{\partial}(v)            &= \hspace{-4mm}\int\limits_{\partial D \cap \partial \Omega} \tau \kappa \bar{u}\, v\, d\bm{s},

        with :math:`\bar{u}` denoting the farfield boundary value we impose.

        """
        bonus = self.root.optimizations.bonus_int_order
        dS = ngs.ds(skeleton=True, definedon=self.mesh.Boundaries(bnd), bonus_intorder=bonus.bnd)
        U, V = self.TnT['U']
        n  = self.mesh.normal
        
        # Extract the boundary value imposed.
        uin = bc.fields.phi

        # User must specify an inlet value.
        if uin is None:
            raise ValueError("Must impose a value for the solution!")
        
        # Convection term.
        bn = ngs.InnerProduct( self.root.convection_velocity, self.mesh.normal) 
        bp = 0.5*( bn + bla.abs(bn) )
        bm = 0.5*( bn - bla.abs(bn) )

        blf['U']['convection'] += ngs.InnerProduct( bp*U, V ) * dS
        lf['U']['convection'] = -bm*uin * V * dS

        # Diffusion term (if diffusion enabled).
        if not self.root.is_inviscid:
            k = self.root.diffusion_coefficient
            tau = self.get_penalty_coefficient()
                
            blf['U']['diffusion'] += ngs.InnerProduct( tau*k*U, V ) * dS
            lf['U']['diffusion'] = tau*k*uin*V * dS


 
 




