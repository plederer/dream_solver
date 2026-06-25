r""" Dimensionless incompressible Navier-Stokes equations

We consider the dimensionless incompressible Navier-Stokes equations

.. math::
    \frac{\partial \vec{u}}{\partial t} + \vec{u} \cdot \nabla \vec{u} - \frac{1}{Re} \div{(\mat{\tau})} + \nabla p = 0

"""
from __future__ import annotations

import ngsolve as ngs
import dream.bla as bla

from dream.solver import SolverConfiguration
from dream.config import dream_configuration, equation
from dream.mesh import BoundaryConditions, DomainConditions

from .config import flowfields, BCS, DCS
from .viscosity import DynamicViscosity, Constant, Powerlaw
from .spatial import IncompressibleFiniteElement, TaylorHood, HDivHDG


class IncompressibleFlowSolver(SolverConfiguration):

    def __init__(self, mesh=None, **default):

        bcs = BoundaryConditions(mesh, BCS)
        dcs = DomainConditions(mesh, DCS)

        self._reynolds_number = ngs.Parameter(150.0)

        DEFAULT = {
            'reynolds_number': self._reynolds_number.Get(),
            'dynamic_viscosity': Constant(mesh, self),
            'convection': False,
        }
        DEFAULT.update(default)

        super().__init__(mesh=mesh, bcs=bcs, dcs=dcs, **DEFAULT)

    @dream_configuration
    def fem(self) -> TaylorHood:
        """ Returns the finite element method """
        return self._fem

    @fem.setter
    def fem(self, fem: IncompressibleFiniteElement) -> None:
        OPTIONS = [TaylorHood, HDivHDG]
        self._fem = self._get_configuration_option(fem, OPTIONS, IncompressibleFiniteElement)

    @dream_configuration
    def reynolds_number(self) -> ngs.Parameter:
        """ Returns the Reynolds number """
        return self._reynolds_number

    @reynolds_number.setter
    def reynolds_number(self, reynolds_number: ngs.Parameter) -> None:

        if isinstance(reynolds_number, ngs.Parameter):
            reynolds_number = reynolds_number.Get()

        if reynolds_number <= 0:
            raise ValueError("Invalid Reynolds number. Value has to be > 0!")

        self._reynolds_number.Set(reynolds_number)

    @dream_configuration
    def dynamic_viscosity(self) -> Constant | Powerlaw:
        r""" Sets the dynamic viscosity for the incompressible flow solver.

            :setter: Sets the dynamic viscosity, defaults to Constant
            :getter: Returns the dynamic viscosity
        """
        return self._dynamic_viscosity

    @dynamic_viscosity.setter
    def dynamic_viscosity(self, dynamic_viscosity: DynamicViscosity) -> None:
        OPTIONS = [Constant, Powerlaw]
        self._dynamic_viscosity = self._get_configuration_option(dynamic_viscosity, OPTIONS, DynamicViscosity)

    @dream_configuration
    def convection(self) -> bool:
        """ Returns the convection flag """
        return self._convection

    @convection.setter
    def convection(self, convection: bool) -> None:

        self._convection = bool(convection)

    def get_solution_fields(self, *fields, default_fields=True) -> flowfields:

        if default_fields:
            fields = ('velocity', 'pressure') + fields

        return super().get_solution_fields(*fields)

    @equation
    def velocity(self, u: flowfields):
        if u.u is not None:
            return u.u

    @equation
    def pressure(self, u: flowfields):
        if u.p is not None:
            return u.p

    @equation
    def kinematic_viscosity(self, u: flowfields):
        return self.dynamic_viscosity.kinematic_viscosity(u)

    @equation
    def deviatoric_stress_tensor(self, u: flowfields):

        if u.tau is not None:
            return u.tau
        else:
            Re = self.reynolds_number

            nu = self.kinematic_viscosity(u)
            strain = self.strain_rate_tensor(u)

            return 2 * nu/Re * strain

    @equation
    def strain_rate_tensor(self, u: flowfields):
        if u.eps is not None:
            return u.eps
        elif u.grad_u is not None:
            return 0.5 * (u.grad_u + u.grad_u.trans)

    @equation
    def drag_coefficient(
            self, u: flowfields, uinf: flowfields, drag_direction: tuple[float, ...] = (1, 0),
            aera: float = 1.0) -> float:
        r""" Returns the definition of the drag coefficient. 
             Needs to be integrated over a surface, due to the inclusion of the boundary normal vector :math:`\bm{n}_{bnd}`.

            .. math::
                C_d = \frac{1}{\frac{1}{2} \rho_\infty |\bm{u}_\infty|^2 A} \bm{n}_{drag} \left(\mat{\tau} - p \mat{\I} \right) \bm{n}_{bnd}

            :param U: A dictionary containing the flow quantities
            :type U: flowfields
            :param Uinf: A dictionary containing the reference flow quantities
            :type Uinf: flowfields
            :param drag_direction: A container containing the drag direction :math:`\bm{n}_{drag}`
            :type drag_direction: tuple[float, ...]
            :param aera: The reference area :math:`A`
            :type aera: float
            :return: The drag coefficient
            :rtype: float
        """
        return self._get_aerodynamic_coefficient(u, uinf, drag_direction, aera)

    @equation
    def lift_coefficient(
            self, u: flowfields,  uinf: flowfields, lift_direction: tuple[float, ...] = (0, 1),
            aera: float = 1.0) -> float:
        r""" Returns the definition of the lift coefficient. 
             Needs to be integrated over a surface, due to the inclusion of the boundary normal vector :math:`\bm{n}_{bnd}`.

            .. math::
                C_l = \frac{1}{\frac{1}{2} \rho_\infty |\bm{u}_\infty|^2 A} \bm{n}_{lift} \left(\mat{\tau} - p \mat{\I} \right) \bm{n}_{bnd} 

            :param U: A dictionary containing the flow quantities
            :type U: flowfields
            :param dU: A dictionary containing the gradients of the flow quantities for the evaluation of the viscous stress tensor
            :type dU: flowfields
            :param Uinf: A dictionary containing the reference flow quantities
            :type Uinf: flowfields
            :param lift_direction: A container containing the lift direction :math:`\bm{n}_{lift}`
            :type lift_direction: tuple[float, ...]
            :param aera: The reference area :math:`A`
            :type aera: float
            :return: The drag coefficient
            :rtype: float
        """
        return self._get_aerodynamic_coefficient(u, uinf, lift_direction, aera)

    def _get_aerodynamic_coefficient(
            self, u: flowfields, uref: flowfields, direction: tuple[float, ...],
            aera: float) -> float:

        sigma = -self.pressure(u) * ngs.Id(self.mesh.dim)
        if not self.dynamic_viscosity.is_inviscid:
            sigma += self.deviatoric_stress_tensor(u)

        return bla.inner(sigma * self.mesh.normal, bla.unit_vector(direction))/(0.5 * aera * uref.u**2)
