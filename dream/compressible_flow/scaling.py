r""" Definitions of dimensionless compressible flow equations.

This module defines the scaling for compressible flow equations, which are used to
scale the governing equations of compressible flow to dimensionless form. 

In the following, we derive the dimensionless compressible flow equations based on the choice
of three different scalings: **aerodynamic**, **acoustic**, and **aeroacoustic**. We denote the 
dimensionful and dimensionless variables with and without overline e.g. the density :math:`\overline{\rho}, \rho`, respectively.
Let us introduce the reference quantities used for the:

    .. list-table:: **Reference quantities**
        :widths: 15 20 20 20
        :header-rows: 1

        * - Quantity
          - Aerodynamic
          - Acoustic
          - Aeroacoustic 

        * - :math:`\rho_{ref}`
          - :math:`\overline{\rho}_\infty`
          - :math:`\overline{\rho}_\infty`
          - :math:`\overline{\rho}_\infty`

        * - :math:`u_{ref}`
          - :math:`|\overline{\vec{u}}_\infty|`
          - :math:`\overline{c}_\infty`
          - :math:`|\overline{\vec{u}}_\infty| + \overline{c}_\infty`

        * - :math:`T_{ref}`
          - :math:`\overline{T}_\infty \Ma_\infty^2 (\gamma - 1)`
          - :math:`\overline{T}_\infty (\gamma - 1)`
          - :math:`\overline{T}_\infty (1+\Ma_\infty)^2 (\gamma - 1)`

        * - :math:`L_{ref}`
          - :math:`\overline{L}`
          - :math:`\overline{L}`
          - :math:`\overline{L}`

From Buckingham :math:`\pi` theorem, the remaining quantitites are all functions of the reference quantities above:

    .. list-table::
        :widths: 15 20 20 20

        * - :math:`p_{ref}`
          - :math:`\rho_{ref} u_{ref}^2`
          - :math:`\rho_{ref} u_{ref}^2`
          - :math:`\rho_{ref} u_{ref}^2`

        * - :math:`t_{ref}`
          - :math:`\frac{L_{ref}}{u_{ref}}`
          - :math:`\frac{L_{ref}}{u_{ref}}`
          - :math:`\frac{L_{ref}}{u_{ref}}`

        * - :math:`\rho E_{ref}`
          - :math:`\rho_{ref} u_{ref}^2`
          - :math:`\rho_{ref} u_{ref}^2`
          - :math:`\rho_{ref} u_{ref}^2`

With these reference quantities, we can now start deriving the dimensionless compressible flow equations.

Continuity equation
    .. math::

        \frac{\partial \overline{\rho}}{\partial \overline{t}} + \overline{\div}(\overline{\rho \vec{u}}) &= 0, \\
        \frac{\rho_{ref}}{t_{ref}}\frac{\partial \rho}{\partial t} + \frac{\rho_{ref} u_{ref}}{L_{ref}} \div(\rho \vec{u}) &= 0, \\
        \frac{\partial \rho}{\partial t} + \div(\rho \vec{u}) &= 0.

Momentum equation
    .. math::

        \frac{\partial \overline{\rho \vec{u}}}{\partial \overline{t}} + \overline{\div}(\overline{\rho \vec{u}} \otimes \overline{\vec{u}} + \overline{p} \mat{I} - \overline{\mat{\tau}})&= 0, \\
        \frac{\rho_{ref} u_{ref}}{t_{ref}}\frac{\partial \rho \vec{u}}{\partial t} +  \frac{\rho_{ref} u_{ref}^2}{L_{ref}} \div(\rho \vec{u} \otimes \vec{u} + p \mat{I} - \mat{\tau})&= 0, \\
        \frac{\partial \rho \vec{u}}{\partial t} +  \div(\rho \vec{u} \otimes \vec{u} + p \mat{I} - \mat{\tau})&= 0.

Energy equation
    .. math::

        \frac{\partial \overline{\rho E}}{\partial \overline{t}} + \overline{\div}((\overline{\rho E}  + \overline{p}) \overline{\vec{u}}  - \overline{\mat{\tau}}\overline{\vec{u}} + \overline{\vec{q}}) &= 0, \\
        \frac{\rho E_{ref}}{t_{ref}}\frac{\partial \rho E}{\partial t} + \frac{\rho_{ref} u^3_{ref}}{L_{ref}} \div((\rho E  + p) \vec{u}  - \mat{\tau}\vec{u} + \vec{q}) &= 0, \\
        \frac{\partial \rho E}{\partial t} + \div((\rho E  + p) \vec{u}  - \mat{\tau}\vec{u} + \vec{q}) &= 0.
"""
from __future__ import annotations
import typing
import ngsolve as ngs

from dream.config import Configuration

if typing.TYPE_CHECKING:
    from .solver import CompressibleFlowSolver


class Scaling(Configuration, is_interface=True):
    """ Base class for compressible flow scalings. """

    root: CompressibleFlowSolver

    @property
    def dim_fields(self):

        if self.root.dimensional_fields is None:
            raise ValueError("Dimensional fields have to be set to access dimensional quantities.")

        return self.root.dimensional_fields

    @property
    def density(self) -> float:
        r""" Returns the dimensionless farfield density. 

            .. math::
                \rho_\infty = 1
        """
        return 1.0

    @property
    def reference_density(self) -> ngs.CF:
        r""" Returns the reference density.

            .. math::
                \rho_{ref} = \overline{\rho}_\infty
        """
        return self.dim_fields.rho
    
    @property
    def reference_velocity(self) -> ngs.CF:
        r""" Returns the reference velocity. """
        raise NotImplementedError("reference_velocity has to be implemented in subclass.")

    @property
    def reference_pressure(self) -> ngs.CF:
        r""" Returns the reference pressure.

            .. math::
                \p_{ref} = \rho_{ref} u^2_{ref}
        """
        return self.dim_fields.rho * self.reference_velocity**2
    
    @property
    def reference_time_scale(self) -> ngs.CF:
        r""" Returns the reference time.

            .. math::
                t_{ref} = \frac{L_{ref}}{u_{ref}}
        """
        return self.dim_fields.L / self.reference_velocity


class Aerodynamic(Scaling):

    name = "aerodynamic"

    @property
    def reference_velocity(self) -> ngs.CF:
        r""" Returns the reference velocity. 

            .. math::
                u_{ref} = |\overline{\vec{u}}_\infty|
        """
        return self.dim_fields.u

    @property
    def velocity(self):
        r""" Returns the dimensionless farfield velocity. 

            .. math::
                |\vec{u}_\infty| = 1
        """
        return 1.0

    @property
    def reference_temperature(self) -> ngs.CF:
        r""" Returns the reference temperature. 

            .. math::
                T_{ref} = \overline{T}_\infty \Ma^2_\infty (\gamma - 1)
        """
        gamma = self.root.equation_of_state.heat_capacity_ratio
        return self.dim_fields.T * self.root.mach_number**2 * (gamma - 1)

    @property
    def temperature(self) -> ngs.CF:
        r""" Returns the dimensionless farfield temperature. 

            .. math::
                T_\infty = \frac{1}{\Ma^2_\infty (\gamma - 1)}
        """
        gamma = self.root.equation_of_state.heat_capacity_ratio
        return 1/(self.root.mach_number**2 * (gamma - 1))

    @property
    def pressure(self) -> ngs.CF:
        r""" Returns the dimensionless farfield pressure.

            .. math::
                p_\infty = \frac{1}{\Ma^2_\infty \gamma}
        """
        gamma = self.root.equation_of_state.heat_capacity_ratio
        return 1/(self.root.mach_number**2 * gamma)

    @property
    def reference_reynolds_number(self) -> ngs.CF:
        r""" Returns the reference Reynolds number. 

            .. math:: 
                \Re_{ref} = \Re_\infty
        """
        return self.root.reynolds_number

    @property
    def speed_of_sound(self) -> float:
        r""" Returns the dimensionless farfield speed of sound. 

            .. math::
                c_\infty = \frac{1}{\Ma_\infty}
        """
        Ma = self.root.mach_number

        if Ma.Get() <= 0.0:
            raise ValueError("Aerodynamic scaling requires Mach number > 0.")

        return 1/Ma


class Acoustic(Scaling):

    name = "acoustic"

    @property
    def reference_velocity(self) -> ngs.CF:
        r""" Returns the reference velocity. 

            .. math::
                u_{ref} = \overline{c}_\infty
        """
        return self.dim_fields.c

    @property
    def velocity(self):
        r""" Returns the dimensionless farfield velocity. 

            .. math::
                |\vec{u}_\infty| = \Ma_\infty
        """
        return self.root.mach_number

    @property
    def reference_temperature(self) -> ngs.CF:
        r""" Returns the reference temperature. 

            .. math::
                T_{ref} = \overline{T}_\infty (\gamma - 1)
        """
        gamma = self.root.equation_of_state.heat_capacity_ratio
        return self.dim_fields.T * (gamma - 1)

    @property
    def temperature(self) -> ngs.CF:
        r""" Returns the dimensionless farfield temperature. 

            .. math::
                T_\infty = \frac{1}{(\gamma - 1)}
        """
        gamma = self.root.equation_of_state.heat_capacity_ratio
        return 1/(gamma - 1)

    @property
    def pressure(self) -> ngs.CF:
        r""" Returns the dimensionless farfield pressure.

            .. math::
                p_\infty = \frac{1}{\gamma}
        """
        gamma = self.root.equation_of_state.heat_capacity_ratio
        return 1/gamma

    @property
    def reference_reynolds_number(self) -> ngs.CF:
        r""" Returns the reference Reynolds number. 

            .. math:: 
                \Re_{ref} = \frac{\Re_\infty}{\Ma_\infty}
        """
        return self.root.reynolds_number/self.root.mach_number

    @property
    def speed_of_sound(self) -> float:
        r""" Returns the dimensionless farfield speed of sound. 

            .. math::
                c_\infty = 1
        """
        return 1


class Aeroacoustic(Scaling):

    name = "aeroacoustic"

    @property
    def reference_velocity(self) -> ngs.CF:
        r""" Returns the reference velocity. 

            .. math::
                u_{ref} = |\overline{\vec{u}}_\infty| + \overline{c}_\infty
        """
        return self.dim_fields.u + self.dim_fields.c

    @property
    def velocity(self):
        r""" Returns the dimensionless farfield velocity. 

            .. math::
                |\vec{u}_\infty| = \frac{\Ma_\infty}{1 + \Ma_\infty}
        """
        return self.root.mach_number/(1 + self.root.mach_number)

    @property
    def reference_temperature(self) -> ngs.CF:
        r""" Returns the reference temperature. 

            .. math::
                T_{ref} = \overline{T}_\infty (1+ \Ma_\infty)^2 (\gamma - 1)
        """
        gamma = self.root.equation_of_state.heat_capacity_ratio
        return self.dim_fields.T * (1 + self.root.mach_number)**2 * (gamma - 1)

    @property
    def temperature(self) -> ngs.CF:
        r""" Returns the dimensionless farfield temperature. 

            .. math::
                T_\infty = \frac{1}{(1+\Ma_\infty)^2 (\gamma - 1)}
        """
        gamma = self.root.equation_of_state.heat_capacity_ratio
        return 1/((1 + self.root.mach_number)**2 * (gamma - 1))

    @property
    def pressure(self) -> ngs.CF:
        r""" Returns the dimensionless farfield pressure.

            .. math::
                p_\infty = \frac{1}{(1+\Ma_\infty)^2 \gamma}
        """
        gamma = self.root.equation_of_state.heat_capacity_ratio
        return 1/((1 + self.root.mach_number)**2 * gamma)

    @property
    def reference_reynolds_number(self) -> ngs.CF:
        r""" Returns the reference Reynolds number. 

            .. math:: 
                \Re_{ref} = \frac{\Re_\infty (1+\Ma_\infty)}{\Ma_\infty}
        """
        return self.root.reynolds_number * (1 + self.root.mach_number)/self.root.mach_number

    @property
    def speed_of_sound(self) -> float:
        r""" Returns the dimensionless farfield speed of sound. 

            .. math::
                c_\infty = \frac{1}{1 + \Ma_\infty}
        """
        return 1/(1 + self.root.mach_number)
