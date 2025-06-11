""" Definitions of boundary/domain conditions for compressible flow """
from __future__ import annotations
import ngsolve as ngs

from dream import bla
from dream.config import quantity, dream_configuration, ngsdict
from dream.solver import FiniteElementMethod
from dream.mesh import (Condition,
                        Periodic,
                        Initial,
                        Force,
                        Perturbation,
                        SpongeLayer,
                        PSpongeLayer,
                        GridDeformation
                        )


class CompressibleFiniteElementMethod(FiniteElementMethod):

    def set_boundary_conditions(self) -> None:
        """ Boundary conditions for compressible flows are set weakly. Therefore we do nothing here."""
        pass


class flowfields(ngsdict):
    """ Mutable mapping for flow quantities.

        Literal mathematical symbols as key names are converted to their respective quantities,
        if predefined. Values are converted to NGSolve CoefficientFunctions.

        >>> fields = flowfields(rho=1.0, velocity=(1.0, 0.0))
        >>> fields
        {'density': CoefficientFunction((1.0)), 'velocity': CoefficientFunction((1.0, 0.0))}
        >>> fields['density'] = 5.0
        {'density': CoefficientFunction((5.0)), 'velocity': CoefficientFunction((1.0, 0.0))}
    """
    rho = quantity('density', r"\rho")
    u = quantity('velocity', r"\bm{u}")
    rho_u = quantity('momentum', r"\rho \bm{u}")
    p = quantity('pressure', r"p")
    T = quantity('temperature', r"T")
    rho_E = quantity('energy', r"\rho E")
    E = quantity('specific_energy', r"E")
    rho_Ei = quantity('inner_energy', r"\rho E_i")
    Ei = quantity('specific_inner_energy', r"E_i")
    rho_Ek = quantity('kinetic_energy', r"\rho E_k")
    Ek = quantity('specific_kinetic_energy', r"E_k")
    rho_H = quantity('enthalpy', r"\rho H")
    H = quantity('specific_enthalpy', r"H")
    c = quantity('speed_of_sound', r"c")
    grad_rho = quantity('density_gradient', r"\nabla \rho")
    grad_u = quantity('velocity_gradient', r"\nabla \bm{u}")
    grad_rho_u = quantity('momentum_gradient', r"\nabla (\rho \bm{u})")
    grad_p = quantity('pressure_gradient', r"\nabla p")
    grad_T = quantity('temperature_gradient', r"\nabla T")
    grad_rho_E = quantity('energy_gradient', r"\nabla (\rho E)")
    grad_E = quantity('specific_energy_gradient')
    grad_rho_Ei = quantity('inner_energy_gradient')
    grad_Ei = quantity('specific_inner_energy_gradient')
    grad_rho_Ek = quantity('kinetic_energy_gradient')
    grad_Ek = quantity('specific_kinetic_energy_gradient')
    grad_rho_H = quantity('enthalpy_gradient')
    grad_H = quantity('specific_enthalpy_gradient')
    grad_c = quantity('speed_of_sound_gradient')
    eps = quantity('strain_rate_tensor', r"\bm{\varepsilon}")


class FarField(Condition):
    r""" Farfield condition for compressible flow.

    The farfield condition is used to set the subsonic and supersonic inflow/outflow conditions for compressible 
    flows in a characteristic way. It acts partially non-reflecting for acoustic waves on both inflow and outflow boundaries. 

    :param fields: Dictionary of flow quantities :math:`\vec{U}_\infty` to be set at the farfield boundaries.
    :type fields: flowfields
    :param use_identity_jacobian: Flag to use the identity jacobian for the farfield condition.
    :type use_identity_jacobian: bool

    :note: See :func:`~dream.compressible.conservative.spatial.HDG.add_farfield_formulation` for the implementation of 
              the farfield condition in the :class:`~dream.compressible.conservative.spatial.HDG` formulation.

    """

    name = "farfield"

    def __init__(self,
                 fields: flowfields | None = None,
                 use_identity_jacobian: bool = True):

        self.fields = fields
        self.use_identity_jacobian = use_identity_jacobian

        super().__init__()

    @dream_configuration
    def fields(self) -> flowfields:
        """ Returns the fields of the farfield condition """
        if self._fields is None:
            raise ValueError("Farfield fields not set!")
        return self._fields

    @fields.setter
    def fields(self, fields: ngsdict) -> None:
        if isinstance(fields, (dict, ngsdict)):
            self._fields = flowfields(**fields)
        elif fields is None:
            self._fields = None
        else:
            raise TypeError(f"Farfield fields must be of type '{flowfields}' or '{dict}'")

    @dream_configuration
    def use_identity_jacobian(self) -> bool:
        """ Returns the identity jacobian flag """
        return self._use_identity_jacobian

    @use_identity_jacobian.setter
    def use_identity_jacobian(self, use_identity_jacobian: bool) -> None:
        self._use_identity_jacobian = bool(use_identity_jacobian)


class Outflow(Condition):
    r""" Outflow condition for subsonic compressible flow.

    The outflow condition is used to set the subsonic outflow conditions for compressible flows by setting
    the static pressure :math:`p_\infty` at the outflow boundaries. By the hard imposition of the pressure,
    the outflow condition acts reflecting for acoustic waves.
    """

    name = "outflow"

    def __init__(self,
                 pressure: float | flowfields | None = None):

        self.fields = pressure

        super().__init__()

    @dream_configuration
    def fields(self) -> flowfields:
        """ Returns the set pressure """
        if self._fields is None:
            raise ValueError("Initial fields not set!")
        return self._fields

    @fields.setter
    def fields(self, fields: ngsdict) -> None:
        if isinstance(fields, flowfields):
            self._fields = flowfields(**fields)
        elif bla.is_scalar(fields):
            self._fields = flowfields(pressure=fields)
        elif fields is None:
            self._fields = None
        else:
            raise TypeError(f"Pressure must be a scalar or '{flowfields}'")


class CBC(Condition):

    name = "cbc"

    def __init__(self,
                 fields: flowfields | None = None,
                 target: str = "farfield",
                 relaxation_factor: float = 0.28,
                 tangential_relaxation: float = 0.0,
                 is_viscous_fluxes: bool = False):

        self.fields = fields
        self.target = target
        self.relaxation_factor = relaxation_factor
        self.tangential_relaxation = tangential_relaxation
        self.is_viscous_fluxes = is_viscous_fluxes
        super().__init__()

    @dream_configuration
    def fields(self) -> flowfields:
        """ Returns the fields of the farfield condition """
        if self._fields is None:
            raise ValueError("Farfield fields not set!")
        return self._fields

    @fields.setter
    def fields(self, fields: ngsdict) -> None:
        if isinstance(fields, (dict, ngsdict)):
            self._fields = flowfields(**fields)
        elif fields is None:
            self._fields = None
        else:
            raise TypeError(f"Farfield fields must be of type '{flowfields}' or '{dict}'")

    @dream_configuration
    def target(self) -> str:
        """ Returns the type of target state """
        return self._target

    @target.setter
    def target(self, target: str) -> None:
        target = str(target).lower()

        OPTIONS = ["farfield", "outflow", "mass_inflow", "temperature_inflow"]
        if target not in OPTIONS:
            raise ValueError(f"Invalid target '{target}'. Options are: {OPTIONS}")

        self._target = target

    @dream_configuration
    def relaxation_factor(self) -> dict[str, float]:
        """ Returns the relaxation factors for the different fields """
        return self._relaxation_factor

    @relaxation_factor.setter
    def relaxation_factor(self, factor: float | list | tuple | dict) -> None:
        characteristic = ["acoustic_in", "entropy", "vorticity_0", "vorticity_1", "acoustic_out"]
        if bla.is_scalar(factor):
            characteristic = dict.fromkeys(characteristic, factor)
        elif isinstance(factor, (list, tuple)):
            characteristic = {key: value for key, value in zip(characteristic, factor)}
        elif isinstance(factor, dict):
            characteristic = {key: value for key, value in zip(characteristic, factor.values())}

        self._relaxation_factor = characteristic

    @dream_configuration
    def tangential_relaxation(self) -> float:
        """ Returns the tangential relaxation factor """
        return self._tangential_relaxation

    @tangential_relaxation.setter
    def tangential_relaxation(self, tangential_relaxation: float) -> None:
        if isinstance(tangential_relaxation, ngs.Parameter):
            tangential_relaxation = tangential_relaxation.Get()
        self._tangential_relaxation = float(tangential_relaxation)

    @dream_configuration
    def is_viscous_fluxes(self) -> bool:
        """ Returns the viscous fluxes flag """
        return self._is_viscous_fluxes

    @is_viscous_fluxes.setter
    def is_viscous_fluxes(self, viscous_fluxes: bool) -> None:
        self._is_viscous_fluxes = bool(viscous_fluxes)

    def get_relaxation_matrix(self, dim: int, **kwargs) -> dict[str, ngs.CF]:

        factors = self.relaxation_factor.copy()
        if dim == 2:
            factors.pop("vorticity_1")

        return factors


class GRCBC(CBC):
    r""" Generalized Relaxation Condition Boundary Condition

    The GRCBC is a generalized relaxation condition for compressible flows. It is used to relax 
    the conservative variables at the boundary towards a given target state :math:`\vec{U}^-`. 
    The relaxation is done by a CFL-like diagonal matrix. Additionally, tangential relaxation and 
    viscous fluxes can improve the non-reflecting behavior on planar boundaries :cite:`PellmenreichCharacteristicBoundaryConditions2025`.

    :note: Currently, only supported with implicit time schemes.
    """

    name = "grcbc"

    def get_relaxation_matrix(self, dim: int, **kwargs) -> ngs.CF:
        dt = kwargs.get('dt', None)
        if dt is None:
            raise ValueError("Time step 'dt' not provided!")

        C = super().get_relaxation_matrix(dim)
        return bla.diagonal(tuple(C.values()))/dt


class NSCBC(CBC):

    name = "nscbc"

    def __init__(self,
                 fields=None,
                 target="farfield",
                 relaxation_factor=0.28,
                 tangential_relaxation=0.0,
                 is_viscous_fluxes=False,
                 length: float = 1.0):
        self.length = length
        super().__init__(fields, target, relaxation_factor, tangential_relaxation, is_viscous_fluxes)

    @dream_configuration
    def length(self) -> float:
        """ Returns the length scale """
        return self._length

    @length.setter
    def length(self, length: float):
        self._length = length

    def get_relaxation_matrix(self, dim: int, **kwargs) -> ngs.CF:
        c = kwargs.get('c', None)
        M = kwargs.get('M', None)
        if c is None or M is None:
            raise ValueError("Speed of sound 'c' and Mach number 'M' not provided!")

        sigmas = super().get_relaxation_matrix(dim)
        eig = [c * (1 - M**2)] + dim * [c] + [c * (1 - M**2)]
        eig = [factor*eig/self.length for factor, eig in zip(sigmas.values(), eig)]
        return bla.diagonal(eig)


class InviscidWall(Condition):
    r""" Inviscid wall condition for compressible flow.

    The inviscid wall condition is used to set the no-penetration condition 
    i.e. :math:`\vec{u} \cdot \vec{n} = 0` for compressible flows.

    """

    name = "inviscid_wall"


class Symmetry(Condition):
    r""" Symmetry condition for compressible flow.

    The symmetry condition imposes :math:`\vec{u} \cdot \vec{n} = 0` on a symmetry plane.

    """

    name = "symmetry"


class IsothermalWall(Condition):
    """ Isothermal wall condition for compressible flow.
    
    The isothermal wall condition sets the temperature :math:`T_w` and no-slip conditions at the wall boundaries.
    """

    name = "isothermal_wall"

    def __init__(self,
                 temperature: float | flowfields | None = None):

        self.fields = temperature
        super().__init__()

    @dream_configuration
    def fields(self) -> flowfields:
        """ Returns the set temperature """
        if self._fields is None:
            raise ValueError("Initial fields not set!")
        return self._fields

    @fields.setter
    def fields(self, fields: ngsdict) -> None:
        if isinstance(fields, flowfields):
            self._fields = flowfields(**fields)
        elif bla.is_scalar(fields):
            self._fields = flowfields(temperature=fields)
        elif fields is None:
            self._fields = None
        else:
            raise TypeError(f"Temperature must be a scalar or '{flowfields}'")


class AdiabaticWall(Condition):
    r""" Adiabatic wall condition for compressible flow.
    
    The adiabatic wall condition sets zero heat flux :math:`\vec{q} \cdot \vec{n}$` and no-slip conditions at the wall boundaries.
    """

    name = "adiabatic_wall"


BCS = [FarField, Outflow, GRCBC, NSCBC, InviscidWall, Symmetry, IsothermalWall, AdiabaticWall, Periodic]


# ------- Domain Conditions ------- #


class PML(Condition):

    name = "pml"


DCS = [PML, Force, Perturbation, Initial, GridDeformation, PSpongeLayer, SpongeLayer]
