""" Definitions of boundary/domain conditions for compressible flow """
from __future__ import annotations
import ngsolve as ngs
import typing

from dream import bla
from dream.config import quantity, dream_configuration, ngsdict, Integrals
from dream.solver import FiniteElementMethod
from dream.mesh import (Condition,
                        Periodic,
                        Initial,
                        Perturbation,
                        SpongeLayer,
                        PSpongeLayer,
                        GridDeformation
                        )

if typing.TYPE_CHECKING:
    from .solver import CompressibleFlowSolver


class CompressibleFiniteElementMethod(FiniteElementMethod):

    root: CompressibleFlowSolver

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            'bonus_int_order': ('convection', 'diffusion'),
        }

        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    def add_symbolic_spatial_forms(self, blf: Integrals, lf: Integrals):

        self.add_convection_form(blf, lf)

        if not self.root.dynamic_viscosity.is_inviscid:
            self.add_diffusion_form(blf, lf)

        self.add_boundary_conditions(blf, lf)
        self.add_domain_conditions(blf, lf)

    def add_convection_form(self, blf: Integrals, lf: Integrals):
        raise NotImplementedError("Overload this method in derived class!")

    def add_diffusion_form(self, blf: Integrals, lf: Integrals):
        raise NotImplementedError("Overload this method in derived class!")

    def add_boundary_conditions(self, blf: Integrals, lf: Integrals):
        raise NotImplementedError("Overload this method in derived class!")

    def add_domain_conditions(self, blf: Integrals, lf: Integrals):
        raise NotImplementedError("Overload this method in derived class!")

    def get_max_bonus_integration(self) -> int:
        return max(max(subdict.values()) for subdict in self.bonus_int_order.values())
    
    def get_domain_boundary_mask(self) -> ngs.GridFunction:
        """ 
        Returns a Gridfunction that is 0 on the domain boundaries and 1 on the domain interior.
        """

        fes = ngs.FacetFESpace(self.mesh, order=0)
        mask = ngs.GridFunction(fes, name="mask")
        mask.vec[:] = 0

        bnd_dofs = fes.GetDofs(self.mesh.Boundaries(self.root.bcs.get_domain_boundaries()))
        mask.vec[~bnd_dofs] = 1

        return mask

    def set_boundary_conditions(self) -> None:
        """ Boundary conditions for compressible flows are set weakly. Therefore we do nothing here."""
        pass


class ConservativeFiniteElementMethod(CompressibleFiniteElementMethod):

    def get_conservative_fields(self, U: ngs.CoefficientFunction, with_gradients: bool = False) -> flowfields:

        dU = None
        if isinstance(U, ngs.GridFunction):
            dU = ngs.grad(U)
            U = U.components
        elif isinstance(U, ngs.comp.ProxyFunction):
            dU = ngs.grad(U)

        U_ = flowfields()
        U_.rho = U[0]
        U_.rho_u = U[slice(1, self.mesh.dim + 1)]
        U_.rho_E = U[self.mesh.dim + 1]

        U_.u = self.root.velocity(U_)
        U_.rho_Ek = self.root.kinetic_energy(U_)
        U_.rho_Ei = self.root.inner_energy(U_)
        U_.p = self.root.pressure(U_)
        U_.T = self.root.temperature(U_)
        U_.c = self.root.speed_of_sound(U_)

        if isinstance(U, ngs.comp.ProxyFunction):
            U_.U = U

        if dU is not None and with_gradients:
            U_.grad_rho = dU[0, :]
            U_.grad_rho_u = dU[slice(1, self.mesh.dim + 1), :]
            U_.grad_rho_E = dU[self.mesh.dim + 1, :]

            U_.grad_u = self.root.velocity_gradient(U_, U_)
            U_.grad_rho_Ek = self.root.kinetic_energy_gradient(U_, U_)
            U_.grad_rho_Ei = self.root.inner_energy_gradient(U_, U_)
            U_.grad_p = self.root.pressure_gradient(U_, U_)
            U_.grad_T = self.root.temperature_gradient(U_, U_)

        return U_

    def get_solution_fields(self) -> flowfields:
        return self.get_conservative_fields(self.gfus['U'], with_gradients=True)

    def initialize_time_scheme_gridfunctions(self, *spaces):

        SPACES = ['U']
        SPACES.extend(spaces)

        super().initialize_time_scheme_gridfunctions(*SPACES)

    def set_initial_conditions(self):

        U = self.mesh.MaterialCF({dom: ngs.CF(
            (self.root.density(dc.fields),
                self.root.momentum(dc.fields),
                self.root.energy(dc.fields))) for dom, dc in self.root.dcs.items(Initial)})

        self.gfus['U'].Set(U)

        super().set_initial_conditions()


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
    omega = quantity('vorticity', r"\bm{\omega}")


class dimensionalfields(dict):
    """ Mutable mapping for scalar dimensionalisation fields. """
    rho = quantity('density', r"\rho \, [\mathrm{kg/m^3}]")
    u = quantity('velocity', r"\bm{u} \, [\mathrm{m/s}]")
    T = quantity('temperature', r"T \, [\mathrm{K}]")
    c = quantity('speed_of_sound', r"c \, [\mathrm{m/s}]")
    L = quantity('length scale', r"L \, [\mathrm{m}]")
    t = quantity('time scale', r"t \, [\mathrm{s}]")
    mu = quantity('viscosity', r"\mu \, [\mathrm{kg/(m \cdot s)}]")
    k = quantity('thermal_conductivity', r"k \, [\mathrm{W/(m \cdot K)}]")
    c_p = quantity('specific_heat_capacity_p', r"c_p \, [\mathrm{J/(kg \cdot K)}]")

    def __init__(self,
                 rho_inf: float,
                 u_inf: float,
                 T_inf: float,
                 L: float = 1.0,
                 mu_inf: float = 1.7894e-5,
                 k_inf: float = 2.587e-2,
                 c_p: float = 1004.5, **kwargs):

        self.rho = rho_inf
        self.u = u_inf
        self.T = T_inf
        self.L = L
        self.mu = mu_inf
        self.k = k_inf
        self.c_p = c_p
        super().__init__(**kwargs)


class FarField(Condition):
    r""" Farfield condition for compressible flow.

    The farfield condition is used to set the subsonic and supersonic inflow/outflow conditions for compressible 
    flows in a characteristic way. It acts partially non-reflecting for acoustic waves on both inflow and outflow boundaries. 

    :param fields: Dictionary of flow quantities :math:`\vec{U}_\infty` to be set at the farfield boundaries.
    :type fields: flowfields
    :param use_identity_jacobian: Flag to use the identity jacobian for the farfield condition.
    :type use_identity_jacobian: bool

    :note: See :func:`~dream.compressible.conservative.hdg.HDG.add_farfield_formulation` for the implementation of 
              the farfield condition in the :class:`~dream.compressible.conservative.hdg.HDG` formulation.

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

class Inflow(Condition):
    r""" Inflow condition for subsonic compressible flow.

    The inflow condition is used to set the subsonic inflow conditions for compressible flows by setting
    the static variables: density and velocity, :math:`rho_\infty, u_\infty, v_\infty` at the inflow boundaries. By the hard imposition of these variables, the inflow condition acts reflecting for acoustic waves. Also, this can become too constricting/unstable for internal flows, due to choking.
    """

    name = "inflow"

    def __init__(self, density: float, velocity: float):

        self.density = density
        self.velocity = velocity

        super().__init__()

    @dream_configuration
    def density(self) -> ngs.CF:
        if self._density is None:
            raise ValueError(f"Density value must be specified.")
        return self._density

    @density.setter
    def density(self, value: ngs.CF) -> None:
        self._density = ngs.CF( value )

    @dream_configuration
    def velocity(self) -> ngs.CF:
        if self._velocity is None:
            raise ValueError(f"Velocity values must be specified.")
        return self._velocity

    @velocity.setter
    def velocity(self, value: ngs.CF) -> None:
        if value.dim != 2:
            raise ValueError(f"Velocity vector must have a dimension of 2.")
        self._velocity = ngs.CF( value )

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


class InterfaceBC(Condition):

    name = "interface"

    def __init__(self,
                 fields: flowfields | None = None):

        self.fields = fields

        super().__init__()

    @dream_configuration
    def fields(self) -> flowfields:
        if self._fields is None:
            raise ValueError("InterfaceBC fields not set!")
        return self._fields

    @fields.setter
    def fields(self, fields: ngsdict) -> None:
        if isinstance(fields, (dict, ngsdict)):
            self._fields = flowfields(**fields)
        elif fields is None:
            self._fields = None
        else:
            raise TypeError(f"InterfaceBC fields must be of type '{flowfields}' or '{dict}'")


class Dirichlet(Condition):
    r""" Dirichlet boundary condition for compressible flow.

    The Dirichlet condition is used to set the conservative variables :math:`\vec{U} = (\rho, \rho \vec{u}, \rho E)`
    at the boundary. It is mainly used for testing purposes.
    """

    name = "dirichlet"

    def __init__(self, fields: flowfields):
        self.fields = fields
        super().__init__()


class Force(Condition):

    def __init__(self,
                 continuum: float | None = None,
                 momentum: float | None = None,
                 energy: float | None = None,
                 flux: ngs.CF | None = None,
                 order: int = 0,
                 is_constant: bool = True):

        super().__init__()
        self.order = order
        self._continuum = continuum
        self._momentum = momentum
        self._energy = energy
        self._flux = flux
        self.is_constant = is_constant

    def get_force_vector(self, dim: int = 2) -> ngs.CF:
        return ngs.CF((self.continuum(), self.momentum(dim), self.energy()))

    def continuum(self) -> ngs.CF:
        return ngs.CF(self._continuum) if self._continuum is not None else 0.0

    def momentum(self, dim: int = 2) -> ngs.CF:
        return ngs.CF(self._momentum) if self._momentum is not None else tuple(dim * [0.0])

    def energy(self) -> ngs.CF:
        return ngs.CF(self._energy) if self._energy is not None else 0.0

    @dream_configuration
    def order(self) -> int:
        return self._order

    @order.setter
    def order(self, order):
        self._order = int(order)


BCS = [FarField, Outflow, Inflow, GRCBC, NSCBC, InviscidWall, Symmetry, IsothermalWall, AdiabaticWall, InterfaceBC, Periodic, Dirichlet]


# ------- Domain Conditions ------- #


class PML(Condition):

    name = "pml"


DCS = [PML, Force, Perturbation, Initial, GridDeformation, PSpongeLayer, SpongeLayer]
