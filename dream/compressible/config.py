from __future__ import annotations
import ngsolve as ngs

from dream import bla
from dream.config import quantity, configuration, ngsdict
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


class CompressibleFiniteElement(FiniteElementMethod, is_interface=True):

    @property
    def gfu(self) -> ngs.GridFunction:
        return self.cfg.gfu

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


class dimensionfulfields(ngsdict):
    L = quantity("length")
    rho = quantity("density")
    rho_u = quantity("momentum")
    u = quantity("velocity")
    c = quantity("speed_of_sound")
    T = quantity("temperature")
    p = quantity("pressure")


class FarField(Condition):

    name = "farfield"

    @configuration(default=None)
    def fields(self, fields) -> flowfields:
        if fields is not None:
            fields = flowfields(**fields)
        return fields

    @fields.getter_check
    def fields(self) -> None:
        if self.data['fields'] is None:
            raise ValueError("Farfield fields not set!")

    @configuration(default=True)
    def identity_jacobian(self, use_identity_jacobian: bool):
        return bool(use_identity_jacobian)

    fields: flowfields


class Outflow(Condition):

    name = "outflow"

    @configuration(default=None)
    def fields(self, fields) -> flowfields:
        if fields is not None:
            if bla.is_scalar(fields):
                fields = flowfields(p=fields)
            else:
                fields = flowfields(**fields)
        return fields

    @fields.getter_check
    def fields(self) -> None:
        if self.data['fields'] is None:
            raise ValueError("Outflow fields not set!")

    fields: flowfields


class CBC(Condition):

    name = "cbc"

    @configuration(default=None)
    def fields(self, fields) -> flowfields:
        if fields is not None:
            fields = flowfields(**fields)
        return fields

    @fields.getter_check
    def fields(self) -> None:
        if self.data['fields'] is None:
            raise ValueError("CBC fields not set!")

    @configuration(default="farfield")
    def target(self, target: str):

        target = str(target).lower()

        options = ["farfield", "outflow", "mass_inflow", "temperature_inflow"]
        if target not in options:
            raise ValueError(f"Invalid target '{target}'. Options are: {options}")

        return target

    @configuration(default=0.28)
    def relaxation_factor(self, factor: float):
        characteristic = ["acoustic_in", "entropy", "vorticity_0", "vorticity_1", "acoustic_out"]
        if bla.is_scalar(factor):
            characteristic = dict.fromkeys(characteristic, factor)
        elif isinstance(factor, (list, tuple)):
            characteristic = {key: value for key, value in zip(characteristic, factor)}
        elif isinstance(factor, dict):
            characteristic = {key: value for key, value in zip(characteristic, factor.values())}
        return characteristic

    @configuration(default=0.0)
    def tangential_relaxation(self, tangential_relaxation: bla.SCALAR):
        return bla.as_scalar(tangential_relaxation)

    @configuration(default=False)
    def is_viscous_fluxes(self, viscous_fluxes: bool):
        return bool(viscous_fluxes)

    def get_relaxation_matrix(self, **kwargs) -> dict[str, ngs.CF]:

        factors = self.relaxation_factor.copy()
        if self.mesh.dim == 2:
            factors.pop("vorticity_1")

        return factors

    fields: flowfields
    target: str
    relaxation_factor: dict
    tangential_relaxation: ngs.CF
    is_viscous_fluxes: bool


class GRCBC(CBC):

    name = "grcbc"

    def get_relaxation_matrix(self, **kwargs) -> ngs.CF:
        dt = kwargs.get('dt', None)
        if dt is None:
            raise ValueError("Time step 'dt' not provided!")

        C = super().get_relaxation_matrix()
        return bla.diagonal(tuple(C.values()))/dt


class NSCBC(CBC):

    name = "nscbc"

    @configuration(default=1)
    def length(self, length: float):
        return length

    def get_relaxation_matrix(self, **kwargs) -> ngs.CF:
        c = kwargs.get('c', None)
        M = kwargs.get('M', None)
        if c is None or M is None:
            raise ValueError("Speed of sound 'c' and Mach number 'M' not provided!")

        sigmas = super().get_relaxation_matrix()
        eig = [c * (1 - M**2)] + self.mesh.dim * [c] + [c * (1 - M**2)]
        eig = [factor*eig/self.length for factor, eig in zip(sigmas.values(), eig)]
        return bla.diagonal(eig)

    length: float


class InviscidWall(Condition):

    name = "inviscid_wall"


class Symmetry(Condition):

    name = "symmetry"


class IsothermalWall(Condition):

    name = "isothermal_wall"

    @configuration(default=None)
    def fields(self, fields) -> flowfields:
        if fields is not None:
            if bla.is_scalar(fields):
                fields = flowfields(T=fields)
            else:
                fields = flowfields(**fields)
        return fields

    @fields.getter_check
    def fields(self) -> None:
        if self.data['fields'] is None:
            raise ValueError("Isothermal Wall fields not set!")

    fields: flowfields


class AdiabaticWall(Condition):

    name = "adiabatic_wall"


BCS = [FarField, Outflow, GRCBC, NSCBC, InviscidWall, Symmetry, IsothermalWall, AdiabaticWall, Periodic]


# ------- Domain Conditions ------- #


class PML(Condition):

    name = "pml"


DCS = [PML, Force, Perturbation, Initial, GridDeformation, PSpongeLayer, SpongeLayer]
