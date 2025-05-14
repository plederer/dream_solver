from __future__ import annotations

import logging
import ngsolve as ngs

from dream import bla
from dream.config import dream_configuration, equation
from dream.mesh import (BoundaryConditions, DomainConditions)
from dream.solver import SolverConfiguration
from dream.time import TimeConfig

#from .eos import IdealGas, EquationOfState
#from .viscosity import Inviscid, Constant, Sutherland, DynamicViscosity
#from .scaling import Aerodynamic, Aeroacoustic, Acoustic, Scaling
#from .riemann_solver import LaxFriedrich, Roe, HLL, HLLEM, Upwind, RiemannSolver
#from .config import flowfields, BCS, DCS, CompressibleFiniteElementMethod
#from .formulations import ConservativeFiniteElementMethod
#from .time import CompressibleTransient, CompressiblePseudoTimeStepping 

from .config import flowfields, BCS, DCS
from .conservative import ScalarTransportFiniteElementMethod, HDG
from .time import ScalarTransportTransient 
from .riemann_solver import RiemannSolver, LaxFriedrich, Upwind

logger = logging.getLogger(__name__)


class ScalarTransportSolver(SolverConfiguration):

    name = "scalar_transport"

    def __init__(self, mesh: ngs.Mesh, **default) -> None:
        bcs = BoundaryConditions(mesh, BCS)
        dcs = DomainConditions(mesh, DCS)

        self._diffusion_coefficient = ngs.Parameter(1.0e-2)

        DEFAULT = {
            "fem": HDG(mesh, self),
            "time": ScalarTransportTransient(mesh, self),
            "riemann_solver": LaxFriedrich(mesh, self),
            "convection_velocity": [1.0] + [0.0 for _ in range(mesh.dim-1)],
            "diffusion_coefficient": 1.0e-2,
            "is_inviscid": False
        }
        DEFAULT.update(default)
        
        super().__init__(mesh=mesh, bcs=bcs, dcs=dcs, **DEFAULT)

    @dream_configuration
    def fem(self) -> ScalarTransportFiniteElementMethod:
        return self._fem

    @fem.setter
    def fem(self, fem):
        OPTIONS = [ScalarTransportFiniteElementMethod]
        self._fem = self._get_configuration_option(fem, OPTIONS, ScalarTransportFiniteElementMethod)

    @dream_configuration
    def time(self) -> ScalarTransportTransient:
        r""" Sets the time configuration for the scalar transport solver. 

            :getter: Returns the time configuration
            :setter: Sets the time configuration, defaults to ScalarTransportTransient
        """
        return self._time
    
    @time.setter
    def time(self, time: TimeConfig):
        OPTIONS = [ScalarTransportTransient]
        self._time = self._get_configuration_option(time, OPTIONS, TimeConfig)

    @dream_configuration
    def riemann_solver(self) -> LaxFriedrich | Upwind:
        r""" Sets the Riemann solver for the scalar transport solver.

            :getter: Returns the Riemann solver
            :setter: Sets the Riemann solver, defaults to LaxFriedrich
        """
        return self._riemann_solver

    @riemann_solver.setter
    def riemann_solver(self, riemann_solver: str | LaxFriedrich | Upwind):
        OPTIONS = [LaxFriedrich, Upwind]
        self._riemann_solver = self._get_configuration_option(riemann_solver, OPTIONS, RiemannSolver)

    @dream_configuration
    def convection_velocity(self) -> ngs.CF:
        r""" Sets the convection/wind velocity.

            :getter: Returns the convection speed
            :setter: Sets the convection speed, defaults to (1.0, 0.0)
        """
        return self._convection_velocity

    @convection_velocity.setter
    def convection_velocity(self, convection_velocity: ngs.CF):
        
        if not isinstance( convection_velocity, ngs.CF ):
            convection_velocity = ngs.CF( tuple(convection_velocity) )

        if not convection_velocity.dim == self.mesh.dim:
            raise ValueError("Convection speed must match mesh dimension.")
      
        self._convection_velocity = convection_velocity

    @dream_configuration
    def diffusion_coefficient(self) -> ngs.Parameter:
        r""" Sets the diffusivity constant.

            :getter: Returns the diffusivity constant
            :setter: Sets the diffusivity constant, defaults to 0.001
        """
        return self._diffusion_coefficient

    @diffusion_coefficient.setter
    def diffusion_coefficient(self, diffusion_coefficient: float):
        self._diffusion_coefficient.Set(diffusion_coefficient)

    @dream_configuration
    def is_inviscid(self) -> bool:
        return self._is_inviscid

    @is_inviscid.setter
    def is_inviscid(self, is_inviscid: bool) -> None:
        self._is_inviscid = bool(is_inviscid)

    def get_convective_flux(self, U: flowfields) -> ngs.CF:
        return self.convection_velocity * U.phi 

    @equation
    def phi(self, U: flowfields):
        if U.phi is not None:
            return U.phi





