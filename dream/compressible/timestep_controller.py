""" Definitions of time-step controllers. """
from __future__ import annotations
import logging
import ngsolve as ngs
import typing

from dream.config import Configuration, dream_configuration

logger = logging.getLogger(__name__)



class TimeStepController(Configuration, is_interface=True):
    
    root: CompressibleFlowSolver

    def initialize(self) -> None:
        pass
    def process_iteration(self, iteration: int) -> None:
        pass

    @staticmethod
    def get_cfl_estimate(cfg: CompressibleFlowSolver, dt: float) -> float:
        
        # Create an instance of this class, temporarily.
        instance = PhysicalTimeStepController(mesh=cfg.mesh, root=cfg)
        # Initialize its grid points. 
        instance.initialize()

        # Use the instance to calculate the time step estimate.
        time_step_estimate = instance.get_time_step_estimate()
        
        # Return the CFL estimate and delete the temporary object.
        return dt / time_step_estimate



class PhysicalTimeStepController(TimeStepController):

    name: str = "physical_controller"

    def __init__(self, mesh, root=None, **default):
        
        DEFAULT = {'rate': 10,
                   'cfl': 1.0}

        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)
    
    @dream_configuration
    def rate(self) -> int:
        r""" Sets the time step estimation rate.

            :getter: Returns the computation rate
            :setter: Sets the computation rate, defaults to 10
        """
        return self._rate

    @rate.setter
    def rate(self, rate: int) -> None:
        if rate < 0:
            raise ValueError("Time step computation rate must be +ve.")
        self._rate = rate
    
    @dream_configuration
    def cfl(self) -> float:
        r""" Sets the estimate CFL number.

            :getter: Returns the CFL number 
            :setter: Sets the CFL number, defaults to 1.0
        """
        return self._cfl

    @cfl.setter
    def cfl(self, cfl: float) -> None:
        if cfl < 0.0:
            raise ValueError("Input CFL number must be +ve.")
        self._cfl = cfl

    def initialize(self) -> None:
        self.initialize_evaluation_points(mesh=self.root.mesh,
                                          order=self.root.fem.order,
                                          bonus=self.root.fem.get_max_bonus_integration())

    def initialize_evaluation_points(self, mesh: ngs.Mesh, order: int, bonus: int = 0):
        from dream.mesh import get_integration_points_physical_space, get_local_meshsize_from_points
        
        # Initialize element_types with the type of the first element.
        element_types = [next(iter(mesh.Elements())).type]

        # Loop through all elements in the mesh.
        for e in mesh.Elements():
            
            # If the element type is not already in element_types, add it.
            if e.type not in element_types:
                element_types.append(e.type) 

        if len(element_types) != 1:
            raise NotImplementedError( f"Adaptive time is currently resticted to a uniform element type, you have {len(element_type)} types." )

        # TODO: generalize this to hybrid meshes, having multiple element types.
        element_type = element_types[0]

        # Book-keep the integration points in physical space and their element length scales.
        self.xy = get_integration_points_physical_space(mesh, element_type=element_type, order=order, bonus=bonus)
        self.he = get_local_meshsize_from_points(mesh, x=self.xy[:,0], y=self.xy[:,1])
    
    def get_time_step_estimate(self, cfl: float = 1.0) -> float:
        return cfl * self.root.get_accurate_time_step_estimate(self.xy, self.he)

    def get_cfl_estimate(self, cfg: CompressibleFlowSolver, dt: float) -> float:
        return dt/self.get_time_step_estimate(cfg)

    def process_iteration(self, iteration: int) -> None:
        
        if iteration % self.rate == 0:
            dtmin = self.get_time_step_estimate(cfl=self.cfl)
            logger.info( f"estimated min(dt), for CFL = {self.cfl} is: {dtmin:.5e}" ) 




