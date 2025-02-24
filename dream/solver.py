from __future__ import annotations
import ngsolve as ngs
import logging
import typing

from math import isnan

from .compressible import CompressibleFlowConfiguration
from .mesh import is_mesh_periodic
from .config import UniqueConfiguration, InterfaceConfiguration, interface, configuration, unique
from .time import ExplicitMethod, StationaryConfig, TransientConfig, PseudoTimeSteppingConfig
from .io import IOConfiguration

logger = logging.getLogger(__name__)


class BonusIntegrationOrder(UniqueConfiguration):

    @configuration(default=0)
    def vol(self, vol):
        return int(vol)

    @configuration(default=0)
    def bnd(self, bnd):
        return int(bnd)

    @configuration(default=0)
    def bbnd(self, bbnd):
        return int(bbnd)

    @configuration(default=0)
    def bbbnd(self, bbbnd):
        return int(bbbnd)

    vol: int
    bnd: int
    bbnd: int
    bbbnd: int


class Inverse(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    def get_inverse(self, blf: ngs.BilinearForm, fes: ngs.FESpace, pre: ngs.BilinearForm = None):
        NotImplementedError()


class Direct(Inverse):

    name: str = "direct"

    @configuration(default="umfpack")
    def solver(self, solver: str):
        return solver

    def get_inverse(self, blf: ngs.BilinearForm, fes: ngs.FESpace, pre: ngs.BilinearForm = None):
        return blf.mat.Inverse(fes.FreeDofs(blf.condense), inverse=self.solver)

    solver: str


class NonlinearMethod(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    def reset_status(self):
        self.is_converged = False
        self.is_nan = False

    def set_solution_routine_attributes(self):
        raise NotImplementedError()

    def solve_update_step(self):
        raise NotImplementedError()

    def update_solution(self):
        raise NotImplementedError()

    def set_iteration_error(self, it: int | None = None, t: float | None = None):
        raise NotImplementedError()

    def log_iteration_error(self, error: float, it: int | None = None, t: float | None = None):
        msg = f"residual: {error:8e}"
        if it is not None:
            msg += f" | iteration: {it}"
        if t is not None:
            msg += f" | t: {t}"
        logger.info(msg)


class NewtonsMethod(NonlinearMethod):

    name: str = "newton"

    @configuration(default=1)
    def damping_factor(self, damping_factor: float):
        return float(damping_factor)

    def set_solution_routine_attributes(self):
        self.fes = self.cfg.pde.fes
        self.gfu = self.cfg.pde.gfu
        self.inverse = self.cfg.solver.inverse

        # Notice, the blf form is based on the implicit bilinear form in the solver.
        self.blf = self.cfg.solver.blfi
        self.lf = self.cfg.solver.lf

        self.residual = self.gfu.vec.CreateVector()
        self.temporary = self.gfu.vec.CreateVector()

    def solve_update_step(self):

        self.blf.Apply(self.gfu.vec, self.residual)
        self.residual.data -= self.lf.vec
        self.blf.AssembleLinearization(self.gfu.vec)

        inv = self.cfg.solver.inverse.get_inverse(self.blf, self.fes)
        if self.blf.condense:
            self.residual.data += self.blf.harmonic_extension_trans * self.residual
            self.temporary.data = inv * self.residual
            self.temporary.data += self.blf.harmonic_extension * self.temporary
            self.temporary.data += self.blf.inner_solve * self.residual
        else:
            self.temporary.data = inv * self.residual

    def update_solution(self):
        self.gfu.vec.data -= self.damping_factor * self.temporary

    def set_iteration_error(self, it: int | None = None, t: float | None = None):
        error = ngs.sqrt(ngs.InnerProduct(self.temporary, self.residual)**2)

        self.is_converged = error < self.cfg.solver.convergence_criterion
        self.log_iteration_error(error, it, t)
        self.error = error

        if isnan(error):
            self.is_nan = True
            logger.error("Solution process diverged!")

    damping_factor: float


class Solver(InterfaceConfiguration, is_interface=True):

    cfg: SolverConfiguration

    @interface(default=Direct)
    def inverse(self, inverse: Inverse):
        return inverse

    def initialize(self):

        self.fes = self.cfg.pde.fes
        condense = self.cfg.optimizations.static_condensation
        compile = self.cfg.optimizations.compile

        self.blfi = ngs.BilinearForm(self.fes, condense=condense)
        self.blfe = ngs.BilinearForm(self.fes)
        self.lf   = ngs.LinearForm(self.fes)

        # Implicit bilinear form. 
        for name, cf in self.cfg.pde.blfi.items():
            logger.debug(f"Adding {name} to the BilinearForm!")

            if compile.realcompile:
                self.blfi += cf.Compile(**compile)
            else:
                self.blfi += cf


        # Explicit bilinear form.
        for name, cf in self.cfg.pde.blfe.items():
            logger.debug(f"Adding {name} to the BilinearForm!")

            if compile.realcompile:
                self.blfe += cf.Compile(**compile)
            else:
                self.blfe += cf

        # Linear form.
        for name, cf in self.cfg.pde.lf.items():
            logger.debug(f"Adding {name} to the LinearForm!")

            if compile.realcompile:
                self.lf += cf.Compile(**compile)
            else:
                self.lf += cf

    inverse: Direct


class LinearSolver(Solver):

    name: str = "linear"

    def AllocateRKStageData(self):

        #Us[nStage];
        #for i in range(nStage):
        #    Us[i] = new ...
        self.RKnStage = 3
        #self.Us = [self.cfg.pde.gfu.vec.CreateVector() for _ in range(self.RKnStage)]
        self.RKa = [1.0, 0.75, 1.0/3.0]
        self.RKb = [1.0, 0.25, 2.0/3.0]
        self.RKc = [0.0, 1.00, 0.5]

        # Number of tentative solutions to store is one for an SSP-RK3.
        self.Us = self.cfg.pde.gfu.vec.CreateVector()

        #Us = []
        #for iStage in range(nStage):
        #    Us.append( self.cfg.pde.gfu.vec.CreateVector() )
    
        #Us[0] --- Us[nStage-1]
        

    def solve(self): 

        self.AllocateRKStageData()

        self.blfi.Assemble()
        minv = self.cfg.solver.inverse.get_inverse(self.blfi, self.cfg.pde.fes)

        print( self.cfg.time.scheme.name == 'implicit_euler' )
        if not isinstance( self.cfg.time.scheme, ExplicitMethod ):
            raise TypeError("banana")

        print( self.cfg.time )

        for t in self.cfg.time.start_solution_routine():
            print( t )

            # Extract the current time step.
            dt = self.cfg.time.timer.step

            # Extract the solution at the known time: t^n.
            Un = self.cfg.pde.gfu 

            #self.rhs = self.cfg.pde.gfu.vec.CreateVector()
            #
            #self.blfe.Apply( Un.vec, self.rhs )

            ## NOTE, can be assembled once.
            ##self.blfi.Assemble()
            ## NOTE, can be precomputed once.
            ##minv = self.cfg.solver.inverse.get_inverse(self.blfi, self.cfg.pde.fes)
            #
            #Un.vec.data += minv * self.rhs 

            

            # TESTING
            #self.Us[0].data = minv * self.rhs
            #self.Us[1].data = Un.vec
            #Un.vec.data = 0.1*self.Us[0] + 0.2*self.Us[1] 
            #self.blfe.Apply(self.Us[1], self.rhs)

            self.Us.data  = self.cfg.pde.gfu.vec
            self.res = self.cfg.pde.gfu.vec.CreateVector()

            for iStage in range(self.RKnStage):

                alpha = self.RKa[iStage]
                beta  = self.RKb[iStage]
                oma   = 1.0 - alpha 
                #bdt   = beta * dt
                bdt   = beta # The dt is already in the "minv"

                self.blfe.Apply( Un.vec, self.res )
                self.res.data = minv * self.res

                Un.vec.data = oma*Un.vec + alpha*self.Us + bdt*self.res


    # CHANGED
    #def solve(self):

    #    self.blf.Assemble()

    #    if hasattr(self, 'dirichlet'):
    #        self.dirichlet.data = self.blf.mat * self.gfu.vec

    #    for t in self.cfg.time.start_solution_routine():
    #        self.lf.Assemble()

    #        if hasattr(self, 'dirichlet'):
    #            self.lf.vec.data -= self.proj * self.dirichlet

    #        self.cfg.pde.gfu.vec.data += self.inverse.get_inverse(self.blf, self.cfg.pde.fes) * self.lf.vec

    #def set_dirichlet_vector(self):

    #    periodic = self.cfg.pde.bcs.get_periodic_boundaries(True)
    #    dofs = ~self.fes.FreeDofs() & ~self.fes.GetDofs(self.cfg.mesh.Boundaries(periodic))

    #    if any(dofs):
    #        self.proj = ngs.Projector(dofs, False)   # dirichlet mask
    #        self.dirichlet = self.gfu.vec.CreateVector()
    #        self.dirichlet[:] = 0

    #def initialize(self):
    #    super().initialize()
    #    self.gfu = self.cfg.pde.gfu
    #    self.set_dirichlet_vector()


class NonlinearSolver(Solver):

    name: str = "nonlinear"

    @interface(default=NewtonsMethod)
    def method(self, method: NonlinearMethod):
        return method

    @configuration(default=10)
    def max_iterations(self, max_iterations: int):
        return int(max_iterations)

    @configuration(default=1e-8)
    def convergence_criterion(self, convergence_criterion: float):
        if convergence_criterion <= 0:
            raise ValueError("Convergence Criterion must be greater zero!")
        return float(convergence_criterion)

    def initialize(self):
        super().initialize()
        self.method.set_solution_routine_attributes()

    def solve(self):

        for t in self.cfg.time.start_solution_routine():

            self.method.reset_status()

            for it in range(self.max_iterations):

                self.cfg.time.solver_iteration_update(it)

                self.method.solve_update_step()
                self.method.set_iteration_error(it, t)

                if self.method.is_nan:
                    break

                self.method.update_solution()

                if self.method.is_converged:
                    break

            if self.method.is_nan:
                break

    method: NonlinearMethod
    max_iterations: int
    convergence_criterion: float


class Compile(UniqueConfiguration):

    @configuration(default=True)
    def realcompile(self, flag: bool):
        return bool(flag)

    @configuration(default=False)
    def wait(self, flag: bool):
        return bool(flag)

    @configuration(default=False)
    def keep_files(self, flag: bool):
        return bool(flag)

    realcompile: bool
    wait: bool
    keep_files: bool


class Optimizations(UniqueConfiguration):

    @unique(default=Compile)
    def compile(self, compile):
        return compile

    @configuration(default=False)
    def static_condensation(self, static_condensation):
        return bool(static_condensation)

    @unique(default=BonusIntegrationOrder)
    def bonus_int_order(self, dict_):
        return dict_

    compile: Compile
    static_condensation: bool
    bonus_int_order: BonusIntegrationOrder


# This is the main class responsible for the solver configuration.
class SolverConfiguration(UniqueConfiguration):

    # Its abbreviated name is 'cfg'.
    name: str = 'cfg'

    # Class constructor, which initializes the mesh and calls its parent's ctor.
    def __init__(self, mesh: ngs.Mesh, **kwargs) -> None:
        self.mesh             = mesh
        self.mesh.normal      = ngs.specialcf.normal(mesh.dim)
        self.mesh.tangential  = ngs.specialcf.tangential(mesh.dim)
        self.mesh.meshsize    = ngs.specialcf.mesh_size
        self.mesh.is_periodic = is_mesh_periodic(mesh)

        # Calls the parent class constructor, i.e. UniqueConfiguration.
        super().__init__(cfg=self, mesh=self.mesh, **kwargs)

    @interface(default=CompressibleFlowConfiguration)
    def pde(self, pde):
        return pde

    @interface(default=LinearSolver)
    def solver(self, solver: Solver):
        return solver

    @interface(default=StationaryConfig)
    def time(self, time):
        return time

    @unique(default=Optimizations)
    def optimizations(self, optimizations):
        return optimizations

    @unique(default=IOConfiguration)
    def io(self, io):
        return io

    @configuration(default=None)
    def info(self, info=None):

        if info is None:
            info = {}

        if not isinstance(info, dict):
            raise ValueError("Info must be a dictionary!")

        return info

    def set_solver_documentation(self, doc: str):
        if isinstance(doc, str):
            ...
        elif isinstance(doc, typing.Sequence):
            doc = "\n".join(doc)
        else:
            raise ValueError("Documentation must be a string or sequence!")

        self.doc = doc

    def set_grid_deformation(self):
        grid_deformation = self.pde.dcs.get_grid_deformation_function()
        self.mesh.SetDeformation(grid_deformation)

    # This tells the user/editor about what solver options are avaiable.
    pde:           CompressibleFlowConfiguration
    solver:        LinearSolver | NonlinearSolver
    time:          StationaryConfig | TransientConfig | PseudoTimeSteppingConfig
    optimizations: Optimizations
    io:            IOConfiguration

