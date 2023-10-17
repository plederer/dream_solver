""" 
Simulation of an isothermal vortex in an Euler setting with 
strength 0.01, radius 0.1 and polynomial order 6. The sound region consists of a
square domain with bounding box (-1, -1) x (1, 1).
"""
from ngsolve import *
from netgen.occ import OCCGeometry
from dream import *
from dream.utils import RectangleDomain, Rectangle1DGrid
from benchmarks.template import Benchmark

ngsglobals.msg_level = 0
SetNumThreads(32)

tree = ResultsDirectoryTree()

cfg = SolverConfiguration()
cfg.formulation = "conservative"
cfg.fem = "hdg"
cfg.scaling = "aeroacoustic"
cfg.riemann_solver = 'hllem'

cfg.Mach_number = 0
cfg.heat_capacity_ratio = 1.4

cfg.order = 6
cfg.bonus_int_order = {VOL: cfg.order, BND: cfg.order}

cfg.time.simulation = "transient"
cfg.time.scheme = "BDF2"
cfg.time.step = 0.2
cfg.time.interval = (0, 150)

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.convergence_criterion = 1e-16

cfg.compile_flag = True
cfg.static_condensation = True


class Vortex(Benchmark):

    def __init__(self,
                 cfg: SolverConfiguration,
                 tree: ResultsDirectoryTree,
                 M: float,
                 maxh: float = 0.15,
                 buffer: bool = False,
                 draw: bool = False) -> None:

        super().__init__(cfg, tree, draw)

        self.cfg.Mach_number = M
        self.sound = RectangleDomain(H=2, W=2, mat="sound", maxh=maxh)
        self.buffer = buffer
        self.comment = __doc__

        # Far-field values
        self.farfield = INF.farfield((1, 0), self.cfg)
        rho_inf = self.farfield.density
        u_inf = CF(self.farfield.velocity)
        p_inf = self.farfield.pressure

        gamma = self.cfg.heat_capacity_ratio
        c = INF.speed_of_sound(cfg)

        Gamma = 0.01 * M/(M + 1)
        Rv = 0.1
        r = sqrt(x**2 + y**2)
        psi = Gamma * exp(-r**2/(2*Rv**2))

        self.cfg.info["Gamma"] = Gamma
        self.cfg.info["Radius"] = Rv

        # Vortex Isothermal
        u_0 = u_inf + CF((psi.Diff(y), -psi.Diff(x)))
        p_0 = p_inf * exp(-gamma/2*(Gamma/(c * Rv))**2 * exp(-r**2/Rv**2))
        rho_0 = rho_inf * exp(-gamma/2*(Gamma/(c * Rv))**2 * exp(-r**2/Rv**2))
        self.p_00 = p_inf * exp(-gamma/2*(Gamma/(c * Rv))**2)
        self.initial = State(u_0, rho_0, p_0)

    def add_meta_data(self):
        self.cfg.info["Sound Domain"] = (f"W: {self.sound.W}, H: {self.sound.H}")
        self.cfg.info["ndof"] = self.solver.formulation.fes.ndof

    def get_mesh(self):
        main = Rectangle1DGrid(self.sound)

        if self.buffer:
            sponge_length = 2*self.sound.W
            maxh = sponge_length/(self.cfg.order + 1)
            fronts = [RectangleDomain(maxh, maxh, mat=str(k), maxh=maxh/2) for k in range(self.cfg.order, -1, -1)]

            for front in fronts:
                main.add_front(front)

        main.add_periodic('top', 'bottom')
        geo = OCCGeometry(main.get_face(), dim=2)
        self.main = main

        return Mesh(geo.GenerateMesh())

    def set_boundary_conditions(self):
        self.solver.boundary_conditions.set(bcs.FarField(self.farfield), "left")
        self.solver.boundary_conditions.set(bcs.Periodic(), "top|bottom")

    def set_domain_conditions(self):
        self.solver.domain_conditions.set(dcs.Initial(self.initial))

    def draw_scenes(self):
        self.solver.drawer.draw()
        self.solver.drawer.draw_acoustic_pressure(self.farfield.pressure, autoscale=False, min=-1e-2, max=1e-2)
        Draw((self.solver.formulation.pressure() - self.farfield.pressure)/(self.p_00 - self.farfield.pressure), self.solver.mesh, "p*")


class NSCBC(Vortex):

    def __init__(self, cfg: SolverConfiguration,
                 tree: ResultsDirectoryTree,
                 M: float,
                 dim: str = "1d",
                 draw: bool = False) -> None:
        super().__init__(cfg, tree, M, buffer=False, draw=draw)
        name = f"nscbc{dim}_M{M}"
        self.tree.directory_name = name

        self.dim = dim
        self.sigma = 0.28
        self.L = 1

    def add_meta_data(self):
        super().add_meta_data()
        self.cfg.info["Dim"] = self.dim
        self.cfg.info["Sigma"] = self.sigma
        self.cfg.info["Reference Length"] = self.L

    def set_boundary_conditions(self):
        super().set_boundary_conditions()

        if self.dim == "1d":
            self.solver.boundary_conditions.set(
                bcs.Outflow_NSCBC(self.farfield.pressure, self.sigma, self.L, tangential_convective_fluxes=False),
                "right")
        elif self.dim == "2d":
            self.solver.boundary_conditions.set(
                bcs.Outflow_NSCBC(self.farfield.pressure, self.sigma, self.L, tangential_convective_fluxes=True),
                "right")


class PSponge(Vortex):

    def __init__(self,
                 cfg: SolverConfiguration,
                 tree: ResultsDirectoryTree,
                 M: float,
                 dB: float,
                 projection: str,
                 draw: bool = False) -> None:
        super().__init__(cfg, tree, M, buffer=True, draw=draw)

        self.projection = projection
        self.dB = dB

        name = f"psponge_P{projection}_dB{-dB}_M{M}"
        self.tree.directory_name = name

        if projection == "k":
            self.order = dcs.PSpongeLayer.range(self.cfg.order)
        elif projection == "0":
            self.order = tuple(dcs.PSpongeLayer.SpongeOrder(k, 0) for k in range(cfg.order, -1, -1))

    def set_boundary_conditions(self):
        super().set_boundary_conditions()
        self.solver.boundary_conditions.set(bcs.FarField(self.farfield), "right")

    def add_meta_data(self):
        super().add_meta_data()
        self.cfg.info["Projection"] = self.projection
        self.cfg.info["Dezibel Reduction"] = self.dB
        self.cfg.info["\u03A0-Sponge Orders"] = self.order
        self.cfg.info["\u03A0-Sponge Lengths X"] = self.lengths_x
        self.cfg.info["\u03A0-Sponge Function X"] = self.sponge_x
        self.cfg.info["\u03A0-Sponge Weights X"] = self.weights_x
        self.cfg.info["Grid Factor"] = 15
        self.cfg.info["Grid Function X"] = repr(self.grid.x)

    def set_domain_conditions(self):
        super().set_domain_conditions()

        fronts = self.main.front

        # Gridstretching
        x0, xn = self.sound.W/2, sum([front.W for front in fronts]) + self.sound.W/2
        self.x_ = BufferCoordinate.x(x0, xn)

        grid_x = GridDeformationFunction.ExponentialThickness(15, self.x_, False)
        self.grid = GridDeformationFunction(grid_x)

        for front in fronts:
            self.solver.domain_conditions.set(dcs.GridDeformation(self.grid), front.mat)

        self.weights_x = []
        self.lengths_x = []

        for order, front in zip(self.order, fronts):

            lx = self.grid.x.deformed_length(front.x_)
            self.lengths_x.append(lx)
            sigma_x = SpongeWeight.quadratic(lx, cfg.Mach_number.Get(), dB=self.dB)
            self.weights_x.append(sigma_x)

            sponge_x = SpongeFunction.quadratic(front.x_, weight=sigma_x, mirror=False)

            self.solver.domain_conditions.set(dcs.PSpongeLayer(
                order.high, order.low, sponge_x, state=self.farfield), domain=front.mat)

        self.sponge_x = sponge_x


class Sponge(Vortex):

    def __init__(self,
                 cfg: SolverConfiguration,
                 tree: ResultsDirectoryTree,
                 M: float,
                 dB: float,
                 draw: bool = False) -> None:
        super().__init__(cfg, tree, M, buffer=True, draw=draw)

        self.dB = dB

        name = f"sponge_dB{-dB}_M{M}"
        self.tree.directory_name = name

    def set_boundary_conditions(self):
        super().set_boundary_conditions()
        self.solver.boundary_conditions.set(bcs.FarField(self.farfield), "right")

    def add_meta_data(self):
        super().add_meta_data()
        self.cfg.info["Dezibel Reduction"] = self.dB
        self.cfg.info["\u03A0-Sponge Length X"] = self.length_x
        self.cfg.info["\u03A0-Sponge Function X"] = self.sponge_x
        self.cfg.info["\u03A0-Sponge Weight X"] = self.weight_x

    def set_domain_conditions(self):
        super().set_domain_conditions()

        fronts = self.main.front

        # Gridstretching
        x0, xn = self.sound.W/2, sum([front.W for front in fronts]) + self.sound.W/2
        self.x_ = BufferCoordinate.x(x0, xn)

        grid_x = GridDeformationFunction.ExponentialThickness(15, self.x_, False)
        self.grid = GridDeformationFunction(grid_x)

        for front in fronts:
            self.solver.domain_conditions.set(dcs.GridDeformation(self.grid), front.mat)

        # Sponge Layer
        lx = self.grid.x.deformed_length(self.x_)
        sigma_x = SpongeWeight.quadratic(lx, self.cfg.Mach_number.Get(), dB=self.dB)
        sponge_x = SpongeFunction.quadratic(self.x_, sigma_x, mirror=False)
        for front in fronts:
            self.solver.domain_conditions.set(dcs.SpongeLayer(self.farfield, sponge_x), domain=front.mat)

        self.length_x = lx
        self.weight_x = sigma_x
        self.sponge_x = sponge_x


if __name__ == "__main__":
    Mach_numbers = [0.05]

    for Mach in Mach_numbers:

        benchmarks = [
            NSCBC(cfg, tree, Mach, "1d"),
            NSCBC(cfg, tree, Mach, "2d"),
            Sponge(cfg, tree, Mach, dB=-40),
            Sponge(cfg, tree, Mach, dB=-200),
            PSponge(cfg, tree, Mach, dB=-40, projection="0"),
            PSponge(cfg, tree, Mach, dB=-200, projection="0"),
            PSponge(cfg, tree, Mach, dB=-40, projection="k"),
            PSponge(cfg, tree, Mach, dB=-200, projection="k"),
        ]

        for benchmark in benchmarks:
            benchmark.start(True)
