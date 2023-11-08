"""
Simulation of an Gaussian pressure pulse in an Euler setting at
Mach number zero and polynomial order 6. The sound region consists of a
square domain with bounding box (-1, -1) x (1, 1).
"""
from dream import ResultsDirectoryTree, SolverConfiguration
from ngsolve import *
from netgen.occ import OCCGeometry
from dream import *
from dream.utils import RectangleDomain, RectangleGrid
from benchmarks.template import Benchmark
from math import log

ngsglobals.msg_level = 0
SetNumThreads(32)

tree = ResultsDirectoryTree()

cfg = SolverConfiguration()
cfg.formulation = "conservative"
cfg.scaling = "acoustic"
cfg.riemann_solver = 'lax_friedrich'

cfg.Mach_number = 0
cfg.heat_capacity_ratio = 1.4

cfg.order = 6
cfg.bonus_int_order_bnd = cfg.order
cfg.bonus_int_order_vol = cfg.order

cfg.time.simulation = "transient"
cfg.time.scheme = "BDF2"
cfg.time.step = 0.05
cfg.time.interval = (0, 100)

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.convergence_criterion = 1e-16

cfg.compile_flag = True
cfg.static_condensation = True

# Far-field values
farfield = INF.farfield((0, 0), cfg)
rho_inf = farfield.density
u_inf = CF(farfield.velocity)
p_inf = farfield.pressure

# Initial values
AMP, beta = 0.5, 0.25
u_0 = u_inf
p_0 = p_inf * (1 + AMP * exp(-log(2)*(x**2 + y**2)/beta**2))
rho_0 = rho_inf * (1 + AMP * exp(-log(2)*(x**2 + y**2)/beta**2))
initial = State(u_0, rho_0, p_0)


class Pulse(Benchmark):

    def __init__(self,
                 cfg: SolverConfiguration,
                 tree: ResultsDirectoryTree,
                 maxh: float = 0.2,
                 buffer: bool = False,
                 factor: int = 1,
                 draw: bool = False) -> None:

        super().__init__(cfg, tree, draw)

        self.sound = RectangleDomain(H=2, W=2, mat="sound", maxh=maxh)
        self.buffer = buffer
        self.factor = factor
        self.comment = __doc__
        self.save_state_every = 2

    def add_meta_data(self):
        self.cfg.info["Sound Domain"] = (f"W: {self.sound.W}, H: {self.sound.H}")
        self.cfg.info["ndof"] = self.solver.formulation.fes.ndof

    def get_mesh(self):
        main = RectangleGrid(self.sound)

        if self.buffer:
            sponge_length = self.factor * self.sound.W
            maxh = sponge_length/(self.cfg.order + 1)
            rings = [RectangleDomain(2*maxh, 2*maxh, mat=str(k), maxh=maxh/self.factor)
                     for k in range(self.cfg.order, -1, -1)]

            for ring in rings:
                main.add_ring(ring)

        geo = OCCGeometry(main.get_face(), dim=2)
        self.main = main

        return Mesh(geo.GenerateMesh())

    def set_domain_conditions(self):
        self.solver.domain_conditions.set(dcs.Initial(initial))

    def draw_scenes(self):
        self.solver.drawer.draw()
        self.solver.drawer.draw_acoustic_pressure(p_inf, autoscale=False, min=-1e-2, max=1e-2)
        self.solver.drawer.draw_mach_number()


class NSCBC(Pulse):

    def __init__(self, cfg: SolverConfiguration,
                 tree: ResultsDirectoryTree,
                 dim: str = "1d",
                 draw: bool = False) -> None:
        super().__init__(cfg, tree, buffer=False, draw=draw)
        name = f"nscbc_{dim}"
        self.tree.directory_name = name
        self.cfg.fem = "edg"

        self.dim = dim
        self.sigma = 0.25
        self.L = 1

    def add_meta_data(self):
        super().add_meta_data()
        self.cfg.info["Dim"] = self.dim
        self.cfg.info["Sigma"] = self.sigma
        self.cfg.info["Reference Length"] = self.L

    def set_boundary_conditions(self):
        if self.dim == "1d":
            self.solver.boundary_conditions.set(
                bcs.NSCBC(p_inf, self.sigma, self.L, tangential_convective_fluxes=False),
                "left|right|bottom|top")
        elif self.dim == "2d":
            self.solver.boundary_conditions.set(
                bcs.NSCBC(p_inf, self.sigma, self.L, tangential_convective_fluxes=True),
                "left|right|bottom|top")


class PSponge(Pulse):

    def __init__(self,
                 cfg: SolverConfiguration,
                 tree: ResultsDirectoryTree,
                 dB: float,
                 projection: str,
                 factor: int = 1,
                 draw: bool = False) -> None:
        super().__init__(cfg, tree, buffer=True, factor=factor, draw=draw)
        self.cfg.fem = "hdg"

        self.projection = projection
        self.dB = dB

        name = f"psponge_fac{factor}_P{projection}_dB{-dB}"
        self.tree.directory_name = name

        if projection == "k":
            self.order = dcs.PSpongeLayer.range(self.cfg.order)
        elif projection == "0":
            self.order = tuple(dcs.PSpongeLayer.SpongeOrder(k, 0) for k in range(cfg.order, -1, -1))

    def add_meta_data(self):
        super().add_meta_data()
        self.cfg.info["Projection"] = self.projection
        self.cfg.info["Dezibel Reduction"] = self.dB
        self.cfg.info["\u03A0-Sponge Orders"] = self.order
        self.cfg.info["\u03A0-Sponge Lengths X"] = self.lengths_x
        self.cfg.info["\u03A0-Sponge Lengths Y"] = self.lengths_y
        self.cfg.info["\u03A0-Sponge Function X"] = self.sponge_x
        self.cfg.info["\u03A0-Sponge Function Y"] = self.sponge_y
        self.cfg.info["\u03A0-Sponge Weights X"] = self.weights_x
        self.cfg.info["\u03A0-Sponge Weights Y"] = self.weights_y
        self.cfg.info["Grid Factor"] = 15/self.factor
        self.cfg.info["Grid Function X"] = repr(self.grid.x)
        self.cfg.info["Grid Function Y"] = repr(self.grid.y)

    def set_domain_conditions(self):
        super().set_domain_conditions()

        rings = self.main.rings

        # Gridstretching
        x0, xn = self.sound.W/2, sum([ring.W - ring.Wi for ring in rings])/2 + self.sound.W/2
        y0, yn = self.sound.H/2, sum([ring.H - ring.Hi for ring in rings])/2 + self.sound.H/2
        self.x_, self.y_ = BufferCoordinate.x(x0, xn), BufferCoordinate.y(y0, yn)

        grid_x = GridDeformationFunction.ExponentialThickness(15/self.factor, self.x_, True)
        grid_y = GridDeformationFunction.ExponentialThickness(15/self.factor, self.y_, True)
        self.grid = GridDeformationFunction(grid_x, grid_y)

        for ring in rings:
            self.solver.domain_conditions.set(dcs.GridDeformation(self.grid), ring.mat)

        self.weights_x = []
        self.weights_y = []

        self.lengths_x = []
        self.lengths_y = []

        for order, ring in zip(self.order, rings):

            lx = self.grid.x.deformed_length(ring.x_)
            ly = self.grid.y.deformed_length(ring.y_)

            self.lengths_x.append(lx)
            self.lengths_y.append(ly)

            sigma_x = SpongeWeight.quadratic(lx, cfg.Mach_number.Get(), dB=self.dB)
            sigma_y = SpongeWeight.quadratic(ly, cfg.Mach_number.Get(), dB=self.dB)

            self.weights_x.append(sigma_x)
            self.weights_y.append(sigma_y)

            sponge_x = SpongeFunction.quadratic(ring.x_, weight=sigma_x, mirror=True)
            sponge_y = SpongeFunction.quadratic(ring.y_, weight=sigma_y, mirror=True)

            self.solver.domain_conditions.set(dcs.PSpongeLayer(
                order.high, order.low, sponge_x, sponge_y, state=farfield), domain=ring.mat)

        self.sponge_x = sponge_x
        self.sponge_y = sponge_y

    def set_boundary_conditions(self):
        self.solver.boundary_conditions.set(bcs.FarField(farfield), "left|right|bottom|top")


class Sponge(Pulse):

    def __init__(self,
                 cfg: SolverConfiguration,
                 tree: ResultsDirectoryTree,
                 dB: float,
                 draw: bool = False) -> None:
        super().__init__(cfg, tree, buffer=True, draw=draw)
        self.cfg.fem = "hdg"
        self.dB = dB

        name = f"sponge_dB{-dB}"
        self.tree.directory_name = name

    def add_meta_data(self):
        super().add_meta_data()
        self.cfg.info["Dezibel Reduction"] = self.dB
        self.cfg.info["\u03A0-Sponge Length X"] = self.length_x
        self.cfg.info["\u03A0-Sponge Length Y"] = self.length_y
        self.cfg.info["\u03A0-Sponge Function X"] = self.sponge_x
        self.cfg.info["\u03A0-Sponge Function Y"] = self.sponge_y
        self.cfg.info["\u03A0-Sponge Weight X"] = self.weight_x
        self.cfg.info["\u03A0-Sponge Weight Y"] = self.weight_y

    def set_domain_conditions(self):
        super().set_domain_conditions()

        rings = self.main.rings

        # Gridstretching
        x0, xn = self.sound.W/2, sum([ring.W - ring.Wi for ring in rings])/2 + self.sound.W/2
        y0, yn = self.sound.H/2, sum([ring.H - ring.Hi for ring in rings])/2 + self.sound.H/2
        self.x_, self.y_ = BufferCoordinate.x(x0, xn), BufferCoordinate.y(y0, yn)

        grid_x = GridDeformationFunction.ExponentialThickness(15, self.x_, True)
        grid_y = GridDeformationFunction.ExponentialThickness(15, self.y_, True)
        self.grid = GridDeformationFunction(grid_x, grid_y)

        # Sponge Layer
        lx = self.grid.x.deformed_length(self.x_)
        ly = self.grid.y.deformed_length(self.y_)
        sigma_x = SpongeWeight.quadratic(lx, self.cfg.Mach_number.Get(), dB=self.dB)
        sigma_y = SpongeWeight.quadratic(ly, self.cfg.Mach_number.Get(), dB=self.dB)
        sponge_x = SpongeFunction.quadratic(self.x_, sigma_x, mirror=True)
        sponge_y = SpongeFunction.quadratic(self.y_, sigma_y, mirror=True)
        for ring in rings:
            self.solver.domain_conditions.set(dcs.SpongeLayer(farfield, sponge_x, sponge_y), domain=ring.mat)

        self.length_x = lx
        self.length_y = ly
        self.weight_x = sigma_x
        self.weight_y = sigma_y
        self.sponge_x = sponge_x
        self.sponge_y = sponge_y

    def set_boundary_conditions(self):
        self.solver.boundary_conditions.set(bcs.FarField(farfield), "left|right|bottom|top")


if __name__ == "__main__":

    benchmarks = [
        NSCBC(cfg, tree, "1d"),
        NSCBC(cfg, tree, "2d"),
        Sponge(cfg, tree, dB=-40),
        Sponge(cfg, tree, dB=-200),
        PSponge(cfg, tree, dB=-40, projection="0"),
        PSponge(cfg, tree, dB=-200, projection="0"),
        PSponge(cfg, tree, dB=-40, projection="k"),
        PSponge(cfg, tree, dB=-200, projection="k"),
        PSponge(cfg, tree, dB=-40, projection="0", factor=2),
        PSponge(cfg, tree, dB=-200, projection="0", factor=2),
        PSponge(cfg, tree, dB=-40, projection="k", factor=2),
        PSponge(cfg, tree, dB=-200, projection="k", factor=2),
    ]

    for benchmark in benchmarks:
        benchmark.start(True)
