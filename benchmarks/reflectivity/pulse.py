
from ngsolve import *
from netgen.occ import OCCGeometry
from dream import *
from dream.utils import RectangleDomain, RectangleGrid
from benchmarks.template import Benchmark
from math import log

ngsglobals.msg_level = 0
SetNumThreads(8)

tree = ResultsDirectoryTree()

cfg = SolverConfiguration()
cfg.formulation = "conservative"
cfg.scaling = "acoustic"
cfg.riemann_solver = 'lax_friedrich'

cfg.Mach_number = 0
cfg.heat_capacity_ratio = 1.4

cfg.order = 6
cfg.bonus_int_order = {VOL: cfg.order, BND: cfg.order}

cfg.time.simulation = "transient"
cfg.time.scheme = "BDF2"
cfg.time.step = 0.05
cfg.time.interval = (0, 25)

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 100
cfg.convergence_criterion = 1e-16

cfg.compile_flag = True
cfg.static_condensation = True

# Far-field values
rho_inf = 1
u_inf = cfg.Mach_number * CF((0, 0))
p_inf = 1/cfg.heat_capacity_ratio
farfield = State(u_inf, rho_inf, p_inf)

# Initial values
AMP, beta = 0.8, 0.25
rho_0 = rho_inf
u_0 = u_inf
p_0 = p_inf * (1 + AMP * exp(-log(2)*(x**2 + y**2)/beta**2))
initial = State(u_0, rho_0, p_0)


class Pulse(Benchmark):

    def __init__(self,
                 cfg: SolverConfiguration,
                 tree: ResultsDirectoryTree,
                 grid_factor: int = 5,
                 dB: float = -40,
                 maxh: float = 0.4,
                 draw: bool = False) -> None:

        name = self.__class__.__name__.lower()
        name += f"_dB{-dB}_gf{grid_factor}"
        tree.directory_name = name

        super().__init__(cfg, tree, draw)

        self.sound = RectangleDomain(H=2, W=2, mat="sound", maxh=maxh)
        self.grid_factor = grid_factor
        self.dB = dB
        self.order = dcs.PSpongeLayer.range(cfg.order)

    def add_meta_data(self):
        self.cfg.info["Sound Domain"] = (f"W: {self.sound.W}, H: {self.sound.H}")
        self.cfg.info["Dezibel Reduction"] = self.dB
        self.cfg.info["Grid Factor"] = self.grid_factor
        self.cfg.info["Grid Function X"] = repr(self.grid_x)
        self.cfg.info["Grid Function Y"] = repr(self.grid_y)

    def get_mesh(self):
        main = RectangleGrid(self.sound)

        sponge_length = self.sound.W
        maxh = sponge_length/len(self.order)
        rings = [RectangleDomain(2*maxh, 2*maxh, mat=str(k.high), maxh=maxh) for k in self.order]

        for ring in rings:
            main.add_ring(ring)

        geo = OCCGeometry(main.get_face(), dim=2)
        self.main = main

        return Mesh(geo.GenerateMesh())

    def set_boundary_conditions(self):
        self.solver.boundary_conditions.set(bcs.Outflow(p_inf), "left|right|bottom|top")

    def set_domain_conditions(self):
        self.solver.domain_conditions.set(dcs.Initial(initial))

        rings = self.main.rings

        # Gridstretching
        x0, xn = self.sound.W/2, sum([ring.W - ring.Wi for ring in rings])/2 + self.sound.W/2
        y0, yn = self.sound.H/2, sum([ring.H - ring.Hi for ring in rings])/2 + self.sound.H/2
        self.x_, self.y_ = BufferCoordinate.x(x0, xn), BufferCoordinate.y(y0, yn)

        self.grid_x = GridDeformationFunction.ExponentialThickness(self.grid_factor, self.x_, True)
        self.grid_y = GridDeformationFunction.ExponentialThickness(self.grid_factor, self.y_, True)

        self.grid = GridDeformationFunction(self.grid_x, self.grid_y)

        for ring in rings:
            self.solver.domain_conditions.set(dcs.GridDeformation(self.grid), ring.mat)

    def draw_scenes(self):
        self.solver.drawer.draw()
        self.solver.drawer.draw_acoustic_pressure(p_inf, autoscale=False, min=-1e-2, max=1e-2)
        self.solver.drawer.draw_mach_number()


class PSponge(Pulse):

    def add_meta_data(self):
        super().add_meta_data()
        self.cfg.info["\u03A0-Sponge Orders"] = self.order
        self.cfg.info["\u03A0-Sponge Lengths X"] = self.lengths_x
        self.cfg.info["\u03A0-Sponge Lengths Y"] = self.lengths_y
        self.cfg.info["\u03A0-Sponge Function X"] = self.sponge_x
        self.cfg.info["\u03A0-Sponge Function Y"] = self.sponge_y
        self.cfg.info["\u03A0-Sponge Weights X"] = self.weights_x
        self.cfg.info["\u03A0-Sponge Weights Y"] = self.weights_y

    def set_domain_conditions(self):
        super().set_domain_conditions()

        rings = self.main.rings

        self.weights_x = []
        self.weights_y = []

        self.lengths_x = []
        self.lengths_y = []

        for order, ring in zip(self.order, rings):

            lx = self.grid_x.deformed_length(ring.x_)
            ly = self.grid_y.deformed_length(ring.y_)

            self.lengths_x.append(lx)
            self.lengths_y.append(ly)

            # sigma_x = SpongeWeight.constant(lx, self.cfg.Mach_number.Get(), dB=self.dB)
            # sigma_y = SpongeWeight.constant(ly, self.cfg.Mach_number.Get(), dB=self.dB)
            sigma_x = SpongeWeight.quadratic(lx, cfg.Mach_number.Get(), dB=self.dB)
            sigma_y = SpongeWeight.quadratic(ly, cfg.Mach_number.Get(), dB=self.dB)

            self.weights_x.append(sigma_x)
            self.weights_y.append(sigma_y)

            # sponge_x = SpongeFunction.constant(weight=sigma_x)
            # sponge_y = SpongeFunction.constant(weight=0)
            sponge_x = SpongeFunction.quadratic(ring.x_, weight=sigma_x, mirror=True)
            sponge_y = SpongeFunction.quadratic(ring.y_, weight=sigma_y, mirror=True)

            self.solver.domain_conditions.set(dcs.PSpongeLayer(
                order.high, order.low, sponge_x, sponge_y, state=farfield), domain=ring.mat)

        self.sponge_x = sponge_x
        self.sponge_y = sponge_y


class Sponge(Pulse):

    def add_meta_data(self):
        super().add_meta_data()
        self.cfg.info["\u03A0-Sponge Length X"] = self.length_x
        self.cfg.info["\u03A0-Sponge Length Y"] = self.length_y
        self.cfg.info["\u03A0-Sponge Function X"] = self.sponge_x
        self.cfg.info["\u03A0-Sponge Function Y"] = self.sponge_y
        self.cfg.info["\u03A0-Sponge Weight X"] = self.weight_x
        self.cfg.info["\u03A0-Sponge Weight Y"] = self.weight_y

    def set_domain_conditions(self):
        super().set_domain_conditions()

        rings = self.main.rings

        # Sponge Layer
        lx = self.grid_x.deformed_length(self.x_)
        ly = self.grid_y.deformed_length(self.y_)
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


if __name__ == "__main__":

    benchmarks = [
        PSponge(cfg, tree, grid_factor=5, dB=-20),
        PSponge(cfg, tree, grid_factor=10, dB=-20),
        PSponge(cfg, tree, grid_factor=15, dB=-20),
        PSponge(cfg, tree, grid_factor=5, dB=-40),
        PSponge(cfg, tree, grid_factor=10, dB=-40),
        PSponge(cfg, tree, grid_factor=15, dB=-40),
        PSponge(cfg, tree, grid_factor=5, dB=-60),
        PSponge(cfg, tree, grid_factor=10, dB=-60),
        PSponge(cfg, tree, grid_factor=15, dB=-60),
        Sponge(cfg, tree, grid_factor=5, dB=-20),
        Sponge(cfg, tree, grid_factor=10, dB=-20),
        Sponge(cfg, tree, grid_factor=15, dB=-20),
        Sponge(cfg, tree, grid_factor=5, dB=-40),
        Sponge(cfg, tree, grid_factor=10, dB=-40),
        Sponge(cfg, tree, grid_factor=15, dB=-40),
        Sponge(cfg, tree, grid_factor=5, dB=-60),
        Sponge(cfg, tree, grid_factor=10, dB=-60),
        Sponge(cfg, tree, grid_factor=15, dB=-60),
    ]

    for benchmark in benchmarks:
        benchmark.start(True)
