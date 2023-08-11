
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
cfg.bonus_int_order_bnd = cfg.order
cfg.bonus_int_order_vol = cfg.order

cfg.time.simulation = "transient"
cfg.time.scheme = "BDF2"
cfg.time.step = 0.2
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
                 dB: float = -40,
                 func: float = "constant",
                 maxh: float = 0.4,
                 draw: bool = False) -> None:

        name = self.__class__.__name__.lower()
        name += f"_dB{-dB}_{func}"
        tree.directory_name = name

        super().__init__(cfg, tree, draw)

        self.sound = RectangleDomain(H=2, W=2, mat="sound", maxh=maxh)
        self.dB = dB
        self.func = func
        self.order = dcs.PSpongeLayer.range(cfg.order)

    def add_meta_data(self):
        self.cfg.info["Sound Domain"] = (f"W: {self.sound.W}, H: {self.sound.H}")
        self.cfg.info["Dezibel Reduction"] = self.dB
        self.cfg.info["Function"] = self.func

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
        x0, xn = self.sound.W/2, sum([ring.W - ring.Wi for ring in rings])/2 + self.sound.W/2
        y0, yn = self.sound.H/2, sum([ring.H - ring.Hi for ring in rings])/2 + self.sound.H/2
        self.x_, self.y_ = BufferCoordinate.x(x0, xn), BufferCoordinate.y(y0, yn)

    def draw_scenes(self):
        self.solver.drawer.draw_acoustic_pressure(p_inf, autoscale=False, min=-1e-2, max=1e-2)


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

            lx = ring.x_.length
            ly = ring.y_.length

            self.lengths_x.append(lx)
            self.lengths_y.append(ly)

            if self.func == "constant":
                sigma_x = SpongeWeight.constant(lx, self.cfg.Mach_number.Get(), dB=self.dB)
                sigma_y = SpongeWeight.constant(ly, self.cfg.Mach_number.Get(), dB=self.dB)
                sponge_x = SpongeFunction.constant(weight=sigma_x)
                sponge_y = SpongeFunction.constant(weight=0)

            elif self.func == "quadratic":
                sigma_x = SpongeWeight.quadratic(lx, cfg.Mach_number.Get(), dB=self.dB)
                sigma_y = SpongeWeight.quadratic(ly, cfg.Mach_number.Get(), dB=self.dB)
                sponge_x = SpongeFunction.quadratic(ring.x_, weight=sigma_x, mirror=True)
                sponge_y = SpongeFunction.quadratic(ring.y_, weight=sigma_y, mirror=True)

            self.weights_x.append(sigma_x)
            self.weights_y.append(sigma_y)

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
        lx = self.x_.length
        ly = self.y_.length

        if self.func == "constant":
            sigma_x = SpongeWeight.constant(lx, self.cfg.Mach_number.Get(), dB=self.dB)
            sigma_y = SpongeWeight.constant(ly, self.cfg.Mach_number.Get(), dB=self.dB)
            sponge_x = SpongeFunction.constant(weight=sigma_x)
            sponge_y = SpongeFunction.constant(weight=0)

        elif self.func == "quadratic":
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
        PSponge(cfg, tree, dB=-20, func="constant"),
        PSponge(cfg, tree, dB=-40, func="constant"),
        PSponge(cfg, tree, dB=-80, func="constant"),
        PSponge(cfg, tree, dB=-120, func="constant"),
        PSponge(cfg, tree, dB=-160, func="constant"),
        PSponge(cfg, tree, dB=-200, func="constant"),
        PSponge(cfg, tree, dB=-20, func="quadratic"),
        PSponge(cfg, tree, dB=-40, func="quadratic"),
        PSponge(cfg, tree, dB=-80, func="quadratic"),
        PSponge(cfg, tree, dB=-120, func="quadratic"),
        PSponge(cfg, tree, dB=-160, func="quadratic"),
        PSponge(cfg, tree, dB=-200, func="quadratic"),
        Sponge(cfg, tree, dB=-20, func="constant"),
        Sponge(cfg, tree, dB=-40, func="constant"),
        Sponge(cfg, tree, dB=-80, func="constant"),
        Sponge(cfg, tree, dB=-120, func="constant"),
        Sponge(cfg, tree, dB=-160, func="constant"),
        Sponge(cfg, tree, dB=-200, func="constant"),
        Sponge(cfg, tree, dB=-20, func="quadratic"),
        Sponge(cfg, tree, dB=-40, func="quadratic"),
        Sponge(cfg, tree, dB=-80, func="quadratic"),
        Sponge(cfg, tree, dB=-120, func="quadratic"),
        Sponge(cfg, tree, dB=-160, func="quadratic"),
        Sponge(cfg, tree, dB=-200, func="quadratic"),
    ]

    for benchmark in benchmarks:
        benchmark.start(True)
