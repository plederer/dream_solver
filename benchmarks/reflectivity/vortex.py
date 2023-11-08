
from dream import ResultsDirectoryTree
from ngsolve import *
from netgen.occ import OCCGeometry
from dream import *
from dream.utils import RectangleDomain, Rectangle1DGrid
from benchmarks.template import Benchmark

ngsglobals.msg_level = 0
SetNumThreads(8)

tree = ResultsDirectoryTree()


class Vortex(Benchmark):

    def settings(self, Mach_number: float) -> SolverConfiguration:

        cfg = SolverConfiguration()
        cfg.formulation = "conservative"
        cfg.scaling = "aeroacoustic"
        cfg.riemann_solver = 'lax_friedrich'

        cfg.Mach_number = Mach_number
        cfg.heat_capacity_ratio = 1.4

        cfg.order = 6
        cfg.bonus_int_order_bnd = cfg.order
        cfg.bonus_int_order_vol = cfg.order

        cfg.time.simulation = "transient"
        cfg.time.scheme = "BDF2"
        cfg.time.step = 0.05
        cfg.time.interval = (0, 30)

        cfg.linear_solver = "pardiso"
        cfg.damping_factor = 1
        cfg.max_iterations = 100
        cfg.convergence_criterion = 1e-16

        cfg.compile_flag = True
        cfg.static_condensation = True

        # Farfield
        rho_inf = INF.density(cfg)
        u_inf = CF(INF.velocity((1, 0), cfg))
        p_inf = INF.pressure(cfg)
        self.farfield = State(u_inf, rho_inf, p_inf)

        # Vortex Isothermal
        gamma = cfg.heat_capacity_ratio
        T_inf = INF.temperature(cfg)
        c = INF.speed_of_sound(cfg)

        Gamma = 0.01
        Rv = 0.1
        r = sqrt(x**2 + y**2)
        psi = Gamma * exp(-r**2/(2*Rv**2))

        cfg.info["Gamma"] = Gamma
        cfg.info["Radius"] = Rv

        # Vortex Isothermal
        u_0 = u_inf + CF((psi.Diff(y), -psi.Diff(x)))
        p_0 = p_inf * exp(-gamma/2*(Gamma/(c * Rv))**2 * exp(-r**2/Rv**2))
        rho_0 = rho_inf * exp(-gamma/2*(Gamma/(c * Rv))**2 * exp(-r**2/Rv**2))
        self.p_00 = p_inf * exp(-gamma/2*(Gamma/(c * Rv))**2)
        self.initial = State(u_0, rho_0, p_0)

        return cfg

    def __init__(self,
                 tree: ResultsDirectoryTree,
                 M: float,
                 maxh: float = 0.15,
                 draw: bool = False) -> None:

        cfg = self.settings(M)

        super().__init__(cfg, tree, draw)

        self.sound = RectangleDomain(H=2, W=2, mat="sound", maxh=maxh)
        # self.order = dcs.PSpongeLayer.range(cfg.order)
        self.order = tuple(dcs.PSpongeLayer.SpongeOrder(i, 0) for i in range(cfg.order, -1, -1))

    def add_meta_data(self):
        self.cfg.info["Sound Domain"] = (f"W: {self.sound.W}, H: {self.sound.H}")
        self.cfg.info["Dofs"] = self.solver.formulation.fes.ndof

    def get_mesh(self):
        main = Rectangle1DGrid(self.sound)

        sponge_length = self.sound.W*4
        maxhs = sponge_length/len(self.order)
        maxhs = [self.sound.maxh*1.25**i for i in range(1, len(self.order) + 1)]
        fronts = [RectangleDomain(W=3*maxh, mat=f"order_{k.high}", maxh=maxh) for (k, maxh) in zip(self.order, maxhs)]

        for front in fronts:
            main.add_front(front)
        main.add_periodic('top', 'bottom')

        geo = OCCGeometry(main.get_face(), dim=2)
        self.main = main

        return Mesh(geo.GenerateMesh())

    def set_boundary_conditions(self):
        self.solver.boundary_conditions.set(bcs.FarField(self.farfield), "left|right")
        self.solver.boundary_conditions.set(bcs.Periodic(), "top|bottom")

    def set_domain_conditions(self):
        self.solver.domain_conditions.set(dcs.Initial(self.initial))

        fronts = self.main.front

        # Gridstretching
        x0, xn = self.sound.W/2, sum([front.W for front in fronts]) + self.sound.W/2
        self.x_ = BufferCoordinate.x(x0, xn)

    def draw_scenes(self):
        self.solver.drawer.draw_acoustic_pressure(self.farfield.pressure, autoscale=False, min=-1e-2, max=1e-2)
        self.solver.drawer.draw_particle_velocity(self.farfield.velocity)
        self.solver.drawer.draw_mach_number()
        Draw((self.solver.formulation.pressure() - self.farfield.pressure)/(self.p_00 - self.farfield.pressure), self.solver.mesh, "p*")


class PSponge(Vortex):

    def __init__(
            self, tree: ResultsDirectoryTree, M: float, dB: float = -40, func: str = "constant", maxh: float = 0.15,
            draw: bool = False) -> None:

        self.dB = dB
        self.func = func
        tree.directory_name = f"psponge_vortex_dB{-dB}_M{M}_{func}"
        super().__init__(tree, M, maxh, draw)

    def add_meta_data(self):
        super().add_meta_data()
        self.cfg.info["Sound Domain"] = (f"W: {self.sound.W}, H: {self.sound.H}")
        self.cfg.info["Dezibel Reduction"] = self.dB
        self.cfg.info["Function"] = self.func
        self.cfg.info["\u03A0-Sponge Orders"] = self.order
        self.cfg.info["\u03A0-Sponge Lengths X"] = self.lengths_x
        self.cfg.info["\u03A0-Sponge Function X"] = self.sponge_x
        self.cfg.info["\u03A0-Sponge Weights X"] = self.weights_x

    def set_domain_conditions(self):
        super().set_domain_conditions()

        fronts = self.main.front

        self.weights_x = []
        self.lengths_x = []

        for order, front in zip(self.order, fronts):
            lx = front.x_.length
            self.lengths_x.append(lx)

            if self.func == "constant":
                sigma_x = SpongeWeight.constant(lx, self.cfg.Mach_number.Get(), dB=self.dB)
                sponge_x = SpongeFunction.constant(weight=sigma_x)
            elif self.func == "quadratic":
                sigma_x = SpongeWeight.quadratic(lx, self.cfg.Mach_number.Get(), dB=self.dB)
                sponge_x = SpongeFunction.quadratic(front.x_, weight=sigma_x, mirror=False)

            self.weights_x.append(sigma_x)

            self.solver.domain_conditions.set(dcs.PSpongeLayer(
                order.high, order.low, sponge_x, state=self.farfield), domain=front.mat)

        self.sponge_x = sponge_x


class Sponge(Vortex):

    def __init__(
            self, tree: ResultsDirectoryTree, M: float, dB: float = -40, func: str = "constant", maxh: float = 0.15,
            draw: bool = False) -> None:

        self.dB = dB
        self.func = func
        tree.directory_name = f"sponge_vortex_dB{-dB}_M{M}_{func}"
        super().__init__(tree, M, maxh, draw)

    def add_meta_data(self):
        super().add_meta_data()
        self.cfg.info["Sound Domain"] = (f"W: {self.sound.W}, H: {self.sound.H}")
        self.cfg.info["Dezibel Reduction"] = self.dB
        self.cfg.info["Function"] = self.func
        self.cfg.info["\u03A0-Sponge Length X"] = self.length_x
        self.cfg.info["\u03A0-Sponge Function X"] = self.sponge_x
        self.cfg.info["\u03A0-Sponge Weight X"] = self.weight_x

    def set_domain_conditions(self):
        super().set_domain_conditions()

        fronts = self.main.front

        # Sponge Layer
        lx = self.x_.length
        sigma_x = SpongeWeight.quadratic(lx, self.cfg.Mach_number.Get(), dB=self.dB)
        sponge_x = SpongeFunction.quadratic(self.x_, sigma_x, mirror=True)
        for front in fronts:
            self.solver.domain_conditions.set(dcs.SpongeLayer(self.farfield, sponge_x), domain=front.mat)

        self.length_x = lx
        self.weight_x = sigma_x
        self.sponge_x = sponge_x


class NSCBC(Vortex):

    def __init__(self, tree: ResultsDirectoryTree, M: float, maxh: float = 0.15, draw: bool = False) -> None:
        tree.directory_name = f"nscbc_vortex_dB_M{M}"
        super().__init__(tree, M, maxh, draw)
        self.cfg.bonus_int_order_bnd = 3*self.cfg.order
        self.cfg.bonus_int_order_vol = 3*self.cfg.order

    def add_meta_data(self):
        super().add_meta_data()
        self.cfg.info["Sigma"] = 0.5
        self.cfg.info["Reference Length"] = 2

    def get_mesh(self):
        main = Rectangle1DGrid(self.sound)
        main.add_periodic('top', 'bottom')
        geo = OCCGeometry(main.get_face(), dim=2)
        self.main = main

        return Mesh(geo.GenerateMesh())

    def set_boundary_conditions(self):
        self.solver.boundary_conditions.set(bcs.FarField(self.farfield), "left")
        self.solver.boundary_conditions.set(bcs.NSCBC(
            self.farfield.pressure, sigma=0.5, reference_length=2
        ), "right")
        self.solver.boundary_conditions.set(bcs.Periodic(), "top|bottom")

    def set_domain_conditions(self):
        self.solver.domain_conditions.set(dcs.Initial(self.initial))

        fronts = self.main.front

        # Gridstretching
        x0, xn = self.sound.W/2, sum([front.W for front in fronts]) + self.sound.W/2
        self.x_ = BufferCoordinate.x(x0, xn)


if __name__ == "__main__":

    Mach_numbers = [0.3, 0.6]

    for Mach in Mach_numbers:

        benchmarks = [
            PSponge(tree, Mach, dB=-20, func="quadratic"),
            PSponge(tree, Mach, dB=-40, func="quadratic"),
            Sponge(tree, Mach, dB=-20, func="quadratic"),
            Sponge(tree, Mach, dB=-40, func="quadratic"),
            NSCBC(tree, Mach),
        ]

        for benchmark in benchmarks:
            benchmark.start(True)
