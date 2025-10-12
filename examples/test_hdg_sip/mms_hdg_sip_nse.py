""" Method of manufactured solutions (MMS) for compressible flows.

We consider both the Euler and Navier-Stokes equations in 2D on the unit square
domain. The manufactured solutions are chosen as smooth sinusoidal functions.

Literature:
[1] - Roy, C.J., Nelson, C.C., Smith, T.M. and Ober, C.C. (2004), 
      Verification of Euler/Navier–Stokes codes using the method of manufactured solutions. Int. J. Numer. Meth. Fluids, 44: 599-620. https://doi.org/10.1002/fld.660
"""
# Import modules
import ngsolve as ngs
import numpy as np
from typing import NamedTuple
from dream.compressible import CompressibleFlowSolver, Force, FarField, Inflow, Outflow, Initial, flowfields, dimensionalfields, Dirichlet

from dream.compressible.conservative import ConservativeHDG_SIP, ConservativeHDG, ConservativeDG

ngs.ngsglobals.msg_level = 0
ngs.SetNumThreads(4)


# Starting pseudo-time step.
TIME_DT0 = 0.5


# Define manufactured solutions
class Constants(NamedTuple):
    """ Class to define the sinusoidal manufactured solution constants. """
    c: float
    x: float
    y: float
    xy: float
    a_x: float
    a_y: float
    a_xy: float

    def __call__(self, X, Y, XY, L: float = 1.0):
        x_ = ngs.x * ngs.pi/L
        y_ = ngs.y * ngs.pi/L
        xy_ = ngs.x * ngs.y * ngs.pi/L**2
        return (self.c + self.x * X(self.a_x * x_) + self.y * Y(self.a_y * y_) + self.xy * XY(self.a_xy * xy_))


class EulerMMS(NamedTuple):
    rho: Constants = Constants(1.0,  0.15,  -0.1, 0.0, 1.0, 0.5, 0.0)
    u: Constants = Constants(800,  50.0, -30.0, 0.0, 1.5, 0.6, 0.0)
    v: Constants = Constants(800, -75.0,  40.0, 0.0, 0.5, 2/3, 0.0)
    p: Constants = Constants(1e5, 0.2e5, 0.5e5, 0.0, 2.0, 1.0, 0.0)
    R: float = 287.0
    gamma: float = 1.4
    L: float = 1.0

    def set_dimensional_fields(self, cfg: CompressibleFlowSolver):
        # Parameters for dimensionalization
        c_p = self.gamma * self.R / (self.gamma - 1)
        T = self.p.c / (self.R * self.rho.c)

        # Pass dimensional fields to determine dimensionless numbers
        cfg.equation_of_state.heat_capacity_ratio = self.gamma
        cfg.dimensional_fields = dimensionalfields(rho_inf=self.rho.c, u_inf=self.u.c,
                                                   T_inf=T, c_p=c_p, L=self.L)

    def set_conditions(self, cfg: CompressibleFlowSolver) -> flowfields:

        cfg.dynamic_viscosity = "inviscid"

        cfg.bcs.clear()
        cfg.dcs.clear()

        self.set_dimensional_fields(cfg)
        Ue = self.get_fields(cfg)
        U0 = self.get_initial_fields(cfg)

        cfg.dcs['default'] = Initial(fields=U0)
        cfg.dcs['default'] = self.get_forcing(cfg, 10)
        cfg.bcs['left|right|top|bottom'] = Dirichlet(fields=Ue)

        return Ue

    def get_fields(self, cfg: CompressibleFlowSolver) -> flowfields:
        # Construct dimensionless fields
        U = flowfields()
        U.rho = self.rho(ngs.sin, ngs.cos, ngs.cos)/cfg.scaling.reference_density
        U.u = ngs.CF((self.u(ngs.sin, ngs.cos, ngs.cos), self.v(
            ngs.cos, ngs.sin, ngs.cos)))/cfg.scaling.reference_velocity
        U.p = self.p(ngs.cos, ngs.sin, ngs.sin)/cfg.scaling.reference_pressure
        U.rho_E = cfg.energy(U)

        return U

    def get_initial_fields(self, cfg: CompressibleFlowSolver) -> flowfields:
        # Construct dimensionless initial fields from constant terms only
        U = flowfields()
        U.rho = self.rho.c/cfg.scaling.reference_density
        U.u = ngs.CF((self.u.c, self.v.c))/cfg.scaling.reference_velocity
        U.p = self.p.c/cfg.scaling.reference_pressure
        U.rho_E = cfg.energy(U)
        return U

    def get_forcing(self, cfg: CompressibleFlowSolver, order: int = 0) -> Force:

        def div(F):
            return ngs.CF(tuple(F[i, 0].Diff(ngs.x) + F[i, 1].Diff(ngs.y) for i in range(F.dims[0])))

        U = self.get_fields(cfg)
        F = div(cfg.get_convective_flux(U))
        return Force(F[0], (F[1], F[2]), F[3], order)

    def __str__(self):
        return "EE"


class NavierStokesMMS(NamedTuple):
    rho: Constants = Constants(1.0,    0.1,  0.15,    0.08, 0.75,  1.0, 1.25)
    u: Constants = Constants(70.0,    4.0, -12.0,     7.0,  5/3,  1.5, 0.6)
    v: Constants = Constants(90.0,  -20.0,   4.0,   -11.0,  1.5,  1.0, 0.9)
    p: Constants = Constants(1.0e5, -0.3e5, 0.2e5, -0.25e5,  1.0, 1.25, 0.75)
    #rho: Constants = Constants(1.0,    0.1,  0.15,    0.08, 0.75,  1.0, 1.25)
    #u: Constants = Constants(10.0,     4.0,  -5.0,     7.0,  5/3,  1.5, 0.6)
    #v: Constants = Constants(5.0,    -10.0,   4.0,    -6.0,  1.5,  1.0, 0.9)
    #p: Constants = Constants(1.0e5, -0.3e5, 0.2e5, -0.25e5,  1.0, 1.25, 0.75)
    
    R: float = 287.0
    gamma: float = 1.4
    L: float = 1.0
    mu: float = 10.0
    Pr: float = 1.0

    def set_dimensional_fields(self, cfg: CompressibleFlowSolver):
        # Parameters for dimensionalization
        c_p = self.gamma * self.R / (self.gamma - 1)
        T = self.p.c / (self.R * self.rho.c)

        k_inf = self.mu * c_p / self.Pr
        
        magu = ngs.sqrt( self.u.c**2 + self.v.c**2 )

        # Pass dimensional fields to determine dimensionless numbers
        cfg.equation_of_state.heat_capacity_ratio = self.gamma
        cfg.dimensional_fields = dimensionalfields(rho_inf=self.rho.c, u_inf=magu, T_inf=T,
                                                   mu_inf=self.mu, k_inf=k_inf, c_p=c_p, L=self.L)

    def set_conditions(self, cfg: CompressibleFlowSolver) -> flowfields:

        cfg.dynamic_viscosity = "constant"

        cfg.bcs.clear()
        cfg.dcs.clear()

        self.set_dimensional_fields(cfg)
        Ue = self.get_fields(cfg)
        U0 = self.get_initial_fields(cfg)

        cfg.dcs['default'] = Initial(fields=U0)
        cfg.dcs['default'] = self.get_forcing(cfg, 10)
        cfg.bcs['left|right|top|bottom'] = Dirichlet(fields=Ue)

        return Ue

    def get_fields(self, cfg: CompressibleFlowSolver) -> flowfields:
        # Construct dimensionless fields
        U = flowfields()
        U.rho = self.rho(ngs.sin, ngs.cos, ngs.cos)/cfg.scaling.reference_density
        U.u = ngs.CF((self.u(ngs.sin, ngs.cos, ngs.cos), self.v(
            ngs.cos, ngs.sin, ngs.cos)))/cfg.scaling.reference_velocity
        U.p = self.p(ngs.cos, ngs.sin, ngs.sin)/cfg.scaling.reference_pressure
        U.rho_u = cfg.momentum(U)
        U.rho_E = cfg.energy(U)

        U.grad_rho = ngs.CF((U.rho.Diff(ngs.x), U.rho.Diff(ngs.y)))
        U.grad_p = ngs.CF((U.p.Diff(ngs.x), U.p.Diff(ngs.y)))
        U.grad_u = ngs.CF((U.u[0].Diff(ngs.x), U.u[0].Diff(ngs.y),
                           U.u[1].Diff(ngs.x), U.u[1].Diff(ngs.y)), dims=(2, 2))
        U.grad_T = cfg.temperature_gradient(U, U)

        return U

    def get_initial_fields(self, cfg: CompressibleFlowSolver) -> flowfields:
        # Construct dimensionless initial fields from constant terms only
        U = flowfields()
        U.rho = self.rho.c/cfg.scaling.reference_density
        U.u = ngs.CF((self.u.c, self.v.c))/cfg.scaling.reference_velocity
        U.p = self.p.c/cfg.scaling.reference_pressure
        U.rho_E = cfg.energy(U)

        return U

    def get_forcing(self, cfg: CompressibleFlowSolver, order: int = 0) -> Force:

        def div(F):
            return ngs.CF(tuple(F[i, 0].Diff(ngs.x) + F[i, 1].Diff(ngs.y) for i in range(F.dims[0])))

        U = self.get_fields(cfg)
        FG = div(cfg.get_convective_flux(U) - cfg.get_diffusive_flux(U, U))
        return Force(FG[0], (FG[1], FG[2]), FG[3], order)

    def __str__(self):
        return "NS"


# Define refinement levels, polynomial orders and simulations to run
LEVELS = 4
ORDERS = [1, 2, 3, 4, 5]
SIMULATIONS = [NavierStokesMMS()]


draw = False
write_vtk = True

# Setup solution routine
def mms_routine(func):

    def polynomial_order_routine(cfg: CompressibleFlowSolver, simulation: EulerMMS | NavierStokesMMS, order: int, level: int):

        # Set Finite Element configuration
        func(cfg, simulation, order)

        # Set polynomial order
        cfg.fem.order = order

        #if level > 2:
        #    cfg.fem.scheme.compile = True
        
        # Reset the starting time step.
        cfg.time.timer.step = TIME_DT0

        # Initialize the solver
        cfg.initialize()

        # Get solution fields
        fields = cfg.get_solution_fields("rho_u", "rho_E")

        if draw:
            cfg.io.undraw()
            cfg.io.draw(fields)
        
        if write_vtk:
            path = "hdg_ees_pseudotime" 
            if isinstance(simulation, NavierStokesMMS):
                path = "hdg_nse_pseudotime"
            fn = f"mms_{cfg.fem.order}_{level}" 
            
            fields["Mach"] = cfg.get_local_mach_number( fields )
            fields["Re"] = cfg.get_local_reynolds_number( fields ) * cfg.reynolds_number.Get()
            
            ue = simulation.get_fields(cfg)
            fields["RelErr[rho]"] = 100.0 * (ue.rho - fields.rho)/ue.rho
            fields["RelErr[p]"] = 100.0 * (ue.p - fields.p)/ue.p
            
            cfg.io.vtk.fields = fields 
            cfg.io.vtk.enable=True
            cfg.io.vtk.rate = 10000000 # just output the last step.
            cfg.io.vtk.path = path 
            cfg.io.vtk.filename = fn 

        # Display the statistics of certain quantities.
        display_statistics_unit_square(cfg, simulation)
        
        # Display the output.
        print("###############################################################################################################")
        print(f"Level: {level}, Order: {order}")
        print("###############################################################################################################")

        # Solve the system.
        with ngs.TaskManager():
            cfg.solve()

        return fields

    def display_statistics_unit_square(cfg: CompressibleFlowSolver, simulation: EulerMMS | NavierStokesMMS) -> None:
        print("###############################################################################################################")

        # Mesh points.
        x_vals = np.linspace(0, 1, 50)
        y_vals = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
        Xf = X.flatten()
        Yf = Y.flatten()

        # Analytic solution.
        ue = simulation.get_fields(cfg)
        
        # Pressure.
        p_exact = cfg.pressure(ue)
        p_vals = p_exact(cfg.mesh(x=Xf, y=Yf))
        p_min = cfg.scaling.reference_pressure * p_vals.min()
        p_max = cfg.scaling.reference_pressure * p_vals.max()
        p_avg = cfg.scaling.reference_pressure * p_vals.mean()
        print(f"  pressure(min): {p_min:.5e},\t pressure(max): {p_max:.5e},\t pressure(mean): {p_avg:.5e}")

        # Reynolds number.
        Re_exact = cfg.get_local_reynolds_number(ue) 
        Re_vals = Re_exact(cfg.mesh(x=Xf, y=Yf))
        Re_min = cfg.reynolds_number.Get() * Re_vals.min() 
        Re_max = cfg.reynolds_number.Get() * Re_vals.max()
        Re_avg = cfg.reynolds_number.Get() * Re_vals.mean()
        print(f"  Reynolds(min): {Re_min:.5e},\t Reynolds(max): {Re_max:.5e},\t Reynolds(mean): {Re_avg:.5e}")

        # Reynolds number.
        M_exact = cfg.get_local_mach_number(ue) 
        M_vals = M_exact(cfg.mesh(x=Xf, y=Yf))
        M_min = M_vals.min() 
        M_max = M_vals.max()
        M_avg = M_vals.mean()
        print(f"      Mach(min): {M_min:.5e},\t     Mach(max): {M_max:.5e},\t     Mach(mean): {M_avg:.5e}")


    def simulation_routine(cfg: CompressibleFlowSolver, level, simulation: EulerMMS | NavierStokesMMS):

        # Set necessary conditions
        Ue = simulation.set_conditions(cfg)

        L2 = {}
        for order in ORDERS:

            qorder = 25
            # Solve for different polynomial orders
            Uh = polynomial_order_routine(cfg, simulation, order, level)
            L2[order] = {'rho': ngs.sqrt(ngs.Integrate((Uh.rho - Ue.rho)**2, cfg.mesh, order=qorder)),
                         'rhou': ngs.sqrt(ngs.Integrate((Uh.rho_u[0] - Ue.rho_u[0])**2, cfg.mesh, order=qorder)),
                         'rhov': ngs.sqrt(ngs.Integrate((Uh.rho_u[1] - Ue.rho_u[1])**2, cfg.mesh, order=qorder)),
                         'rhoE': ngs.sqrt(ngs.Integrate((Uh.rho_E - Ue.rho_E)**2, cfg.mesh, order=qorder))}

        return L2

    def decorator(*args, **kwargs):

        # Define initial mesh size
        MAXH = 0.5
        mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=MAXH, quad_dominated=True))

        # Define common solver configuration
        cfg = CompressibleFlowSolver(mesh)
        cfg.scaling = "aerodynamic"
        #cfg.riemann_solver = "lax_friedrich"
        cfg.riemann_solver = "upwind"
        #cfg.riemann_solver = "roe"

        cfg.time = "pseudo_time_stepping"
        cfg.time.timer.step = TIME_DT0
        cfg.time.max_time_step = 50
        cfg.time.increment_at = 10
        cfg.time.increment_factor = 10

        L2 = {}
        for LEVEL in range(LEVELS):

            # Refine Mesh
            if LEVEL > 0:
                mesh.Refine()
                MAXH *= 0.5

            L2[MAXH] = {}
            for simulation in SIMULATIONS:
                L2[MAXH][str(simulation)] = simulation_routine(cfg, LEVEL, simulation)

        return L2

    return decorator


@mms_routine
def conservative_hdg(cfg: CompressibleFlowSolver, simulation: EulerMMS | NavierStokesMMS, order: int):
    # Set only finite element configuration
    #cfg.fem = 'conservative_hdg'
    cfg.fem = 'conservative_hdg_sip'

    cfg.fem.order = order
    cfg.fem.scheme = "implicit_euler"

    cfg.fem.solver = "direct"
    cfg.fem.solver.method = "newton"
    #cfg.fem.solver.method.damping_factor = 1.0
    cfg.fem.solver.method.max_iterations = 300
    cfg.fem.solver.method.convergence_criterion = 1e-18

    #cfg.fem.bonus_int_order = order + 10
    alpha = 10.0
    if isinstance(simulation, NavierStokesMMS):
        if isinstance(cfg.fem, ConservativeDG):
            cfg.fem.viscous_treatment = "interior_penalty"
            cfg.fem.viscous_treatment.interior_penalty_coefficient = alpha
        elif isinstance(cfg.fem, ConservativeHDG_SIP):
            cfg.fem.viscous_treatment = "interior_penalty"
            cfg.fem.viscous_treatment.interior_penalty_coefficient = alpha
        elif isinstance(cfg.fem, ConservativeHDG):
            cfg.fem.mixed_method = "strain_heat"
        else:
            raise ValueError(f"Specified viscous conditions in HDG are not implemented.")


# Run the simulations and collect errors
ERROR = conservative_hdg()

# Plot the results
import matplotlib.pyplot as plt
import numpy as np
import csv


H = sorted(ERROR.keys(), reverse=True)

fig, axes_grid = plt.subplots(2, 4, figsize=(10, 6)) 
axes = {(str(sim), field): axes_grid[i, j]
        for i, sim in enumerate(SIMULATIONS)
        for j, field in enumerate(['rho', 'rhou', 'rhov', 'rhoE'])}

# Open the CSV file and write the plot data. 
with open("mms_hdg_errors_pseudotime.csv", "w", newline="") as f:
    
    # Write the header information.
    writer = csv.writer(f)
    f.write(f"levels (refinement): {LEVELS}\n")
    f.write("\n")
    f.write("------------------------------------------------------------------------\n")
    f.write("h, dt, eq, field, order, error\n")
    f.write("------------------------------------------------------------------------\n")
    
    for key, ax in axes.items():
        sim, field = key 

        for order in ORDERS:
            errors = [ERROR[h][sim][order][field] for h in H]
            ax.loglog(H, errors, marker='o', label=fr"$p={order}$")

            # Save the data onto the file. 
            for h, err in zip(H, errors):
                dt = 1.0 # just a dummy, for the output.
                field_fmt = field
                if field in ["u", "p"]:
                    field_fmt = f"  {field}" # prepend two spaces
                row = [
                        f"{h:.15e}", f"{dt:.15e}", sim, field_fmt, str(order), f"{err:.15e}"
                ]
                f.write(",\t".join(row) + "\n")

        ax.set_xlabel(r"$h$")
        ax.set_title(rf"${field}$")
        ax.legend()

plt.tight_layout()

## Save the whole figure in vector formats.
#plt.savefig("mms_hdg_errors_pseudotime.pdf")
#plt.savefig("mms_hdg_errors_pseudotime.svg")

plt.show()



