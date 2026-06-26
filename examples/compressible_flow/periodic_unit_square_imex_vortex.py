#%%
# Synchronized IMEX scheme on a periodic unit square with fine center region
# Injects an isentropic vortex for testing

# ------- Import Modules ------- #
import numpy as np
import ngsolve as ngs
from netgen.meshing import IdentificationType
from dream.compressible_flow import CompressibleFlowSolver, flowfields, Initial, InterfaceBC, FarField
from dream.mesh import get_rectangular_mesh
from dream.time import SynchronizedIMEXTimeRoutine

ngs.SetNumThreads(4)

# ------- Create Mesh with Fine Center Region ------- #
# Create a periodic unit square [-0.5, 0.5] x [-0.5, 0.5]
# with a fine implicit region in the center x-direction

# Define x-coordinates: coarse on sides, fine in center
Nx_coarse = 10  # Points per coarse region
Nx_fine = 20    # Points in fine center region

xl = np.linspace(-0.5, -0.125, Nx_coarse // 2 + 1)
xm = np.linspace(-0.125, 0.125, Nx_fine + 1)
xr = np.linspace(0.125, 0.5, Nx_coarse // 2 + 1)

# Define y-coordinates: uniform
Ny = 16
y = np.linspace(-0.5, 0.5, Ny + 1)

domains = (('explicit', (xl, y)),
           ('implicit', (xm, y)),
           ('explicit', (xr, y)))

boundaries = (('bottom', ((xl.min(), xr.max()), y.min())),
              ('right', (xr.max(), y)),
              ('top', ((xl.min(), xr.max()), y.max())),
              ('left', (xl.min(), y)),
              ('interface', (xm.min(), y)),
              ('interface', (xm.max(), y)))

mesh = get_rectangular_mesh(domains, boundaries, quads=True, periodic_x=True, periodic_y=False)

implicit_mesh = ngs.Mesh(mesh.ngmesh.GetSubMesh("implicit", "implicit"))
explicit_mesh = ngs.Mesh(mesh.ngmesh.GetSubMesh("explicit", "explicit"))


# ------- Configure General Setting ------- #
cfg = CompressibleFlowSolver(mesh)
cfg.time = "transient"
cfg.time.timer.interval = (0.0, 5.0)
cfg.time.timer.step = 1e-3
cfg.mach_number = 0.3

cfg.dynamic_viscosity = "inviscid"
cfg.equation_of_state = "ideal"
cfg.equation_of_state.heat_capacity_ratio = 1.4
cfg.scaling = "aerodynamic"
cfg.riemann_solver = "upwind"


# ------- Configure Implicit Solver (HDG) ------- #
cfg_imp = CompressibleFlowSolver(implicit_mesh, **cfg.to_dict())

cfg_imp.fem = "conservative_hdg"
cfg_imp.fem.scheme = "sdirk33"
cfg_imp.fem.scheme.compile = True
cfg_imp.fem.order = 3
cfg_imp.fem.bonus_int_order = 6
cfg_imp.fem.viscous_treatment = None

cfg_imp.fem.solver = "direct"
cfg_imp.fem.solver.method = "newton"
cfg_imp.fem.solver.method.damping_factor = 1
cfg_imp.fem.solver.method.max_iterations = 20
cfg_imp.fem.solver.method.freeze_jacobian = "step"
cfg_imp.fem.solver.method.convergence_criterion = 1e-10

cfg_imp.io.vtk.enable = True
cfg_imp.io.vtk.filename = "hdg"
cfg_imp.io.vtk.rate = 10
cfg_imp.io.vtk.subdivision = 0

# ------- Configure Explicit Solver (DG) ------- #
cfg_exp = CompressibleFlowSolver(explicit_mesh, **cfg.to_dict())

cfg_exp.fem = "conservative_dg"
cfg_exp.fem.scheme = "rk_ars33"
cfg_exp.fem.scheme.compile = True
cfg_exp.fem.order = 3
cfg_exp.fem.bonus_int_order = 6
cfg_exp.fem.viscous_treatment = None

cfg_exp.io.vtk.enable = True
cfg_exp.io.vtk.filename = "dg"
cfg_exp.io.vtk.rate = 10
cfg_exp.io.vtk.subdivision = 0

# ------- Define Initial Vortex (from zero_circulation_vortex_2d.py) ------- #
Uinf = cfg_imp.get_farfield_fields((1, 0))
M = cfg_imp.mach_number
gamma = cfg_imp.equation_of_state.heat_capacity_ratio

Mt = 0.01  # Vortex Mach number
R = 0.1    # Vortex core radius
r = ngs.sqrt((ngs.x - 0.0)**2 + ngs.y**2)  # Centered at origin
vt = Mt / M * cfg_imp.scaling.velocity

# Stream function for the vortex
psi = vt * R * ngs.exp((R**2 - r**2) / (2 * R**2))

# Velocity field: u_0 = Uinf + (∂ψ/∂y, -∂ψ/∂x)
u_0 = Uinf.u + ngs.CF((psi.Diff(ngs.y), -psi.Diff(ngs.x)))

# Pressure and density with isentropic relations
p_0 = Uinf.p * (1 - (gamma - 1) / 2 * Mt**2 * ngs.exp((R**2 - r**2) / (R**2)))**(gamma / (gamma - 1))
rho_0 = Uinf.rho * (1 - (gamma - 1) / 2 * Mt**2 * ngs.exp((R**2 - r**2) / (R**2)))**(1 / (gamma - 1))

initial_fields = flowfields(rho=rho_0, u=u_0, p=p_0)

# ------- Setup Boundary Conditions and Initial Conditions ------- #
# Implicit solver
imp_interface = InterfaceBC(fields=None)
cfg_imp.dcs['implicit'] = Initial(initial_fields)
cfg_imp.bcs['interface'] = imp_interface
cfg_imp.bcs['top|bottom'] = FarField(Uinf)

# Explicit solver
exp_interface = InterfaceBC(fields=None)
cfg_exp.dcs['explicit'] = Initial(initial_fields)
cfg_exp.bcs['interface'] = exp_interface
cfg_exp.bcs['top|bottom'] = FarField(Uinf)
cfg_exp.bcs['left|right'] = "periodic"

# ------- Initialize Solvers ------- #
with ngs.TaskManager():
    cfg_imp.fem.initialize_finite_element_spaces()
    cfg_imp.fem.initialize_trial_and_test_functions()
    cfg_imp.fem.initialize_gridfunctions()
    cfg_imp.fem.initialize_time_scheme_gridfunctions()

    cfg_exp.fem.initialize_finite_element_spaces()
    cfg_exp.fem.initialize_trial_and_test_functions()
    cfg_exp.fem.initialize_gridfunctions()
    cfg_exp.fem.initialize_time_scheme_gridfunctions()

# ------- Link Interface Boundary Conditions ------- #
Uh_imp = cfg_imp.get_all_solution_fields()
Uh_exp = cfg_exp.get_all_solution_fields()

imp_interface.fields = Uh_exp
exp_interface.fields = Uh_imp

# ------- Setup Output Fields ------- #
drawing_imp = cfg_imp.get_solution_fields('p', default_fields=False)
drawing_exp = cfg_exp.get_solution_fields('p', default_fields=False)

p_ref = Uinf.p * (1 - (gamma - 1) / 2 * Mt**2 * ngs.exp(1))**(gamma / (gamma - 1))
drawing_imp['p*'] = (drawing_imp.p - Uinf.p) / (p_ref - Uinf.p)
drawing_exp['p*'] = (drawing_exp.p - Uinf.p) / (p_ref - Uinf.p)

cfg_imp.io.vtk.fields = drawing_imp
cfg_exp.io.vtk.fields = drawing_exp

# ------- Setup Symbolic Forms ------- #
cfg_imp.fem.set_boundary_conditions()
cfg_imp.fem.set_initial_conditions()
cfg_imp.fem.initialize_symbolic_forms()

cfg_exp.fem.set_boundary_conditions()
cfg_exp.fem.set_initial_conditions()
cfg_exp.fem.initialize_symbolic_forms()

# cfg_imp.io.draw(drawing_imp)
# cfg_exp.io.draw(drawing_exp)

# ------- Solve with Synchronized IMEX Time Stepping ------- #
with ngs.TaskManager():
    time_routine = SynchronizedIMEXTimeRoutine(cfg_imp, cfg_exp)
    time_routine.solve()
