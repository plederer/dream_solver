# Import modules
import argparse
import ngsolve as ngs
from config import TRANSIENT_CFG, imex_time_refinement_routine, time_refinement_routine, NavierStokesMMS, get_gassner_mms, get_hp_geometry
ngs.SetNumThreads(8)

parser = argparse.ArgumentParser(description='Transient MMS Gassner benchmark')
parser.add_argument('simulation', type=str, help='Implicit, Explicit  or Imex', choices=['implicit', 'explicit', 'imex'])
parser.add_argument('--order', type=int, help='order', default=3)
parser.add_argument('--periods', type=int, help='Temporal periods', default=1)
parser.add_argument('--viscous', type=str, help='Viscous treatment', default='interior_penalty')
parser.add_argument('--IP', type=float, help='Interior penalty coefficient', default=10.0)
OPTIONS = vars(parser.parse_args())

simulation_type = OPTIONS['simulation']
order = OPTIONS['order']
periods = OPTIONS['periods']
viscous = OPTIONS['viscous']
IP = OPTIONS['IP']

filename = f"{simulation_type}_order{order}_{viscous}"
if viscous == 'interior_penalty':
    filename += f"_IP{IP}"

time_interval = [0.0, 1.0]

GENERAL = TRANSIENT_CFG.copy()
GENERAL['reynolds_number'] = 565
GENERAL['prandtl_number'] = 1.0
GENERAL['equation_of_state.heat_capacity_ratio'] = 1.4
GENERAL['riemann_solver'] = 'upwind'
GENERAL['scaling'] = 'aerodynamic'
GENERAL['time.timer.step'] = 0.05
GENERAL['time.timer.interval'] = time_interval
GENERAL['fem'] = ...  # to be set later
GENERAL['fem.solver'] = 'direct'
GENERAL['fem.order'] = order
GENERAL['fem.solver.method'] = 'newton'
GENERAL['fem.solver.method.max_iterations'] = 10
GENERAL['fem.solver.method.convergence_criterion'] = 1e-20
GENERAL['fem.solver.method.damping_factor'] = 1.0
GENERAL['fem.bonus_int_order'] = 10
GENERAL['fem.scheme.compile'] = False
GENERAL['io.vtk.enable'] = False
GENERAL['io.vtk.rate'] = 1
GENERAL['io.vtk.subdivision'] = 2
GENERAL['io.gfu.enable'] = False
GENERAL['io.gfu.rate'] = 1
GENERAL['io.gfu.time_level_rate'] = 10
GENERAL['io.vtk.subdivision'] = 2
GENERAL['io.path'] = f"transient_gassner_benchmark/{filename}"

# GENERAL['timestep_controller'] = "physical_controller"
# GENERAL['timestep_controller.rate'] = 10
# GENERAL['timestep_controller.cfl'] = 1

# HDG Navier-Stokes setting
HDG = GENERAL.copy()
HDG['fem'] = 'conservative_hdg'
HDG['fem.scheme'] = "implicit_euler"

# DG Navier-Stokes setting
DG = GENERAL.copy()
DG['fem'] = 'conservative_dg'
DG['fem.scheme'] = "explicit_euler"
HDG['fem.viscous_treatment'] = viscous
if viscous == 'interior_penalty':
    HDG['fem.viscous_treatment.interior_penalty_coefficient'] = IP
DG['fem.viscous_treatment'] = "interior_penalty"
DG['fem.viscous_treatment.interior_penalty_coefficient'] = IP

IMPLICIT_PAIRS = {'implicit_euler': 0.1,
                  'sdirk22': 0.1,
                  'sdirk33': 0.1
                  }

IMEX_PAIRS = {("implicit_euler", "explicit_euler"): 0.001,
    ("sdirk22", "rk_ars22"): 0.002,
    ("sdirk33", "rk_ars33"): 0.002
}

EXPLICIT_PAIRS = {"explicit_euler": 1e-5,
                  "rk_ars22": 2e-5,
                  "rk_ars33": 4e-5,
                  }

mesh = ngs.Mesh(get_hp_geometry())
explicit_mesh = ngs.Mesh(mesh.ngmesh.GetSubMesh(domains="explicit", faces="explicit"))
implicit_mesh = ngs.Mesh(mesh.ngmesh.GetSubMesh(domains="implicit", faces="implicit"))
for _ in range(2):  # 8x8
    mesh.Refine()
mesh.RefineHP(3, 0.5)

for _ in range(2):  # 8x8
    explicit_mesh.Refine()
    implicit_mesh.Refine()
implicit_mesh.RefineHP(3, 0.5)

if __name__ == "__main__":
    if simulation_type == 'implicit':
        HDG = NavierStokesMMS(HDG, get_gassner_mms, filename=f"hdg_{filename}", periods=periods, domain="implicit|explicit")
        time_refinement_routine(mesh, IMPLICIT_PAIRS, HDG, levels=5)
    elif simulation_type == 'explicit':
        DG = NavierStokesMMS(DG, get_gassner_mms, filename=f"dg_{filename}", periods=periods, domain="implicit|explicit")
        time_refinement_routine(mesh, EXPLICIT_PAIRS, DG, levels=5)
    elif simulation_type == 'imex':
        HDG = NavierStokesMMS(HDG, get_gassner_mms, filename=f"hdg_{filename}", periods=periods, domain="implicit", bnds="bottom|left|right")
        DG = NavierStokesMMS(DG, get_gassner_mms, filename=f"dg_{filename}", periods=periods, domain="explicit", bnds="right|top|left")
        imex_time_refinement_routine(explicit_mesh, implicit_mesh, DG, HDG, IMEX_PAIRS, levels=5)

