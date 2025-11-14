# Import modules
import argparse
import ngsolve as ngs
from config import TRANSIENT_CFG, imex_time_refinement_routine, EulerMMS, NavierStokesMMS, get_gassner_mms, get_geometry
ngs.SetNumThreads(8)

import argparse
parser = argparse.ArgumentParser(description='Transient MMS Gassner benchmark')
parser.add_argument('simulation', type=str, help='Euler or Navier-Stokes', choices=['euler', 'navier-stokes'])
parser.add_argument('pair', type=int, help='pair')
parser.add_argument('--order', type=int, help='order', default=3)
parser.add_argument('--periods', type=int, help='Temporal periods', default=5)
parser.add_argument('--viscous', type=str, help='Viscous treatment', default='interior_penalty')
parser.add_argument('--IP', type=float, help='Interior penalty coefficient', default=10.0)
OPTIONS = vars(parser.parse_args())

simulation_type = OPTIONS['simulation']
order = OPTIONS['order']
PAIR = OPTIONS['pair']
periods = OPTIONS['periods']
viscous = OPTIONS['viscous']
IP = OPTIONS['IP']

filename = f"imex_order{order}_{viscous}"
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
GENERAL['io.vtk.enable'] = True
GENERAL['io.vtk.rate'] = 1
GENERAL['io.vtk.subdivision'] = 2
GENERAL['io.gfu.enable'] = False
GENERAL['io.gfu.rate'] = 1
GENERAL['io.gfu.time_level_rate'] = 10
GENERAL['io.vtk.subdivision'] = 2
GENERAL['io.path'] = f"transient_gassner_imex_{simulation_type}/{filename}"


# HDG Navier-Stokes setting
HDG = GENERAL.copy()
HDG['fem'] = 'conservative_hdg'
HDG['fem.scheme'] = "implicit_euler"

# DG Navier-Stokes setting
DG = GENERAL.copy()
DG['fem'] = 'conservative_dg'
DG['fem.scheme'] = "explicit_euler"
if simulation_type == 'navier-stokes':
    HDG['fem.viscous_treatment'] = viscous
    if viscous == 'interior_penalty':
        HDG['fem.viscous_treatment.interior_penalty_coefficient'] = IP
    DG['fem.viscous_treatment'] = "interior_penalty"
    DG['fem.viscous_treatment.interior_penalty_coefficient'] = IP

PAIRS = [
    {("implicit_euler", "explicit_euler"): 0.00025},
    {("sdirk22", "rk_ars22"): 0.0005   },
    {("sdirk33", "rk_ars33"): 0.001  },
    {("sdirk43", "rk_ars43"): 0.001    },
]

if simulation_type == 'navier-stokes':
    HDG = NavierStokesMMS(HDG, get_gassner_mms, filename=f"hdg_{filename}", periods=periods, domain="implicit", bnds="bottom|top|right")
    DG = NavierStokesMMS(DG, get_gassner_mms, filename=f"dg_{filename}", periods=periods, domain="explicit", bnds="bottom|left|top")
else:
    HDG = EulerMMS(HDG, get_gassner_mms, filename=f"hdg_{filename}", periods=periods, domain="implicit", bnds="bottom|top|right")
    DG = EulerMMS(DG, get_gassner_mms, filename=f"dg_{filename}", periods=periods, domain="explicit", bnds="bottom|left|top")

mesh = ngs.Mesh(get_geometry(imex=True).GenerateMesh(maxh=0.5, quad_dominated=True))
explicit_mesh = ngs.Mesh(mesh.ngmesh.GetSubMesh(domains="explicit", faces="explicit"))
implicit_mesh = ngs.Mesh(mesh.ngmesh.GetSubMesh(domains="implicit", faces="implicit"))
for i in range(2):  # 8x8
    explicit_mesh.Refine()
    implicit_mesh.Refine()

if __name__ == "__main__":
    imex_time_refinement_routine(explicit_mesh, implicit_mesh, DG, HDG, PAIRS[PAIR], levels=5)
