# Import modules
import ngsolve  as ngs
from config import TRANSIENT_CFG, time_refinement_routine, EulerMMS, NavierStokesMMS, get_gassner_mms, get_geometry

ngs.SetNumThreads(8)

import argparse
parser = argparse.ArgumentParser(description='Transient MMS Gassner benchmark')
parser.add_argument('fem', type=str, help='Finite element method', choices=['dg', 'hdg'])
parser.add_argument('simulation', type=str, help='Euler or Navier-Stokes', choices=['euler', 'navier-stokes'])
parser.add_argument('--order', type=int, help='order', default=3)
parser.add_argument('--periods', type=int, help='Temporal periods', default=5)
parser.add_argument('--viscous', type=str, help='Viscous treatment', default='interior_penalty')
parser.add_argument('--IP', type=float, help='Interior penalty coefficient', default=10.0)
OPTIONS = vars(parser.parse_args())

fem = OPTIONS['fem']
simulation_type = OPTIONS['simulation']
order = OPTIONS['order']
periods = OPTIONS['periods']
viscous = OPTIONS['viscous']
IP = OPTIONS['IP']

filename = f"{fem}_order{order}_{viscous}"
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
GENERAL['fem'] = ... # to be set later
GENERAL['fem.solver'] = 'direct'
GENERAL['fem.order'] = order
GENERAL['fem.solver.method'] = 'newton'
GENERAL['fem.solver.method.max_iterations'] = 10
GENERAL['fem.solver.method.convergence_criterion'] = 1e-20
GENERAL['fem.solver.method.damping_factor'] = 1.0
GENERAL['fem.bonus_int_order'] = 10
GENERAL['fem.scheme.compile'] = False
GENERAL['io.vtk.enable'] = False
GENERAL['io.vtk.rate'] = 10
GENERAL['io.vtk.subdivision'] = 2
GENERAL['io.path'] = f"transient_gassner_{simulation_type}/{filename}"
if simulation_type == 'navier-stokes':
    GENERAL['fem.viscous_treatment'] = viscous
    if viscous == 'interior_penalty':
        GENERAL['fem.viscous_treatment.interior_penalty_coefficient'] = IP


# HDG Navier-Stokes setting
HDG = GENERAL.copy()
HDG['fem'] = 'conservative_hdg'

# DG Navier-Stokes setting
DG = GENERAL.copy()
DG['fem'] = 'conservative_dg'

if fem == 'hdg':
    CFG = HDG
    schemes = ['implicit_euler', 'bdf2', 'sdirk22', 'sdirk33']
    schemes = ['bdf3', 'bdf2']
    schemes = ['sdirk22', 'sdirk33', 'bdf3']
elif fem == 'dg':
    CFG = DG
    schemes = ['explicit_euler']

if simulation_type == 'navier-stokes':
    FEM = NavierStokesMMS(CFG, get_gassner_mms, filename=filename, periods=periods, is_periodic=True)
else:
    FEM = EulerMMS(CFG, get_gassner_mms, filename=filename, periods=periods, is_periodic=True)


mesh = ngs.Mesh(get_geometry(FEM.is_periodic).GenerateMesh(maxh=0.5, quad_dominated=True))
for i in range(2): # 8x8
    mesh.Refine()

f_ny = periods/(time_interval[1] - time_interval[0])
dt_min = 1/(2*f_ny)
dt = [dt_min/(2**i) for i in range(1,5)]

if __name__ == "__main__":
    # Start Routine
    time_refinement_routine(mesh, schemes, dt, FEM)
