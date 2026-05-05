# Import modules
import argparse
import ngsolve as ngs
from dream.time import (TimeRoutine, SynchronizedIMEXTimeRoutine, PCIMEXTimeRoutine, LinearPCIMEXTimeRoutine)
from dream.compressible import CompressibleFlowSolver
from config import (TRANSIENT_CFG,
                    STAGE_TO_SCHEME,
                    get_uniform_meshes,
                    get_squashed_meshes,
                    single_transient_routine,
                    imex_transient_routine,
                    FastVortex, SlowVortex)

ngs.SetNumThreads(4)

ROUTINES = {
    'implicit': TimeRoutine,
    'explicit': TimeRoutine,
    'imex_sync': SynchronizedIMEXTimeRoutine,
    'imex_pc_frozen': PCIMEXTimeRoutine,
    'imex_pc_linear': LinearPCIMEXTimeRoutine
}

VORTICES = {
    'fast': FastVortex,
    'slow': SlowVortex
}

MESH = {
    'uniform': get_uniform_meshes,
    'squashed': get_squashed_meshes
}

parser = argparse.ArgumentParser(description='Transient MMS Gassner benchmark')
parser.add_argument('routine', type=str, help='Implicit, Explicit  or Imex', choices=list(ROUTINES))
parser.add_argument('vortex', type=str, help='Vortex type', choices=list(VORTICES))
parser.add_argument('number_of_stages', type=int, help='Number of stages', choices=[1, 2, 3])

parser.add_argument('--dte', type=float, help='Explicit Time step size', default=0.001)
parser.add_argument('--dti', type=float, help='Implicit Time step size', default=0.001)
parser.add_argument('--mesh', type=str, help='Mesh type', choices=list(MESH), default='uniform')

parser.add_argument('--vtkrate', type=int, help='vtkrate', default=1)
parser.add_argument('--gfurate', type=int, help='gfurate', default=1)
USER = vars(parser.parse_args())

dte = USER['dte']
dti = USER['dti']
imp_scheme, exp_scheme = STAGE_TO_SCHEME[USER['number_of_stages']]

output = f"5periods_{USER['vortex']}/{USER['routine']}/"
if USER['routine'] == "implicit":
    output += f"{imp_scheme}_{dti:.6e}"
elif USER['routine'] == "explicit":
    output += f"{exp_scheme}_{dte:.6e}"
elif USER['routine'].startswith("imex"):
    output += f"{imp_scheme}_{dti:.6e}_{exp_scheme}_{dte:.6e}"

GENERAL = TRANSIENT_CFG.copy()
GENERAL['equation_of_state.heat_capacity_ratio'] = 1.4
GENERAL['riemann_solver'] = 'upwind'
GENERAL['time.timer.interval'] = (0.0, 10.0)
GENERAL['fem'] = ...  # to be set later
GENERAL['fem.solver'] = 'direct'
GENERAL['fem.order'] = 3
GENERAL['fem.solver.method'] = 'newton'
GENERAL['fem.solver.method.max_iterations'] = 5
GENERAL['fem.solver.method.convergence_criterion'] = 1e-14
GENERAL['fem.solver.method.damping_factor'] = 1.0
GENERAL['fem.bonus_int_order'] = 6
GENERAL['fem.scheme.compile'] = True
GENERAL['io.sensor.enable'] = True
GENERAL['io.vtk.enable'] = False
GENERAL['io.vtk.rate'] = USER['vtkrate']
GENERAL['io.vtk.subdivision'] = 2
GENERAL['io.gfu.enable'] = False
GENERAL['io.gfu.rate'] = USER['gfurate']
GENERAL['io.gfu.time_level_rate'] = 100
GENERAL['io.vtk.subdivision'] = 2
GENERAL['io.path'] = output

# HDG Navier-Stokes setting
HDG = GENERAL.copy()
HDG['fem'] = 'conservative_hdg'
HDG['fem.scheme'] = imp_scheme
HDG['time.timer.step'] = dti

# DG Navier-Stokes setting
DG = GENERAL.copy()
DG['fem'] = 'conservative_dg'
DG['fem.scheme'] = exp_scheme
DG['time.timer.step'] = dte

mesh, implicit_mesh, explicit_mesh = MESH[USER['mesh']](32, 32)


def get_simulation(mesh, defaults, **kwargs):
    cfg = CompressibleFlowSolver(mesh)
    simulation = VORTICES[USER['vortex']](defaults, **kwargs)
    simulation.set_conditions(cfg)

    return simulation


if __name__ == "__main__":

    if USER['routine'] == 'implicit':
        hdg = get_simulation(mesh, HDG, filename="hdg")
        single_transient_routine(hdg)

    elif USER['routine'] == 'explicit':
        dg = get_simulation(mesh, DG, filename="dg")
        single_transient_routine(dg)

    elif USER['routine'].startswith('imex'):
        hdg = get_simulation(implicit_mesh, HDG, filename="hdg", domain="implicit", periodic="bottom|top")
        dg = get_simulation(explicit_mesh, DG, filename="dg", domain="explicit_left|explicit_right")
        routine = ROUTINES[USER['routine']](hdg.cfg, dg.cfg)
        imex_transient_routine(routine, hdg, dg)
