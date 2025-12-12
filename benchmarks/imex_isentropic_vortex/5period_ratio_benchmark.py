# Import modules
import argparse
import ngsolve as ngs
from config import (TRANSIENT_CFG, 
                    get_meshes,
                    multizone_imex_time_refinement_routine, 
                    local_imex_time_refinement_routine, 
                    time_refinement_routine, 
                    FastVortex, SlowVortex)

ngs.SetNumThreads(4)

parser = argparse.ArgumentParser(description='Transient MMS Gassner benchmark')
parser.add_argument('simulation', type=str, help='Implicit, Explicit  or Imex', choices=['implicit', 'explicit', 'imex', 'local_imex'])
parser.add_argument('vortex', type=str, help='Vortex type', choices=['fast', 'slow'])
parser.add_argument('--ratio', type=int, help='ratio', default=1)
parser.add_argument('--levels', type=int, help='levels', default=1)
parser.add_argument('--label', type=str, help='label', default='')
OPTIONS = vars(parser.parse_args())

simulation_type = OPTIONS['simulation']
vortex = OPTIONS['vortex']
ratio = OPTIONS['ratio']
levels = OPTIONS['levels']
label = OPTIONS['label']

filename = f"{simulation_type}_{vortex}"
if label != "":
    filename += f"_{label}"
if simulation_type == "local_imex" or simulation_type == "implicit":
    filename += f"_{ratio}"

time_interval = [0.0, 10.0]
dte = 0.00025


GENERAL = TRANSIENT_CFG.copy()
GENERAL['equation_of_state.heat_capacity_ratio'] = 1.4
GENERAL['riemann_solver'] = 'upwind'
GENERAL['time.timer.interval'] = time_interval
GENERAL['fem'] = ...  # to be set later
GENERAL['fem.solver'] = 'direct'
GENERAL['fem.order'] = 3
GENERAL['fem.solver.method'] = 'newton'
GENERAL['fem.solver.method.max_iterations'] = 5
GENERAL['fem.solver.method.convergence_criterion'] = 1e-20
GENERAL['fem.solver.method.damping_factor'] = 1.0
GENERAL['fem.bonus_int_order'] = 6
GENERAL['fem.scheme.compile'] = True
GENERAL['io.vtk.enable'] = False
GENERAL['io.vtk.rate'] = 1
GENERAL['io.vtk.subdivision'] = 2
GENERAL['io.gfu.enable'] = True
GENERAL['io.gfu.rate'] = 1
GENERAL['io.gfu.time_level_rate'] = 1000
GENERAL['io.vtk.subdivision'] = 2
GENERAL['io.path'] = f"ratio_benchmark_{dte:.6e}/{filename}"

# HDG Navier-Stokes setting
HDG = GENERAL.copy()
HDG['fem'] = 'conservative_hdg'
HDG['fem.scheme'] = "implicit_euler"

# DG Navier-Stokes setting
DG = GENERAL.copy()
DG['fem'] = 'conservative_dg'
DG['fem.scheme'] = "explicit_euler"

IMPLICIT_PAIRS = {'implicit_euler': ratio * dte,
                #   'sdirk22': 0.1,
                #   'sdirk33': 0.1
                  }

IMEX_PAIRS = {("implicit_euler", "explicit_euler"): 0.001,
                ("sdirk22", "rk_ars22"): 0.001,
                ("sdirk33", "rk_ars33"): 0.001,
}

LOCAL_IMEX_PAIRS = {
    ("implicit_euler", "explicit_euler"): (ratio, dte),
}

EXPLICIT_PAIRS = {"explicit_euler": dte,
                #   "rk_ars22": 0.001,
                #   "rk_ars33": 0.001,
                  }

mesh, implicit_mesh, explicit_mesh = get_meshes(32, 32)

if vortex == 'fast':
    VORTEX = FastVortex
elif vortex == 'slow':
    VORTEX = SlowVortex
    
if __name__ == "__main__":

    if simulation_type == 'implicit':
        HDG = VORTEX(HDG, filename=f"hdg_{filename}")
        runtimes = time_refinement_routine(mesh, IMPLICIT_PAIRS, HDG, levels=levels)

    elif simulation_type == 'explicit':
        DG = VORTEX(DG,  filename=f"dg_{filename}")
        runtimes = time_refinement_routine(mesh, EXPLICIT_PAIRS, DG, levels=levels)

    elif simulation_type == 'imex':
        HDG = VORTEX(HDG,  filename=f"hdg_{filename}", domain="implicit", periodic="bottom|top")
        DG = VORTEX(DG,  filename=f"dg_{filename}", domain="explicit_left|explicit_right")
        runtimes = multizone_imex_time_refinement_routine(explicit_mesh, implicit_mesh, DG, HDG, IMEX_PAIRS, levels=levels)

    elif simulation_type == 'local_imex':
        HDG = VORTEX(HDG,  filename=f"hdg_{filename}", domain="implicit", periodic="bottom|top")
        DG = VORTEX(DG,  filename=f"dg_{filename}", domain="explicit_left|explicit_right")
        runtimes = local_imex_time_refinement_routine(explicit_mesh, implicit_mesh, DG, HDG, LOCAL_IMEX_PAIRS, levels=levels)

    print("Runtimes:", runtimes)
    with open(f"{GENERAL['io.path']}/runtimes.txt", "w") as f:
        for key, value in runtimes.items():
            f.write(f"{key}: {value}\n")

