import argparse
import ngsolve as ngs
from dream.compressible_flow import CompressibleFlowSolver
from dream.io import IOConfiguration
from config import (TRANSIENT_CFG,
                    get_single_mesh,
                    single_transient_routine,
                    Pulse)


MIXED = {
    'mixed': 'mixed_strain_temperature_gradient',
    'ip': 'interior_penalty',
    'dg': 'interior_penalty'
}


parser = argparse.ArgumentParser(description='Run stable time step tests for IMEX schemes')
parser.add_argument('mixed', type=str, help='Viscous treatment for implicit part', choices=list(MIXED.keys()), default='mixed')
parser.add_argument('dt', type=float, help='Time step size')
parser.add_argument('--threads', type=int, help='Number of threads', default=1)
parser.add_argument('--interval', type=float, nargs=2, help='Time interval', default=(0.0, 1/6))
parser.add_argument('--N', type=int, help='Number of elements', default=40)
parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False)
USER = vars(parser.parse_args())

ngs.SetNumThreads(USER['threads'])

ALPHA = 0.001
X = 4.0

N = USER['N']
dt = USER['dt']
interval = USER['interval']
mixed = MIXED[USER['mixed']]
test = USER['test']

io = IOConfiguration(None)
io.path = f"timings/{N}x{USER['threads']}"

mesh, _, _ = get_single_mesh(N, 2, 1, True)

GENERAL = TRANSIENT_CFG.copy()
GENERAL['time.timer.interval'] = tuple(interval)
GENERAL['time.timer.step'] = dt
if test:
    GENERAL['io.vtk.enable'] = True
    GENERAL['io.vtk.rate'] = 1
    GENERAL['io.vtk.subdivision'] = 2
    GENERAL['io.gfu.enable'] = True
    GENERAL['io.gfu.rate'] = 1
    GENERAL['io.gfu.time_level_rate'] = 1

GENERAL['io.path'] = io.path.joinpath(f"runtimes")
GENERAL['io.log.enable'] = test
if USER['mixed'] != 'dg':
    GENERAL['fem.scheme.compile'] = {'realcompile': True, 'wait': True, 'keep_files': False},

HDG = GENERAL.copy()
HDG['fem'] = 'conservative_hdg'
HDG['fem.scheme'] = 'sdirk33'
HDG['fem.viscous_treatment'] = mixed
if USER['mixed'] == 'ip':
    HDG['fem.viscous_treatment.interior_penalty_coefficient'] = 10.0

IDG = GENERAL.copy()
IDG['fem'] = 'conservative_dg'
IDG['fem.scheme'] = 'sdirk33'
IDG['fem.viscous_treatment'] = 'interior_penalty'
IDG['fem.viscous_treatment.interior_penalty_coefficient'] = 1.0

cfg_full_implicit = CompressibleFlowSolver(mesh)
SETTING = HDG if USER['mixed'] != 'dg' else IDG
FULL_IMPLICIT = Pulse(SETTING, filename=f"full_implicit_{USER['mixed']}", alpha=ALPHA, X=X)
FULL_IMPLICIT.set_conditions(cfg_full_implicit)
single_transient_routine(FULL_IMPLICIT,  test=test)

