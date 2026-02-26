import argparse
import ngsolve as ngs
from dream.compressible import CompressibleFlowSolver
from dream.io import IOConfiguration
from config import (TRANSIENT_CFG,
                    get_single_mesh,
                    single_transient_routine,
                    imex_transient_routine,
                    Pulse)

ngs.SetNumThreads(1)

SPLITTING = [
    'explicit',
    'imex',
    'implicit'
]

MIXED = {
    'mixed': 'mixed_strain_temperature_gradient',
    'ip': 'interior_penalty',
    'dg': 'interior_penalty'
}


parser = argparse.ArgumentParser(description='Run stable time step tests for IMEX schemes')
parser.add_argument('splitting', type=str, help='Splitting type', choices=SPLITTING)
parser.add_argument('dt', type=float, help='Time step size')
parser.add_argument('--interval', type=float, nargs=2, help='Time interval', default=(0.0, 1.0))
parser.add_argument('--r', type=int, help='First mesh size ratio', default=1)
parser.add_argument('--Ni', type=int, help='Number of implicit elements', default=2)
parser.add_argument('--N', type=int, help='Number of elements', default=40)
parser.add_argument('--mixed', type=str, help='Viscous treatment for implicit part',
                    choices=list(MIXED.keys()), default='mixed')
parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False)
USER = vars(parser.parse_args())

ALPHA = 0.001
X = 0.1

r = USER['r']
N = USER['N']
Ni = USER['Ni']
dt = USER['dt']
interval = USER['interval']
mixed = MIXED[USER['mixed']]
test = USER['test']

dxi0 = 1.0 / (r*N)

io = IOConfiguration(None)
io.path = f"{N}x{Ni}/r{r}"

mesh, mesh_implicit, mesh_explicit = get_single_mesh(N, Ni, dxi0)

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
# GENERAL['fem.scheme.compile'] = {'realcompile': True, 'wait': True, 'keep_files': False},


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

DG = GENERAL.copy()
DG['fem'] = 'conservative_dg'
DG['fem.scheme'] = 'rk_ars33'
DG['fem.viscous_treatment'] = 'interior_penalty'
DG['fem.viscous_treatment.interior_penalty_coefficient'] = 1.0
if test and USER['splitting'] == 'explicit':
    DG['io.sensor.enable'] = True

if USER['splitting'] == 'explicit':

    cfg_full_explicit = CompressibleFlowSolver(mesh)
    FULL_EXPLICIT = Pulse(DG, filename="full_explicit", alpha=ALPHA, X=X)
    FULL_EXPLICIT.set_conditions(cfg_full_explicit)
    single_transient_routine(FULL_EXPLICIT,  test=test)

elif USER['splitting'] == 'implicit':

    cfg_full_implicit = CompressibleFlowSolver(mesh)
    FULL_IMPLICIT = Pulse(HDG, filename=f"full_implicit_{USER['mixed']}", alpha=ALPHA, X=X)
    FULL_IMPLICIT.set_conditions(cfg_full_implicit)
    single_transient_routine(FULL_IMPLICIT,  test=test)

elif USER['splitting'] == 'imex' and USER['mixed'] == 'dg':

    implicit_cfg = CompressibleFlowSolver(mesh_implicit)
    IMEX_IMPLICIT = Pulse(IDG, filename=f"imex_implicit_{Ni}_{USER['mixed']}", domain="implicit", boundaries='top|bottom', alpha=ALPHA, X=X)
    IMEX_IMPLICIT.set_conditions(implicit_cfg)

    explicit_cfg = CompressibleFlowSolver(mesh_explicit)
    IMEX_EXPLICIT = Pulse(DG, filename=f"imex_explicit_{Ni}_{USER['mixed']}", domain="explicit", alpha=ALPHA, X=X)
    IMEX_EXPLICIT.set_conditions(explicit_cfg)

    imex_transient_routine(IMEX_IMPLICIT, IMEX_EXPLICIT, test=test)

else:

    implicit_cfg = CompressibleFlowSolver(mesh_implicit)
    IMEX_IMPLICIT = Pulse(HDG, filename=f"imex_implicit_{Ni}_{USER['mixed']}", domain="implicit", boundaries='top|bottom', alpha=ALPHA, X=X)
    IMEX_IMPLICIT.set_conditions(implicit_cfg)

    explicit_cfg = CompressibleFlowSolver(mesh_explicit)
    IMEX_EXPLICIT = Pulse(DG, filename=f"imex_explicit_{Ni}_{USER['mixed']}", domain="explicit", alpha=ALPHA, X=X)
    IMEX_EXPLICIT.set_conditions(explicit_cfg)

    imex_transient_routine(IMEX_IMPLICIT, IMEX_EXPLICIT, test=test)
