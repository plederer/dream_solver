import argparse
import ngsolve as ngs
from dream.compressible import CompressibleFlowSolver
from dream.io import IOConfiguration
from config import (TRANSIENT_CFG,
                    get_single_mesh,
                    single_stable_time_step_routine,
                    imex_stable_time_step_routine,
                    Pulse)

ngs.SetNumThreads(4)

SPLITTING = [
    'explicit',
    'imex'
]

MIXED = {
    'mixed': 'mixed_strain_temperature_gradient',
    'ip': 'interior_penalty'
}



parser = argparse.ArgumentParser(description='Run stable time step tests for IMEX schemes')
parser.add_argument('splitting', type=str, help='Splitting type', choices=SPLITTING)
parser.add_argument('dt', type=float, help='Time step size')
parser.add_argument('--interval', type=float, nargs=2, help='Time interval', default=(0.0, 0.15))
parser.add_argument('--r', type=int, help='First mesh size ratio', default=1)
parser.add_argument('--tol', type=float, help='Tolerance for stable time step', default=1e-5)
parser.add_argument('--Ni', type=int, help='Number of implicit elements', default=8)
parser.add_argument('--N', type=int, help='Number of elements', default=40)
parser.add_argument('--mixed', type=str, help='Viscous treatment for implicit part',
                    choices=list(MIXED.keys()), default='mixed')
USER = vars(parser.parse_args())

ALPHA = 0.1
X = 0.1

r = USER['r']
N = USER['N']
Ni = USER['Ni']
dt = USER['dt']
tol = USER['tol']
interval = USER['interval']
mixed = MIXED[USER['mixed']]

dxi0 = 1.0 / (r*N)


io = IOConfiguration(None)
io.path = f"{N}x{Ni}/r{r}"

mesh, mesh_implicit, mesh_explicit = get_single_mesh(N, Ni, dxi0)

GENERAL = TRANSIENT_CFG.copy()
GENERAL['time.timer.interval'] = tuple(interval)
GENERAL['time.timer.step'] = dt
GENERAL['io.path'] = io.path.joinpath("stable_time_steps")
GENERAL['fem.scheme.compile'] = True

HDG = GENERAL.copy()
HDG['fem'] = 'conservative_hdg'
HDG['fem.scheme'] = 'sdirk33'
if USER['mixed'] == 'mixed':
    HDG['fem.viscous_treatment'] = 'mixed_strain_temperature_gradient'
else:
    HDG['fem.viscous_treatment'] = 'interior_penalty'
    HDG['fem.viscous_treatment.interior_penalty_coefficient'] = 10.0
HDG['io.sensor.enable'] = True

DG = GENERAL.copy()
DG['fem'] = 'conservative_dg'
DG['fem.scheme'] = 'rk_ars33'
DG['fem.viscous_treatment'] = 'interior_penalty'
DG['fem.viscous_treatment.interior_penalty_coefficient'] = 1.0

if USER['splitting'] == 'explicit':

    cfg_full_explicit = CompressibleFlowSolver(mesh)
    FULL_EXPLICIT = Pulse(DG, filename="full_explicit", alpha=ALPHA, X=X)
    FULL_EXPLICIT.set_conditions(cfg_full_explicit)
    outputfile = cfg_full_explicit.io.path.joinpath('explicit.csv')
    single_stable_time_step_routine(FULL_EXPLICIT, outputfile=outputfile, tol=tol)

else:

    implicit_cfg = CompressibleFlowSolver(mesh_implicit)
    IMEX_IMPLICIT = Pulse(HDG, filename=f"imex_implicit_{Ni}_{USER['mixed']}", domain="implicit", boundaries='top|bottom', alpha=ALPHA, X=X)
    IMEX_IMPLICIT.set_conditions(implicit_cfg)

    explicit_cfg = CompressibleFlowSolver(mesh_explicit)
    IMEX_EXPLICIT = Pulse(DG, filename=f"imex_explicit_{Ni}_{USER['mixed']}", domain="explicit", alpha=ALPHA, X=X)
    IMEX_EXPLICIT.set_conditions(explicit_cfg)

    outputfile = implicit_cfg.io.path.joinpath(f'imex_{Ni}_{USER["mixed"]}.csv')
    imex_stable_time_step_routine(IMEX_IMPLICIT, IMEX_EXPLICIT, outputfile=outputfile, tol=tol)
