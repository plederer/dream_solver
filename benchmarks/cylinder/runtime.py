import argparse
import ngsolve as ngs
from dream.compressible import CompressibleFlowSolver
from dream.io import IOConfiguration
from config import (TRANSIENT_CFG,
                    get_imex_mesh_from_single_mesh,
                    single_transient_routine,
                    imex_transient_routine,
                    Cylinder)

ngs.SetNumThreads(4)

SPLITTING = [
    'explicit',
    'imex',
    'implicit'
]

MIXED = {
    'mixed': 'mixed_strain_temperature_gradient',
    'ip': 'interior_penalty'
}


parser = argparse.ArgumentParser(description='Run stable time step tests for IMEX schemes')
parser.add_argument('splitting', type=str, help='Splitting type', choices=SPLITTING)
parser.add_argument('dt', type=float, help='Time step size')
parser.add_argument('--Ni', type=int, help='Number of radial elements', default=16)
parser.add_argument('--mixed', type=str, help='Viscous treatment for implicit part',
                    choices=list(MIXED.keys()), default='mixed')
parser.add_argument('--log', action=argparse.BooleanOptionalAction, default=False)

USER = vars(parser.parse_args())

Ni = USER['Ni']
dt = USER['dt']
mixed = MIXED[USER['mixed']]
log = USER['log']

io = IOConfiguration(None)
io.path = "64x32_dr0.05_dphi0.03125_curved"
mesh = io.ngsmesh.load_routine()
mesh_implicit, mesh_explicit = get_imex_mesh_from_single_mesh(mesh, Ni=Ni, Nr=64)

initial_cfg = CompressibleFlowSolver(mesh)
initial_cfg.update(**TRANSIENT_CFG)
initial_cfg.io.path = io.path.joinpath("initial_solution")
initial_cfg.io.gfu.filename = f"initial_{300:.6e}"
initial_cfg.io.log.enable = False

GENERAL = TRANSIENT_CFG.copy()
GENERAL['time.timer.interval'] = (0.0, 0.1)
GENERAL['time.timer.step'] = dt
if log:
    GENERAL['io.vtk.enable'] = True
    GENERAL['io.vtk.rate'] = 1
    GENERAL['io.vtk.subdivision'] = 2
    GENERAL['io.gfu.enable'] = True
    GENERAL['io.gfu.rate'] = 1
    GENERAL['io.gfu.time_level_rate'] = 1

GENERAL['io.path'] = io.path.joinpath("runtimes")
GENERAL['io.log.enable'] = log
GENERAL['fem.scheme.compile'] = {'realcompile': True, 'wait': True, 'keep_files': False},


HDG = GENERAL.copy()
HDG['fem'] = 'conservative_hdg'
HDG['fem.scheme'] = 'sdirk33'
HDG['fem.viscous_treatment'] = mixed
if USER['mixed'] == 'ip':
    HDG['fem.viscous_treatment.interior_penalty_coefficient'] = 10.0
if log:
    HDG['io.sensor.enable'] = True

DG = GENERAL.copy()
DG['fem'] = 'conservative_dg'
DG['fem.scheme'] = 'rk_ars33'
DG['fem.viscous_treatment'] = 'interior_penalty'
DG['fem.viscous_treatment.interior_penalty_coefficient'] = 1.0

mesh.Curve(initial_cfg.fem.order)
mesh_implicit.Curve(initial_cfg.fem.order)
mesh_explicit.Curve(initial_cfg.fem.order)

if USER['splitting'] == 'explicit':

    cfg_full_explicit = CompressibleFlowSolver(mesh)
    FULL_EXPLICIT = Cylinder(DG, filename="full_explicit")
    FULL_EXPLICIT.set_conditions(cfg_full_explicit)
    single_transient_routine(FULL_EXPLICIT, initial_cfg)

elif USER['splitting'] == 'implicit':

    cfg_full_implicit = CompressibleFlowSolver(mesh)
    FULL_IMPLICIT = Cylinder(HDG, filename=f"full_implicit_{USER['mixed']}")
    FULL_IMPLICIT.set_conditions(cfg_full_implicit)
    single_transient_routine(FULL_IMPLICIT, initial_cfg)

else:

    implicit_cfg = CompressibleFlowSolver(mesh_implicit)
    IMEX_IMPLICIT = Cylinder(HDG, filename=f"imex_implicit_{Ni}_{USER['mixed']}", domain="implicit", boundaries='wall')
    IMEX_IMPLICIT.set_conditions(implicit_cfg)

    explicit_cfg = CompressibleFlowSolver(mesh_explicit)
    IMEX_EXPLICIT = Cylinder(DG, filename=f"imex_explicit_{Ni}_{USER['mixed']}", domain="explicit", boundaries='farfield')
    IMEX_EXPLICIT.set_conditions(explicit_cfg)

    imex_transient_routine(IMEX_IMPLICIT, IMEX_EXPLICIT, initial_cfg)
