"""
Transient viscous IMEX MMS benchmark on a locally refined mesh using the manufactured solution of [1].

Comments:
RKARS11 DG requires a dt of at least 0.0001 to remain stable for order 3.
RKARS22 DG requires a dt of at least 0.0005 to remain stable for order 3.
RKARS33 DG requires a dt of at least 0.001 to remain stable for order 3.

References:
[1] - Gassner, G., Lörcher, F. & Munz, CD. A Discontinuous Galerkin Scheme based on a Space-Time Expansion II. 
Viscous Flow Equations in Multi Dimensions. J Sci Comput 34, 260–286 (2008). https://doi.org/10.1007/s10915-007-9169-1
"""
import argparse
import ngsolve as ngs
from config import (TRANSIENT_CFG,
                        STAGE_TO_SCHEME,
                        NavierStokesMMS,
                        get_refined_meshes,
                        get_gassner_mms,
                        transient_convergence_routine,
                        transient_imex_convergence_routine)

ngs.SetNumThreads(4)

IMEX = ('implicit', 'synchronised', 'explicit')

VISCOUS_TREATMENTS = {
    'mixed': 'mixed_strain_temperature_gradient',
    'ip': 'interior_penalty'
}

parser = argparse.ArgumentParser(description='Transient viscous IMEX MMS benchmark')
parser.add_argument('splitting', type=str, help='Splitting method', choices=list(IMEX))
parser.add_argument('stage', type=int, help='Number of temporal stages', choices=[1, 2, 3])
parser.add_argument('viscous_treatment', type=str, help='Viscous treatment', choices=list(VISCOUS_TREATMENTS))
parser.add_argument('--time_steps', type=float, nargs="+", help='Time steps', default=(0.01,))
parser.add_argument('--vtkrate', type=int, help='vtkrate', default=1)
parser.add_argument('--gfurate', type=int, help='gfurate', default=1)
parser.add_argument('--hdg_ip', type=float, help='Interior penalty coefficient', default=10.0)
parser.add_argument('--dg_ip', type=float, help='Interior penalty coefficient', default=1.0)

USER = vars(parser.parse_args())

imp_scheme, exp_scheme = STAGE_TO_SCHEME[USER['stage']]

output = f"transient_mms_refined/navier-stokes/{USER['splitting']}/{USER['stage']}stage_scheme"

GENERAL = TRANSIENT_CFG.copy()
GENERAL['equation_of_state.heat_capacity_ratio'] = 1.4
GENERAL['riemann_solver'] = 'upwind'
GENERAL['mach_number'] = 1.0  # Mach number is set by the MMS
GENERAL['reynolds_number'] = 565.685424949238  # Reynolds number is set by the MMS
GENERAL['time.timer.interval'] = (0.0, 1.0)
GENERAL['fem'] = ...  # To be set
GENERAL['fem.order'] = 3
GENERAL['fem.solver'] = 'direct'
GENERAL['fem.solver.method'] = 'newton'
GENERAL['fem.solver.method.max_iterations'] = 5
GENERAL['fem.solver.method.convergence_criterion'] = 1e-20
GENERAL['fem.solver.method.damping_factor'] = 1.0
GENERAL['fem.bonus_int_order'] = 6
GENERAL['fem.scheme'] = ... # To be set
GENERAL['fem.scheme.compile'] = True
GENERAL['io.sensor.enable'] = True
GENERAL['io.vtk.enable'] = True
GENERAL['io.vtk.rate'] = USER['vtkrate']
GENERAL['io.vtk.subdivision'] = 2
GENERAL['io.gfu.enable'] = True
GENERAL['io.gfu.rate'] = USER['gfurate']
GENERAL['io.gfu.time_level_rate'] = 100
GENERAL['io.vtk.subdivision'] = 2
GENERAL['io.path'] = output

HDG = GENERAL.copy()
HDG['fem'] = 'conservative_hdg'
HDG['fem.scheme'] = imp_scheme
HDG['fem.viscous_treatment'] = VISCOUS_TREATMENTS[USER['viscous_treatment']]
if USER['viscous_treatment'] == 'ip':
    HDG['fem.viscous_treatment.interior_penalty_coefficient'] = USER['hdg_ip']

DG = GENERAL.copy()
DG['fem'] = 'conservative_dg'
DG['fem.scheme'] = exp_scheme
DG['fem.viscous_treatment'] = 'interior_penalty'
DG['fem.viscous_treatment.interior_penalty_coefficient'] = USER['dg_ip']


if __name__ == "__main__":

    mesh, imp_mesh, exp_mesh = get_refined_meshes(32, 32, refinements=5)

    if USER['splitting'] == 'implicit':
        simulation = NavierStokesMMS(HDG, get_gassner_mms, 'hdg', periods=1)
        transient_convergence_routine(mesh, simulation, time_steps=USER['time_steps'])
    elif USER['splitting'] == 'explicit':
        simulation = NavierStokesMMS(DG, get_gassner_mms, 'dg', periods=1)
        transient_convergence_routine(mesh, simulation, time_steps=USER['time_steps'])
    else:
        hdg = NavierStokesMMS(HDG, get_gassner_mms, 'hdg', domain="implicit", boundaries="left|bottom|right", periods=1)
        dg = NavierStokesMMS(DG, get_gassner_mms, 'dg', domain="explicit", boundaries="left|top|right", periods=1)
        transient_imex_convergence_routine(imp_mesh, exp_mesh, hdg, dg, time_steps=USER['time_steps'])
