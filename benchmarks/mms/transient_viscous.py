"""
Transient viscous MMS benchmark using the manufactured solution of [1].

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
                        SCHEME_ORDER,
                        NavierStokesMMS,
                        get_uniform_meshes,
                        get_gassner_mms,
                        transient_convergence_routine)

ngs.SetNumThreads(4)

FEM = {
    'dg': 'conservative_dg',
    'hdg': 'conservative_hdg'
}

VISCOUS_TREATMENTS = {
    'mixed': 'mixed_strain_temperature_gradient',
    'ip': 'interior_penalty'
}


parser = argparse.ArgumentParser(description='Transient viscous MMS benchmark')
parser.add_argument('fem', type=str, help='Finite element method', choices=list(FEM))
parser.add_argument('scheme', type=str, help='Time integration scheme', choices=list(SCHEME_ORDER))
parser.add_argument('viscous_treatment', type=str, help='Viscous treatment', choices=list(VISCOUS_TREATMENTS))
parser.add_argument('--ip', type=float, help='Interior penalty coefficient', default=1.0)
parser.add_argument('--time_steps', type=float, nargs="+", help='Time steps', default=(0.01,))
parser.add_argument('--vtkrate', type=int, help='vtkrate', default=1)
parser.add_argument('--gfurate', type=int, help='gfurate', default=1)

USER = vars(parser.parse_args())

fem = f"{USER['fem']}_{USER['viscous_treatment']}"
if USER['viscous_treatment'] == 'ip':
    fem += f"_{int(USER['ip'])}"
output = f"transient_mms/navier-stokes/{fem}/{USER['scheme']}"

GENERAL = TRANSIENT_CFG.copy()
GENERAL['equation_of_state.heat_capacity_ratio'] = 1.4
GENERAL['riemann_solver'] = 'upwind'
GENERAL['mach_number'] = 1.0  # Mach number is set by the MMS 
GENERAL['reynolds_number'] = 565.685424949238 # Reynolds number is set by the MMS
GENERAL['time.timer.interval'] = (0.0, 1.0)
GENERAL['fem'] = FEM[USER['fem']]
GENERAL['fem.viscous_treatment'] = VISCOUS_TREATMENTS[USER['viscous_treatment']]
GENERAL['fem.viscous_treatment.interior_penalty_coefficient'] = USER['ip']
GENERAL['fem.order'] = 3
GENERAL['fem.solver'] = 'direct'
GENERAL['fem.solver.method'] = 'newton'
GENERAL['fem.solver.method.max_iterations'] = 5
GENERAL['fem.solver.method.convergence_criterion'] = 1e-20
GENERAL['fem.solver.method.damping_factor'] = 1.0
GENERAL['fem.bonus_int_order'] = 6
GENERAL['fem.scheme'] = USER['scheme']
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


simulation = NavierStokesMMS(GENERAL, get_gassner_mms, USER['fem'], periods=1)

if __name__ == "__main__":
    mesh, _, _ = get_uniform_meshes(32, 32)
    transient_convergence_routine(mesh, simulation, time_steps=USER['time_steps'])
