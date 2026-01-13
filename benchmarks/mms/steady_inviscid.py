"""
Steady inviscid MMS benchmark using the manufactured solution of [1].

References:
[1] - Gassner, G., Lörcher, F. & Munz, CD. A Discontinuous Galerkin Scheme based on a Space-Time Expansion II. 
Viscous Flow Equations in Multi Dimensions. J Sci Comput 34, 260–286 (2008). https://doi.org/10.1007/s10915-007-9169-1
"""
import argparse
import ngsolve as ngs
from .config import (PSEUDO_STATIONARY_CFG,
                        EulerMMS,
                        get_uniform_meshes,
                        get_gassner_mms,
                        steady_convergence_routine)

ngs.SetNumThreads(4)

FEM = {
    'dg': 'conservative_dg',
    'hdg': 'conservative_hdg'
}

parser = argparse.ArgumentParser(description='Steady inviscid MMS benchmark')
parser.add_argument('fem', type=str, help='Finite element method', choices=list(FEM))
parser.add_argument('order', type=int, help='Polynomial order', choices=[1, 2, 3, 4, 5])
parser.add_argument('--levels', type=int, nargs=2, help='Refinement levels', default=(0, 5))

USER = vars(parser.parse_args())
output = f"steady_mms/euler/{USER['fem']}/order{USER['order']}"

GENERAL = PSEUDO_STATIONARY_CFG.copy()
GENERAL['equation_of_state.heat_capacity_ratio'] = 1.4
GENERAL['riemann_solver'] = 'upwind'
GENERAL['mach_number'] = 1.0  # Mach number is set by the MMS
GENERAL['time.timer.step'] = 0.001
GENERAL['time.max_time_step'] = 10
GENERAL['time.increment_at'] = 10
GENERAL['time.increment_factor'] = 10
GENERAL['fem'] = FEM[USER['fem']]
GENERAL['fem.scheme'] = 'implicit_euler'
GENERAL['fem.order'] = USER['order']
GENERAL['fem.solver'] = 'direct'
GENERAL['fem.solver.method'] = 'newton'
GENERAL['fem.solver.method.max_iterations'] = 50
GENERAL['fem.solver.method.convergence_criterion'] = 1e-20
GENERAL['fem.solver.method.damping_factor'] = 1.0
GENERAL['fem.bonus_int_order'] = 2*USER['order']
GENERAL['fem.scheme.compile'] = False
GENERAL['io.sensor.enable'] = True
GENERAL['io.vtk.enable'] = True
GENERAL['io.vtk.rate'] = 1
GENERAL['io.vtk.subdivision'] = 2
GENERAL['io.gfu.enable'] = True
GENERAL['io.gfu.rate'] = 1
GENERAL['io.gfu.time_level_rate'] = 100
GENERAL['io.vtk.subdivision'] = 2
GENERAL['io.path'] = output

simulation = EulerMMS(GENERAL, get_gassner_mms, USER['fem'])

if __name__ == "__main__":
    mesh, _, _ = get_uniform_meshes(2, 2)
    steady_convergence_routine(mesh, simulation, levels=USER['levels'])
