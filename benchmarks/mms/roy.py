# Import modules
from config import PSEUDO_STATIONARY_CFG, mesh_routine, RoyMMS

import argparse
parser = argparse.ArgumentParser(description='Flow around cylinder benchmark')
parser.add_argument('Ma', type=float, help='Mach number')
parser.add_argument('Re', type=float, help='Reynolds number')
parser.add_argument('riemann_solver', type=str, help='Riemann solver')
parser.add_argument('fem', type=str, help='Finite element method')
parser.add_argument('--IP', type=float, help='Interior penalty coefficient', default=10.0)
OPTIONS = vars(parser.parse_args())

Ma = OPTIONS['Ma']
Re = OPTIONS['Re']
riemann_solver = OPTIONS['riemann_solver']
fem = OPTIONS['fem']
IP = OPTIONS['IP']

GENERAL = PSEUDO_STATIONARY_CFG.copy()
GENERAL['mach_number'] = Ma
GENERAL['reynolds_number'] = Re
GENERAL['prandtl_number'] = 1.0
GENERAL['equation_of_state.heat_capacity_ratio'] = 1.4
GENERAL['riemann_solver'] = riemann_solver
GENERAL['scaling'] = 'aerodynamic'
GENERAL['time'] = 'pseudo_time_stepping'
GENERAL['time.timer.step'] = 1
GENERAL['time.max_time_step'] = 1
GENERAL['time.increment_at'] = 1
GENERAL['time.increment_factor'] = 5
GENERAL['fem.solver'] = 'direct'
GENERAL['fem.solver.method'] = 'newton'
GENERAL['fem.solver.method.max_iterations'] = 50
GENERAL['fem.solver.method.convergence_criterion'] = 1e-20
GENERAL['fem.solver.method.damping_factor'] = 1.0
GENERAL['fem.scheme'] = "implicit_euler"
GENERAL['fem.bonus_int_order'] = 10

# HDG Navier-Stokes setting
HDG = GENERAL.copy()
HDG['fem'] = 'conservative_hdg'
HDG['fem.mixed_method'] = 'strain_heat'

# DG Navier-Stokes setting
DG = GENERAL.copy()
DG['fem'] = 'conservative_dg'
DG['fem.viscous_treatment'] = 'interior_penalty_method_sdg'
DG['fem.viscous_treatment.interior_penalty_coefficient'] = IP

if fem == 'hdg':
    filename = f"{fem}_Ma{Ma}_Re{Re}_{riemann_solver}"
    HDG['io.path'] = f'roy/{filename}'
    FEM = RoyMMS(HDG, filename=filename)
elif fem == 'dg':
    filename = f"{fem}_Ma{Ma}_Re{Re}_{riemann_solver}_IP{IP}"
    DG['io.path'] = f'roy/{filename}'
    FEM = RoyMMS(DG, filename=filename)

if __name__ == "__main__":
    # Start Routine
    mesh_routine(FEM, levels=5, orders=[1, 2, 3, 4, 5])
