from config import PSEUDO_STATIONARY_CFG, mesh_refinement_routine, NavierStokesMMS, get_roy_mms

import argparse
parser = argparse.ArgumentParser(description='Steady MMS Roy benchmark')
parser.add_argument('Ma', type=float, help='Mach number')
parser.add_argument('Re', type=float, help='Reynolds number')
parser.add_argument('riemann_solver', type=str, help='Riemann solver')
parser.add_argument('fem', type=str, help='Finite element method')
parser.add_argument('--viscous', type=str, help='Viscous treatment', default='interior_penalty')
parser.add_argument('--IP', type=float, help='Interior penalty coefficient', default=10.0)
OPTIONS = vars(parser.parse_args())

Ma = OPTIONS['Ma']
Re = OPTIONS['Re']
riemann_solver = OPTIONS['riemann_solver']
fem = OPTIONS['fem']
viscous = OPTIONS['viscous']
IP = OPTIONS['IP']

filename = f"{fem}_Ma{Ma}_Re{Re}_{riemann_solver}_{viscous}"
if viscous == 'interior_penalty':
    filename += f"_IP{IP}"

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
GENERAL['fem.viscous_treatment'] = viscous
if viscous == 'interior_penalty':
    GENERAL['fem.viscous_treatment.interior_penalty_coefficient'] = IP
GENERAL['io.path'] = f"steady_roy/{filename}"
GENERAL['io.vtk.enable'] = False

if Ma < 0.1:
    GENERAL['scaling'] = 'acoustic'
    GENERAL['time.timer.step'] = 1
    GENERAL['time.max_time_step'] = 32
    GENERAL['time.increment_at'] = 10
    GENERAL['time.increment_factor'] = 2
    GENERAL['fem.solver'] = 'direct'
    GENERAL['fem.solver.method'] = 'newton'
    GENERAL['fem.solver.method.max_iterations'] = 80
    GENERAL['fem.solver.method.convergence_criterion'] = 1e-20
    GENERAL['fem.solver.method.damping_factor'] = 1.0


# HDG Navier-Stokes setting
HDG = GENERAL.copy()
HDG['fem'] = 'conservative_hdg'

# DG Navier-Stokes setting
DG = GENERAL.copy()
DG['fem'] = 'conservative_dg'

if fem == 'hdg':
    FEM = NavierStokesMMS(HDG, get_roy_mms, filename=filename)
elif fem == 'dg':
    FEM = NavierStokesMMS(DG,  get_roy_mms,filename=filename)

if __name__ == "__main__":
    # Start Routine
    mesh_refinement_routine(FEM, levels=3, orders=[1, 2, 3, 4])
