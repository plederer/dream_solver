# Import preliminary modules.
from dream import *
from dream.compressible import Initial, CompressibleFlowSolver, flowfields, FarField, Outflow, InterfaceBC
from dream.time import MultizoneIMEXTimeRoutine
import ngsolve as ngs 
import numpy as np
from matplotlib import pyplot as plt
import netgen.occ as occ
from netgen.meshing import IdentificationType

ngs.ngsglobals.msg_level = 0 
ngs.SetNumThreads(6)



# Class containing all the parameters to define an isentropic vortex. 
class IsentropicVortexParam:
    
    # Center of the vortex. 
    x0 = -2.0
    y0 =  0.0
    
    # Vortex parameters, found in Spiegel et al. (2015). 
    theta = 0.0  # flow angle [deg]. 
    Tinf  = 1.0  # background temperature.
    Pinf  = 1.0  # background pressure.
    Rinf  = 1.0  # background density.
    Minf  = 0.5  # background Mach.
    sigma = 1.0  # Perturbation strength.
    Rv    = 1.0  # Perturbation width.
    
    # Scaling for the maximum strength of the perturbation.
    beta = Minf*ngs.exp(0.5)*5.0*ngs.sqrt(2.0)/(4.0*ngs.pi)
    
    # Convert the angle from degrees to radians.
    theta *= ngs.pi/180.0
    
    # Deduce the background mean velocity.
    uinf = Minf*ngs.cos( theta )
    vinf = Minf*ngs.sin( theta )

    # Specific heat ratio.
    gamma = 1.4


# Class containing the global parameters.
class GlobalParameters:

    Re = 10
    Pr = 0.72
    Ma = IsentropicVortexParam.Minf
    gamma = IsentropicVortexParam.gamma

# Class containing time parameters.
class TimeInfo:
    def __init__(self, t0, tf, dt):
        self.t0 = t0
        self.tf = tf
        self.dt = dt

# Function that deduces the implicit scheme in an IMEX, based on input number of stages.
def get_imex_scheme_implicit(nStage):

    if nStage == 1:
        return "implicit_euler"
    elif nStage == 2:
        return "sdirk22"
    elif nStage == 3:
        return "sdirk33"
    elif nStage == 4:
        return "sdirk43"
    else:
        raise ValueError(f"Number of stages is not implemented (yet).")

# Function that deduces the explicit scheme in an IMEX, based on input number of stages.
def get_imex_scheme_explicit(nStage):

    if nStage == 1:
        return "explicit_euler"
    elif nStage == 2:
        return "rk_ars22"
    elif nStage == 3:
        return "rk_ars33"
    elif nStage == 4:
        return "rk_ars43"
    else:
        raise ValueError(f"Number of stages is not implemented (yet).")

# Function that creates a grid from two zones, based on the input info.
def create_simple_grid(ne, lx, ly):
    
    # Grid dimensions.
    # 1) We assume ly is identical in both grids.
    # 2) We assume x=0 is the interface location.
    lx0 = lx/2
    lx1 = lx-lx0
    dx1 = lx1/2 

    # Overall, initial domain: will end up implicit.
    domain0 = occ.WorkPlane().RectangleC(lx, ly).Face()
    domain0.name = "imp_internal"
    domain0.edges[0].name = 'imp_bottom'
    domain0.edges[1].name = 'interface'
    domain0.edges[2].name = 'imp_top'
    domain0.edges[3].name = 'imp_left'
    
    # Subdomain: will end up explicit.
    domain1 = occ.WorkPlane().MoveTo(dx1, 0).RectangleC(lx1, ly).Face()
    domain1.name = "exp_internal"
    domain1.edges[0].name = 'exp_bottom'
    domain1.edges[1].name = 'exp_right'
    domain1.edges[2].name = 'exp_top'
    domain1.edges[3].name = 'interface'
    
    # Subtract the domains, to get the two regions.
    domain0 -= domain1

    # Make the vertical boundaries periodic.
    domain0.edges[1].Identify( domain0.edges[3], "imp_periodic", IdentificationType.PERIODIC)
    domain1.edges[0].Identify( domain1.edges[2], "exp_periodic", IdentificationType.PERIODIC)
    
    # Deduce the grid spacing.
    maxh = min( lx, ly )/ne

    # Generate the meshes.
    mesh_imp = ngs.Mesh(occ.OCCGeometry(domain0, dim=2).GenerateMesh(maxh=maxh, quad_dominated=True))
    mesh_exp = ngs.Mesh(occ.OCCGeometry(domain1, dim=2).GenerateMesh(maxh=maxh, quad_dominated=True))
    
    return mesh_imp, mesh_exp


# Function that defines the analytic solution.
def get_initial_condition(cfg):
  
    # Extract the starting time.
    t0 = cfg.time.timer.interval[0]

    # Return the analytic solution at time: t0.
    return get_analytic_solution(cfg, t0)


# Function that creates a time-dependant analytic solution.
def get_analytic_solution(cfg, t):

    # Generate an array of 4x3 vortices. If you need something different, modify it.
    # NOTE, in this case, the 4x3 array of vortices assumes:
    #  1) flow aligned in (+ve) x-direction.
    #  2) simulation is done for only 1 period: tf = lx/uInf.
    #fn = ngs.CF(())
    #for i in range(-2,2):
    #    for j in range(-1,2):
    #        fn += get_perturbation(t, i, j)
    
    # Just use one for now, since we are prototyping.
    fn = get_perturbation(t, 0, 0)

    # For convenience, extract the perturbations explicitly.
    dT = fn[0]; du = fn[1]; dv = fn[2]

    # Extract the required parameters to construct the actual variables.
    gamma = IsentropicVortexParam.gamma
    uinf  = IsentropicVortexParam.uinf
    vinf  = IsentropicVortexParam.vinf

    # Abbreviations involving gamma.
    gm1    = gamma - 1.0
    ovg    = 1.0/gamma
    ovgm1  = 1.0/gm1
    govgm1 = gamma*ovgm1

    # Define the primitive variables, by superimposing the perturbations on a background state.
    r = (1.0 + dT)**ovgm1
    u = uinf + du
    v = vinf + dv
    p = ovg*(1.0 + dT)**govgm1

    # Return the analytic expression of the vortex.
    return flowfields( rho=r, u=(u, v), p=p )


# Function that creates a farfield solution from the vortex parameters.
def get_farfield_solution(cfg):

    # For convenience, extract the information of the vortex.
    gamma = IsentropicVortexParam.gamma
    uinf  = IsentropicVortexParam.uinf
    vinf  = IsentropicVortexParam.vinf

    # Abbreviations involving gamma.
    gm1    = gamma - 1.0
    ovg    = 1.0/gamma
    ovgm1  = 1.0/gm1
    govgm1 = gamma*ovgm1

    # Construct the farfield state, which is the unperturbed solution.
    r = 1.0
    u = uinf
    v = vinf
    p = ovg

    return flowfields( rho=r, u=(u,v), p=p )
 


# Function that generates a single isentropic perturbations in the velocity and temperature.
# Here, (ni,nj) are the integers for the multiple of (lx,ly)
# distances between the leading/trailing vortices.
def get_perturbation(t, ni, nj):

    # For convenience, extract the information of the vortex.
    theta = IsentropicVortexParam.theta
    Tinf  = IsentropicVortexParam.Tinf
    Pinf  = IsentropicVortexParam.Pinf
    Rinf  = IsentropicVortexParam.Rinf
    Minf  = IsentropicVortexParam.Minf
    sigma = IsentropicVortexParam.sigma
    Rv    = IsentropicVortexParam.Rv
    beta  = IsentropicVortexParam.beta
    gamma = IsentropicVortexParam.gamma
    x0    = IsentropicVortexParam.x0
    y0    = IsentropicVortexParam.y0
    uinf  = IsentropicVortexParam.uinf
    vinf  = IsentropicVortexParam.vinf

    # Center of the pulse.
    xc = x0 + ni*lx
    yc = y0 + nj*ly

    # Time-dependent pulse center.
    xt = (ngs.x-xc) - uinf*t
    yt = (ngs.y-yc) - vinf*t

    # Abbreviations involving gamma.
    gm1    =  gamma - 1.0
    ovg    =  1.0/gamma
    ovgm1  =  1.0/gm1
    govgm1 =  gamma*ovgm1
    ovs2   =  1.0/(sigma*sigma)

    # The Gaussian perturbation function.
    ovRv   =  1.0/Rv
    f      = -0.5*ovs2*( (xt/Rv)**2 + (yt/Rv)**2 )
    Omega  =  beta*ngs.exp(f)

    # Velocity and temperature perturbations.
    du = -ovRv*yt*Omega
    dv =  ovRv*xt*Omega
    dT = -0.5*gm1*Omega**2

    # Return the Perturbations.
    return ngs.CF( (dT, du, dv) )


# Function that visualizes the data using VTK format.
def init_vtk_stream(cfg, plot_sol_vtk=False, filename="output"):
    if plot_sol_vtk:
        # Extract the specific heat ratio.
        gamma = IsentropicVortexParam.gamma 
    
        # Get a reference to the numerical solution and the analytic solution.
        uh = cfg.get_solution_fields('rho_u')
        ue = get_analytic_solution(cfg, cfg.time.timer.t)
    
        isentropic_deviation = (uh.p/uh.rho**gamma) / (ue.p/ue.rho**gamma) - 1.0
        fields = cfg.get_solution_fields()
        fields["Mach"] = cfg.get_local_mach_number( fields )
        fields["Entropy"] = ngs.log( uh.p/uh.rho**gamma )
        
        fields["Exact[Density]"] = ue.rho
        fields["Exact[Pressure]"] = ue.p
        fields["Exact[Velocity]"] = ue.u
        fields["Exact[Entropy]"] = ngs.log( ue.p/ue.rho**gamma )
    
        fields["Diff[Density]"] = ue.rho - uh.rho
        fields["Diff[Pressure]"] = ue.p - uh.p
        fields["Diff[Velocity]"] = ue.u - uh.u
        fields["Deviation[Isentropy]"] = isentropic_deviation 

        #dU = cfg.get_all_solution_fields() 
        #fields["StressTensor"] = cfg.deviatoric_stress_tensor(fields, dU)
        if cfg.fem.order == 0:
            order = 1
        else:
            order = cfg.fem.order

        cfg.io.vtk.fields = fields 
        cfg.io.vtk.enable=True
        cfg.io.vtk.rate = 50
        cfg.io.vtk.subdivision = cfg.fem.order
        cfg.io.vtk.path = "viscous_structured"
        cfg.io.vtk.filename = filename



# Function that sets the HDG configuration.
def init_hdg_cfg(cfg, time_info, multizone_time, nPoly, nStage, plot_sol_vtk=False):

    cfg.dynamic_viscosity = "constant"
    cfg.equation_of_state = "ideal"
    cfg.equation_of_state.heat_capacity_ratio = GlobalParameters.gamma
    cfg.scaling = "aerodynamic"
    cfg.mach_number = GlobalParameters.Ma
    cfg.reynolds_number = GlobalParameters.Re
    cfg.prandtl_number = GlobalParameters.Pr

    cfg.riemann_solver = "lax_friedrich"
    cfg.fem = "conservative_hdg"
    cfg.fem.order = nPoly
    cfg.fem.viscous_treatment = "mixed_strain_temperature_gradient"
    
    cfg.time = multizone_time
    cfg.fem.scheme = get_imex_scheme_implicit(nStage)
    cfg.time.timer.interval = (time_info.t0, time_info.tf)
    cfg.time.timer.step = time_info.dt
    
    cfg.fem.solver.method = "newton"
    cfg.fem.solver.method.damping_factor = 1
    cfg.fem.solver.method.max_iterations = 10
    cfg.fem.solver.method.convergence_criterion = 1e-10
    
    Uic = get_initial_condition(cfg)
    Uinf = get_farfield_solution(cfg)
    
    # Boundary conditions (non-interface).
    cfg.bcs['imp_top|imp_bottom'] = "periodic"
    cfg.bcs['imp_left'] = FarField(fields=Uinf)
    cfg.dcs['imp_internal'] = Initial(fields=Uic)

    cfg.fem.initialize_finite_element_spaces()
    cfg.fem.initialize_trial_and_test_functions()
    cfg.fem.initialize_gridfunctions()
    cfg.fem.initialize_time_scheme_gridfunctions()

    # Set up VTK stream.
    init_vtk_stream(cfg, plot_sol_vtk, "hdg_nse")



# Function that sets the SDG configuration.
def init_sdg_cfg(cfg, time_info, multizone_time, nPoly, nStage, plot_sol_vtk=False):
    
    cfg.dynamic_viscosity = "constant"
    cfg.equation_of_state = "ideal"
    cfg.equation_of_state.heat_capacity_ratio = GlobalParameters.gamma
    cfg.scaling = "aerodynamic"
    cfg.mach_number = GlobalParameters.Ma
    cfg.reynolds_number = GlobalParameters.Re
    cfg.prandtl_number = GlobalParameters.Pr

    cfg.riemann_solver = "lax_friedrich"
    cfg.fem = "conservative_dg"
    cfg.fem.order = nPoly
    cfg.fem.viscous_treatment = "interior_penalty_method_sdg"
    cfg.fem.viscous_treatment.interior_penalty_coefficient = 1.0

    cfg.time = multizone_time
    cfg.fem.scheme = get_imex_scheme_explicit(nStage)
    cfg.time.timer.interval = (time_info.t0, time_info.tf)
    cfg.time.timer.step = time_info.dt
    
    #cfg.fem.bonus_int_order['convection']['vol'] = 5
    #cfg.fem.bonus_int_order['convection']['bnd'] = 5
    #cfg.fem.bonus_int_order['diffusion']['vol'] = 5
    #cfg.fem.bonus_int_order['diffusion']['bnd'] = 5

    Uic = get_initial_condition(cfg)
    Uinf = get_farfield_solution(cfg)
    
    # Boundary conditions (non-interface).
    cfg.bcs['exp_top|exp_bottom'] = "periodic"
    cfg.bcs['exp_right'] = Outflow(pressure=Uinf)
    cfg.dcs['exp_internal'] = Initial(fields=Uic)

    cfg.fem.initialize_finite_element_spaces()
    cfg.fem.initialize_trial_and_test_functions()
    cfg.fem.initialize_gridfunctions()
    cfg.fem.initialize_time_scheme_gridfunctions()

    # Set up VTK stream.
    init_vtk_stream(cfg, plot_sol_vtk, "sdg_nse")


# Function that imposes the interface conditions
def set_interface_conditions(cfg_imp, cfg_exp):

    # Get the gridfuctions for both regions.
    gfu_imp = cfg_imp.get_all_solution_fields()
    gfu_exp = cfg_exp.get_all_solution_fields()

    # Impose the interface conditions.
    cfg_imp.bcs['interface'] = InterfaceBC(fields=gfu_exp)
    cfg_exp.bcs['interface'] = InterfaceBC(fields=gfu_imp)

    # Finalize the implicit set up.
    cfg_imp.fem.set_boundary_conditions()
    cfg_imp.fem.set_initial_conditions()
    cfg_imp.fem.initialize_symbolic_forms()

    # Finalize the explicit set up.
    cfg_exp.fem.set_boundary_conditions()
    cfg_exp.fem.set_initial_conditions()
    cfg_exp.fem.initialize_symbolic_forms()



# # # 
# Generate the grid.
# # 

# Number of elements per dimension.
ne = 10

# Dimension of the rectangular domain.
lx = 20 
ly = 14 

# Generate a simple grid.
mesh_imp, mesh_exp = create_simple_grid(ne, lx, ly)

# Number of stages in the IMEX scheme.
nStage = 1

# Time information.
time_info = TimeInfo( t0=0.0, tf=8.0, dt=0.01 )

# Polynomial order.
nPoly = 3


# # 
# Solver configuration: Compressible flow.
# # 

cfg_hdg = CompressibleFlowSolver(mesh_imp)
cfg_sdg = CompressibleFlowSolver(mesh_exp)


# Create a multizone time strategy IMEX class.
multizone_time = MultizoneIMEXTimeRoutine(cfg_implicit=cfg_hdg, cfg_explicit=cfg_sdg)

# Initialize the non-interface configurations.
init_hdg_cfg(cfg_hdg, time_info, multizone_time, nPoly, nStage, plot_sol_vtk=True)
init_sdg_cfg(cfg_sdg, time_info, multizone_time, nPoly, nStage, plot_sol_vtk=True)

# Initialize the interface and remaining configuration.
set_interface_conditions(cfg_hdg, cfg_sdg)



# # #
# Run simulation.
# # 

with ngs.TaskManager(): 
    multizone_time.solve()
        
        













