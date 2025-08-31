# Import preliminary modules.
from dream import *
from dream.compressible import Initial, CompressibleFlowSolver, flowfields, FarField, Outflow, InterfaceBC
from dream.time import MultizoneIMEXTimeRoutine
import ngsolve as ngs 
import numpy as np
from matplotlib import pyplot as plt
from netgen.occ import OCCGeometry, WorkPlane
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


# Function that generates a grid.
def create_simple_grid(ne, lx, ly):

    # Select a common element size.
    h0 = min( lx, ly )/ne

    # Generate a simple rectangular geometry.
    domain = WorkPlane().RectangleC(lx, ly).Face()

    # Assign the name of the internal solution in the domain.
    domain.name = 'internal'

    # For convenience, extract and name each of the edges consistently.
    bottom = domain.edges[0]; bottom.name = 'bottom'
    right  = domain.edges[1]; right.name  = 'right'
    top    = domain.edges[2]; top.name    = 'top'
    left   = domain.edges[3]; left.name   = 'left'

    # Pair the boundaries in each direction: vertical and horizontal.
    bottom.Identify(top, "ydir", IdentificationType.PERIODIC)
    #right.Identify(left, "xdir", IdentificationType.PERIODIC)

    # Initialize a rectangular 2D geometry.
    geo = OCCGeometry(domain, dim=2)

    # Discretize the domain.
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=h0, quad_dominated=True))

    # Return our fancy grid.
    return mesh


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

        if cfg.fem.order == 0:
            order = 1
        else:
            order = cfg.fem.order
        
        cfg.io.vtk.fields = fields 
        cfg.io.vtk.enable=True
        cfg.io.vtk.rate = 50
        cfg.io.vtk.subdivision = order
        cfg.io.vtk.path = "inviscid"
        cfg.io.vtk.filename = filename




# # # 
# Generate the grid.
# # 

# Number of elements per dimension.
ne = 8

# Dimension of the rectangular domain.
lx = 20 
ly = 10 

# Generate a simple grid.
mesh = create_simple_grid(ne, lx, ly)


# Time information.
time_info = TimeInfo( t0=0.0, tf=8.0, dt=0.01 )

# Polynomial order.
nPoly = 3

# Number of stages in the IMEX scheme.
nStage = 1

# Create the solver.
cfg = CompressibleFlowSolver(mesh)


# Assign the relevant information and values.
cfg.dynamic_viscosity = "inviscid"
cfg.equation_of_state = "ideal"
cfg.equation_of_state.heat_capacity_ratio = GlobalParameters.gamma
cfg.scaling = "aerodynamic"
cfg.mach_number = GlobalParameters.Ma

cfg.riemann_solver = "lax_friedrich"
cfg.fem = "conservative_dg"
cfg.fem.order = nPoly

cfg.time = "transient"
cfg.fem.scheme = get_imex_scheme_explicit(nStage)
cfg.time.timer.interval = (time_info.t0, time_info.tf)
cfg.time.timer.step = time_info.dt

cfg.fem.solver.method = "newton"
cfg.fem.solver.method.damping_factor = 1
cfg.fem.solver.method.max_iterations = 10
cfg.fem.solver.method.convergence_criterion = 1e-10

Uic = get_initial_condition(cfg)
Uinf = get_farfield_solution(cfg)

# Boundary conditions (non-interface).
#cfg.bcs['left|right'] = "periodic"
cfg.bcs['right'] = FarField(fields=Uinf)
cfg.bcs['left'] = FarField(fields=Uinf)
cfg.bcs['top|bottom'] = "periodic"
cfg.dcs['internal'] = Initial(fields=Uic)

# Allocate the necessary data.
cfg.initialize()

# Set up VTK stream.
init_vtk_stream(cfg, True, "sdg_ees")


# # #
# Run simulation.
# # 

with ngs.TaskManager(): 
    cfg.solve()
        
        












