# Import preliminary modules.
from dream import *
from dream.compressible import Initial, CompressibleFlowSolver, flowfields
import ngsolve as ngs 
import numpy as np
from matplotlib import pyplot as plt
from netgen.occ import OCCGeometry, WorkPlane
from netgen.meshing import IdentificationType

ngs.ngsglobals.msg_level = 0 
ngs.SetNumThreads(4)



# Function that generates a grid.
def create_simple_grid(ne, lx, ly):

    # Select a common element size.
    h0 = min( lx, ly )/float(ne)

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
    right.Identify(left, "xdir", IdentificationType.PERIODIC)

    # Initialize a rectangular 2D geometry.
    geo = OCCGeometry(domain, dim=2)

    # Discretize the domain.
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=h0, quad_dominated=True))

    # Return our fancy grid.
    return mesh


# Class containing all the parameters to define an isentropic vortex. 
class IsentropicVortexParam:
    def __init__(self, cfg):
       
        # Center of the vortex, assumed at the center of the domain. 
        self.x0     = 0.0
        self.y0     = 0.0

        # Vortex parameters, found in Spiegel et al. (2015). 
        self.theta  = 45.0 # flow angle [deg]. 
        self.Tinf   = 1.0  # background temperature.
        self.Pinf   = 1.0  # background pressure.
        self.Rinf   = 1.0  # background density.
        self.Minf   = 0.5  # background Mach.
        self.sigma  = 1.0  # Perturbation strength.
        self.Rv     = 1.0  # Perturbation width.
        
        # Scaling for the maximum strength of the perturbation.
        self.beta   = self.Minf*ngs.exp(0.5)*5.0*ngs.sqrt(2.0)/(4.0*ngs.pi)

        # Convert the angle from degrees to radians.
        self.theta *= ngs.pi/180.0

        # Deduce the background mean velocity.
        self.uinf   = self.Minf*ngs.cos( self.theta )
        self.vinf   = self.Minf*ngs.sin( self.theta )

        # Store gamma here, so we do not pass it around constantly.
        self.gamma  = cfg.equation_of_state.heat_capacity_ratio


# Function that defines the analytic solution.
def get_initial_condition(cfg):
  
    # Extract the starting time.
    t0 = cfg.time.timer.interval[0]

    # Return the analytic solution at time: t0.
    return get_analytic_solution(cfg, t0)


# Function that creates a time-dependant analytic solution.
def get_analytic_solution(cfg, t):

    # Extract the vortex parameters.
    vparam = IsentropicVortexParam(cfg)

    # Generate an array of 4x3 vortices. If you need something different, modify it.
    # NOTE, in this case, the 4x3 array of vortices assumes:
    #  1) flow aligned in (+ve) x-direction.
    #  2) simulation is done for only 1 period: tf = lx/uInf.
    fn = ngs.CF(())
    for i in range(-2,2):
        for j in range(-1,2):
            fn += get_perturbation(vparam, t, i, j)

    # For convenience, extract the perturbations explicitly.
    dT = fn[0]; du = fn[1]; dv = fn[2]

    # Extract the required parameters to construct the actual variables.
    gamma  = vparam.gamma
    uinf   = vparam.uinf
    vinf   = vparam.vinf

    # Abbreviations involving gamma.
    gm1    =  gamma - 1.0
    ovg    =  1.0/gamma
    ovgm1  =  1.0/gm1
    govgm1 =  gamma*ovgm1

    # Define the primitive variables, by superimposing the perturbations on a background state.
    r = (1.0 + dT)**ovgm1
    u = uinf + du
    v = vinf + dv
    p = ovg*(1.0 + dT)**govgm1

    # Return the analytic expression of the vortex.
    return flowfields( rho=r, u=(u, v), p=p )


# Function that generates a single isentropic perturbations in the velocity and temperature.
# Here, (ni,nj) are the integers for the multiple of (lx,ly)
# distances between the leading/trailing vortices.
def get_perturbation(vparam, t, ni, nj):

    # For convenience, extract the information of the vortex.
    theta = vparam.theta
    Tinf  = vparam.Tinf
    Pinf  = vparam.Pinf
    Rinf  = vparam.Rinf
    Minf  = vparam.Minf
    sigma = vparam.sigma
    Rv    = vparam.Rv
    beta  = vparam.beta
    gamma = vparam.gamma
    x0    = vparam.x0
    y0    = vparam.y0
    uinf  = vparam.uinf
    vinf  = vparam.vinf

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





# # # 
# Generate the grid.
# # 

# Number of elements per dimension.
ne = 12

# Dimension of the rectangular domain.
lx = 10.0 
ly = 10.0 

# Generate a simple grid.
mesh = create_simple_grid(ne, lx, ly)



# # 
# Solver configuration: Compressible (inviscid) flow.
# # 

cfg = CompressibleFlowSolver(mesh)

cfg.dynamic_viscosity = "inviscid"
cfg.equation_of_state = "ideal"
cfg.equation_of_state.heat_capacity_ratio = 1.4
cfg.scaling = "acoustic"
cfg.mach_number = 0.0

cfg.riemann_solver = "lax_friedrich"
cfg.fem = "conservative_hdg"
cfg.fem.order = 3

cfg.time = "transient"
cfg.fem.scheme = "dirk34_ldd"
cfg.time.timer.interval = (0.0, 2.0)
cfg.time.timer.step = 0.01

cfg.fem.solver.method = "newton"
cfg.fem.solver.method.damping_factor = 1
cfg.fem.solver.method.max_iterations = 10
cfg.fem.solver.method.convergence_criterion = 1e-10


Uic = get_initial_condition(cfg)

cfg.bcs['left|right'] = "periodic"
cfg.bcs['top|bottom'] = "periodic"
cfg.dcs['internal'] = Initial(fields=Uic)


# Choose whether or not to visualize the solution.
draw_solution = True


# Allocate the necessary data.
cfg.initialize()


# Check what processing is required, if we need to visualize data.
if draw_solution:
    
    # Abbreviations.
    gamma = cfg.equation_of_state.heat_capacity_ratio


    # Get a reference to the numerical solution and the analytic solution.
    uh = cfg.get_solution_fields('rho_u')
    ue = get_analytic_solution(cfg, cfg.time.timer.t)

    # cfg.io.draw({"Density": uh.rho})
    # cfg.io.draw({"Mach": cfg.get_local_mach_number(uh)})
    # cfg.io.draw({"Exact[Density]": ue.rho})
    isentropic_deviation = (uh.p/uh.rho**gamma) / (ue.p/ue.rho**gamma) - 1.0
    fields = cfg.get_solution_fields('p')
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

    cfg.io.vtk.fields = fields 
    cfg.io.vtk.enable=True
    cfg.io.vtk.rate = 10
    cfg.io.vtk.subdivision = cfg.fem.order
    cfg.io.vtk.filename = "test"



# # #
# Run simulation.
# # 

with ngs.TaskManager(): 
    cfg.solve()
        
        













