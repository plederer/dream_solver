from dream import *
from dream.compressible import flowstate, Initial
from ngsolve import *
from netgen.occ import OCCGeometry, WorkPlane
from netgen.meshing import IdentificationType
from ngsolve.meshes import MakeStructured2DMesh


# Class containing all the parameters to define an isentropic vortex. 
# For reference, see Spiegel et al (2015).
class IsentropicVortexParam:
    def __init__(self, PDE, lx, ly):
       
        # Store the domain dimensions.
        self.lx     = lx
        self.ly     = ly

        # Center of the vortex, assumed at the center of the domain. 
        # Currently, this assumes that the domain is: [0,lx],[0,ly].       
        self.x0     = lx/2.0
        self.y0     = ly/2.0

        # Vortex parameters, found in Spiegel et al. (2015). 
        self.theta  = 45.0  # flow angle [deg]. 
        self.Tinf   = 1.0  # background temperature.
        self.Pinf   = 1.0  # background pressure.
        self.Rinf   = 1.0  # background density.
        self.Minf   = 0.5  # background Mach.
        self.sigma  = 1.0  # Perturbation strength.
        self.Rv     = 1.0  # Perturbation width.
        
        # Scaling for the maximum strength of the perturbation.
        self.beta   = self.Minf*exp(0.5)*5.0*sqrt(2.0)/(4.0*pi)
        
        # Convert the angle from degrees to radians.
        self.theta *= pi/180.0

        # Deduce the background mean velocity.
        self.uinf   = self.Minf*cos( self.theta ) 
        self.vinf   = self.Minf*sin( self.theta )

        # Store gamma here, so we do not pass it around constantly.
        self.gamma  = PDE.equation_of_state.heat_capacity_ratio


# Function that defines the initial state of the solution.
def InitialCondition(PDE, TEMPORAL, lx, ly):
  
    # Extract the starting time.
    t0 = TEMPORAL.timer.interval[0] 

    # Return the analytic solution at time: t0.
    return AnalyticSolution(PDE, t0, lx, ly)


# Function that creates a time-dependant analytic solution.
def AnalyticSolution(PDE, t, lx, ly):

    # Extract the vortex parameters.
    vparam = IsentropicVortexParam(PDE, lx, ly)
   
    # Generate an array of 4x3 vortices. If you need something different, modify it.
    # In this case, the 4x3 array of vortices assumes:
    #  1) flow aligned in (+ve) x-direction.
    #  2) simulation is done for only 1 period: tf = lx/uInf.
    fn = CF(())
    for i in range(-2,2):
        for j in range(-1,2):
            fn += GeneratePerturbation(vparam, t, i, j)

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

    # Return the analytic expression of the vortices.
    return flowstate( rho=r, u=(u, v), p=p )


# Function that generates a single isentropic perturbations in the velocity and temperature.
# Here, (ni,nj) are the integers for the multiple of (lx,ly)
# distances between the leading/trailing vortices.
def GeneratePerturbation(vparam, t, ni, nj):

    # For convenience, extract the information of the vortex.
    theta = vparam.theta
    Tinf  = vparam.Tinf
    Pinf  = vparam.Pinf
    Rinf  = vparam.Rinf
    Minf  = vparam.Minf
    sigma = vparam.sigma
    Rv    = vparam.Rv
    beta  = vparam.beta
    lx    = vparam.lx
    ly    = vparam.ly
    gamma = vparam.gamma
    x0    = vparam.x0
    y0    = vparam.y0
    uinf  = vparam.uinf
    vinf  = vparam.vinf

    # Center of the pulse.
    xc = x0 + ni*lx 
    yc = y0 + nj*ly

    # Time-dependent pulse center.
    xt = (x-xc) - uinf*t 
    yt = (y-yc) - vinf*t

    # Abbreviations involving gamma.
    gm1    =  gamma - 1.0
    ovg    =  1.0/gamma
    ovgm1  =  1.0/gm1
    govgm1 =  gamma*ovgm1
    ovs2   =  1.0/(sigma*sigma)
    
    # The Gaussian perturbation function.
    ovRv   =  1.0/Rv
    f      = -0.5*ovs2*( (xt/Rv)**2 + (yt/Rv)**2 )  
    Omega  =  beta*exp(f)

    # Velocity and temperature perturbations.
    du = -ovRv*yt*Omega
    dv =  ovRv*xt*Omega
    dT = -0.5*gm1*Omega**2

    # Return the Perturbations.
    return CF( (dT, du, dv) )
    

# Function that process the information to be included in the VTK file.
def ProcessVTKData(IO, PDE, Uexact):

    # Abbreviation for the specific heat ratio.
    gamma = PDE.equation_of_state.heat_capacity_ratio

    # Extract the usual suspects in the numerical solution.
    IO.vtk.fields = PDE.get_state(rho=True, \
                                    u=True, \
                                    p=True, \
                                    T=True, \
                                    c=True)

    # Extract the Mach number.
    mach = PDE.get_local_mach_number(IO.vtk.fields)
    
    # Compute the specific entropy, s = ln(p/rho^gamma).
    s = log( PDE.get_state(p=True).p/PDE.get_state(rho=True).rho**gamma )

    # Compute the difference between the analytic and numerical solution.
    dR   = Uexact.rho - PDE.get_state(rho=True).rho
    dV   = Uexact.u   - PDE.get_state(u=True).u
    dP   = Uexact.p   - PDE.get_state(p=True).p

    # Include the pointwise error.
    IO.vtk.fields['diff[density]']   = dR
    IO.vtk.fields['diff[velocity]']  = dV
    IO.vtk.fields['diff[pressure]']  = dP
    
    # Include the Mach number of the numerical solution.
    IO.vtk.fields['Mach']            = mach
    # Include the modified specific entropy.
    IO.vtk.fields['Entropy']         = s

    # Include the analytical primitive variables.
    IO.vtk.fields['Exact[Density]']  = Uexact.rho
    IO.vtk.fields['Exact[Velocity]'] = Uexact.u
    IO.vtk.fields['Exact[Pressure]'] = Uexact.p

    # Note, you can check that you didn't screw the IC
    # by making sure the analytic specific entropy is constant
    # (e.g. uncomment below).
    #sexact = log( Uexact.p/Uexact.rho**gamma )
    #IO.vtk.fields['Exact[Entropy]'] = sexact



