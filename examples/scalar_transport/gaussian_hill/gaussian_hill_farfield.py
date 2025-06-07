from dream import *
from dream.scalar_transport import Initial, transportfields, ScalarTransportSolver, FarField
import ngsolve as ngs

from netgen.occ import OCCGeometry, WorkPlane
from netgen.meshing import IdentificationType
from ngsolve.meshes import MakeStructured2DMesh
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from fractions import Fraction



def CreateSimpleGrid(ne, lx, ly):

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

    # Initialize a rectangular 2D geometry.
    geo = OCCGeometry(domain, dim=2)

    # Discretize the domain.
    mesh = ngs.Mesh(geo.GenerateMesh(maxh=h0, quad_dominated=True))

    # Return our fancy grid.
    return mesh


# Number of elements per dimension.
nElem1D = 20

# Dimension of the rectangular domain.
xLength = 3.0 
yLength = 3.0 

# Generate a simple grid.
mesh = CreateSimpleGrid(nElem1D, xLength, yLength)

# Message output detail from netgen.
ngs.ngsglobals.msg_level = 0 
ngs.SetNumThreads(4)


def get_analytic_solution(t, k): 
   
    # Initial pulse location.
    x0 = 0.0 
    y0 = 0.5 
   
    # Pulse center trajectory.
    xc =  x0*ngs.cos(t) - y0*ngs.sin(t)
    yc = -x0*ngs.sin(t) + y0*ngs.cos(t)
    
    # Radial distance (time-dependent).
    r2 = (ngs.x-xc)**2 + (ngs.y-yc)**2
    # Variance of this pulse.
    s2 = get_variance_pulse(t, k)

    # Return the analytic solution.
    return ( 1.0/(s2*ngs.pi) ) * ngs.exp( -r2/(4.0*k*t) )

def get_variance_pulse(t, k): 
    return 4.0*k*t
    
# Solver configuration: Scalar transport equation.
cfg = ScalarTransportSolver(mesh)

cfg.convection_velocity = (-ngs.y, ngs.x)
cfg.diffusion_coefficient = 1.0e-03
cfg.is_inviscid = False

cfg.riemann_solver = "lax_friedrich"
cfg.fem = "dg"                             # NOTE, will get overriden.
cfg.fem.order = 2                          # NOTE, will get overriden.
cfg.fem.interior_penalty_coefficient = 1.0 # NOTE, will get overriden.


cfg.time = "transient"
cfg.fem.scheme = "implicit_euler"
cfg.time.timer.interval = (ngs.pi/2, 5*ngs.pi/2)
cfg.time.timer.step = 0.005

cfg.linear_solver = "pardiso"
cfg.optimizations.static_condensation = False  # NOTE, by default, condensation is turned off.
cfg.optimizations.compile.realcompile = False

U0 = transportfields()
U0.phi = get_analytic_solution(cfg.time.timer.interval[0], cfg.diffusion_coefficient)

Ubc = transportfields()
Ubc.phi = 0.0

cfg.bcs['left|top|bottom|right'] = FarField(fields=Ubc)
cfg.dcs['internal'] = Initial(fields=U0)



# Extract time configuration
t0, tf = cfg.time.timer.interval
dt = cfg.time.timer.step.Get()
nt = int(round((tf - t0) / dt))
    

# Define a decorator for the Gaussian hill routine.
def gaussian_hill_routine(label):
    def decorator(func):
        def wrapper(*args, **kwargs):

            # By default, assume this is a DG formulation.
            cfg.optimizations.static_condensation = False
            
            # Insert options here.
            func(*args, **kwargs)
            cfg.fem.order = 3
            cfg.fem.interior_penalty_coefficient = 1.0

            # In case an HDG formulation is specified, use static condensation.
            if cfg.fem.name == "hdg":
               cfg.optimizations.static_condensation = True
                
            cfg.initialize()
            uh = cfg.fem.get_fields("phi").phi
            ue = get_analytic_solution(cfg.time.timer.t, cfg.diffusion_coefficient)

            # Uncomment to visualize one of the below variables.
            #Uh = cfg.fem.get_fields()
            # cfg.io.draw({"phi": Uh.phi})
            # cfg.io.draw({"Exact[phi]": Uex.phi},  min=0.0, max=10.0)
            #cfg.io.draw({"Diff[phi]": (ue - uh)}, min=-0.1, max=0.1)
            fields = cfg.fem.get_fields()
            fields["Exact[phi]"]  = ue
            fields["Diff[phi]"]  = ue - uh 
            cfg.io.vtk.fields = fields 
            cfg.io.vtk.enable=True
            cfg.io.vtk.rate = 10
            cfg.io.vtk.subdivision = cfg.fem.order+1
            cfg.io.vtk.filename = label
 
            # Integration order (for post-processing).
            qorder = 10
    
            # Data for book-keeping information.
            data = np.zeros((nt, 3), dtype=float)
    
            # Time integration loop.
            for i, t in enumerate(cfg.time.start_solution_routine(True)):
                
                # Get the exact variance.
                s2e = get_variance_pulse(cfg.time.timer.t.Get(), cfg.diffusion_coefficient.Get())
                ## TESTING: computing analytic variance discretly.
                #ue = Uex.phi
                #xe = ngs.Integrate(ngs.x * ue, mesh, order=qorder)
                #ye = ngs.Integrate(ngs.y * ue, mesh, order=qorder)
                #r2e = (ngs.x - xe)**2 + (ngs.y - ye)**2
                #s2e = ngs.Integrate(r2e * ue, mesh, order=(qorder+2) )
                ## TESTING: computing analytic variance discretly.
                
                # Compute centroid
                xh = ngs.Integrate(ngs.x * uh, mesh, order=qorder)
                yh = ngs.Integrate(ngs.y * uh, mesh, order=qorder)
            
                # Compute variance.
                r2h = (ngs.x - xh)**2 + (ngs.y - yh)**2
                s2h = ngs.Integrate(r2h * uh, mesh, order=(qorder+2) )
            
                # Store data: time and normalized error.
                var_dif = s2h/s2e - 1.0
                
                # Compute the L2-norm of the error.
                err = np.sqrt( ngs.Integrate( (ue-uh)**2, mesh, order=qorder) )
                # Book-keep the relevant error metrics.
                data[i] = [cfg.time.timer.t.Get(), var_dif, err]

            return data
        wrapper.label = label 
        return wrapper
    return decorator

# Specialized routines.
@gaussian_hill_routine("implicit_euler(hdg)")
def implicit_euler_hdg():
    cfg.fem = "hdg"
    cfg.fem.scheme = "implicit_euler"
@gaussian_hill_routine("implicit_euler(dg)")
def implicit_euler_dg():
    cfg.fem = "dg"
    cfg.fem.scheme = "implicit_euler"

@gaussian_hill_routine("bdf2(hdg)")
def bdf2_hdg():
    cfg.fem = "hdg"
    cfg.fem.scheme = "bdf2"
@gaussian_hill_routine("bdf2(dg)")
def bdf2_dg():
    cfg.fem = "dg"
    cfg.fem.scheme = "bdf2"

@gaussian_hill_routine("sdirk22(hdg)")
def sdirk22_hdg():
    cfg.fem = "hdg"
    cfg.fem.scheme = "sdirk22"
@gaussian_hill_routine("sdirk22(dg)")
def sdirk22_dg():
    cfg.fem = "dg"
    cfg.fem.scheme = "sdirk22"

@gaussian_hill_routine("sdirk33(hdg)")
def sdirk33_hdg():
    cfg.fem = "hdg"
    cfg.fem.scheme = "sdirk33"
@gaussian_hill_routine("sdirk33(dg)")
def sdirk33_dg():
    cfg.fem = "dg"
    cfg.fem.scheme = "sdirk33"

@gaussian_hill_routine("imex_rk_ars443(dg)")
def imex_rk_ars443_dg():
    cfg.fem = "dg"
    cfg.fem.scheme = "imex_rk_ars443"

@gaussian_hill_routine("ssprk3(dg)")
def ssprk3_dg():
    cfg.fem = "dg"
    cfg.fem.scheme = "ssprk3"

@gaussian_hill_routine("crk4(dg)")
def crk4_dg():
    cfg.fem = "dg"
    cfg.fem.scheme = "crk4"

@gaussian_hill_routine("explicit_euler(dg)")
def explicit_euler_dg():
    cfg.fem = "dg"
    cfg.fem.scheme = "explicit_euler"




# Assign colors to base method names
method_colors = {
    "implicit_euler": "tab:red",
    "bdf2": "tab:blue",
    "sdirk22": "tab:green",
    "sdirk33": "tab:purple"
}

# Helper function that sets up a generic figure.
def setup_fig(xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlim(t0, tf)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.grid(True)
    ax.xaxis.set_major_locator(MultipleLocator(base=np.pi/2))
    ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    return fig,ax

# Helper for formatting ticks as multiples of pi.
def pi_formatter(x, pos):
    frac = Fraction(x / np.pi).limit_denominator(8)
    if frac.numerator == 0:
        return "0"
    elif frac.denominator == 1:
        return f"{frac.numerator}π"
    else:
        return f"{frac.numerator}π/{frac.denominator}"

# Helper function that specifies the plot style properties.
def get_plot_style(label, method_colors):
    if label.endswith("(hdg)"):
        base_name = label.replace("(hdg)", "").strip()
        linestyle = "-"
        marker = 'o'
    elif label.endswith("(dg)"):
        base_name = label.replace("(dg)", "").strip()
        linestyle = "--"
        marker = '^'
    else:
        base_name = label.strip()
        linestyle = ":"
        marker = '.'

    color = method_colors.get(base_name, "black")

    return {
        "label": label,
        "linestyle": linestyle,
        "marker": marker,
        "markersize": 8,
        "markevery": 50,
        "color": color
    }




# Run simulation(s).
routines = [implicit_euler_hdg, implicit_euler_dg,
                      bdf2_hdg,           bdf2_dg,
                   sdirk22_hdg,        sdirk22_dg,
                   sdirk33_hdg,        sdirk33_dg]

# Figure 1: Linear plot of the relative error in the variance.
fig1, ax1 = setup_fig("Time", r"$s^2_h / s^2_e - 1$", "Time Evolution of Relative Variance Error")

# Figure 2: logaraithmic plot of the relative error in the L2-norm.
fig2, ax2 = setup_fig("Time", r"L2-norm[Ue - Uh]", "Time Evolution of Relative L2-norm[Ue - Uh]")

# Run each simulation.
for routine in routines:
    print(f"Running {routine.label}...")
    data = routine()
    style = get_plot_style(routine.label, method_colors)

    ax1.plot(data[:, 0], data[:, 1], **style)
    ax2.semilogy(data[:, 0], data[:, 2], **style)

# Add IMEX-RK for DG only.
data = imex_rk_ars443_dg()
ax1.plot(data[:, 0], data[:, 1], label=imex_rk_ars443_dg.label, linestyle='--', color='k')
ax2.semilogy(data[:, 0], data[:, 2], label=imex_rk_ars443_dg.label, linestyle='--', color='k')

# Final touches.
ax1.legend(frameon=False)
ax2.legend(loc="upper right", frameon=False)
fig1.tight_layout()
fig2.tight_layout()
plt.show()






