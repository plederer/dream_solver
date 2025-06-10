from dream import *
from dream.scalar_transport import Initial, transportfields, ScalarTransportSolver
from ngsolve import *
from ngsolve.meshes import Make1DMesh
import numpy as np  
from matplotlib import pyplot as plt 


# Number of elements per dimension.
ne = 30
x0 = -2.0 
x1 =  2.0 
lx = x1-x0

# Generate a simple grid.
mesh = Make1DMesh(ne, periodic=True, mapping=lambda x: lx*x + x0 )

# Message output detail from netgen.
ngs.ngsglobals.msg_level = 0 
ngs.SetNumThreads(4)


# Define analytic solution.
def get_analytic_solution(t, v): 

    # Initial pulse location.
    x0 = 0.0

    # Variance.
    s2 = get_variance_pulse()
    # Standard deviation.
    mu = sqrt(s2)
    # Pulse center/expectation.
    xt = -v*t + ngs.x 
    # Radial distance (time-dependent).
    r2 = (xt-x0)**2
    
    # Return the analytic solution.
    #return ngs.exp( -r2/(2.0*s2) )/( mu*sqrt(ngs.pi) )

    # Number of cycle throughout the simulation.
    n_cycle = 4

    f = ngs.exp( -r2/(2.0*s2) )/( mu*sqrt(ngs.pi) )
    for i in range(1,n_cycle):
        f += ngs.exp( -(xt-x0+i*lx)**2/(2.0*s2) )/( mu*sqrt(ngs.pi) )

    return f 

def get_variance_pulse(): 
    mu = 0.15 # Standard deviation.
    return mu*mu




# Solver configuration: pure advection equation.
cfg = ScalarTransportSolver(mesh)

cfg.convection_velocity = (1.0,)
cfg.is_inviscid = True

cfg.riemann_solver = "lax_friedrich"
cfg.fem = "dg" # NOTE, by default, DG is used.
cfg.fem.order = 0


cfg.time = "transient"
cfg.fem.scheme = "implicit_euler"
cfg.time.timer.interval = (0.0, 12.0)
cfg.time.timer.step = 0.01

# cfg.linear_solver = "pardiso"
cfg.optimizations.static_condensation = False  # NOTE, by default, condensation is turned off.
cfg.optimizations.compile.realcompile = False

U0 = transportfields()
U0.phi = get_analytic_solution(cfg.time.timer.interval[0], cfg.convection_velocity[0])

cfg.bcs['left|right'] = "periodic"
cfg.dcs['dom'] = Initial(fields=U0)


t0, tf = cfg.time.timer.interval
dt = cfg.time.timer.step.Get()
nt = int(round((tf - t0) / dt))


plt.ioff()
def _init_plot(x, y1, y2, label):
    fig, ax = plt.subplots()
    num_line, = ax.plot(x, y1, 'r-', label="numerical")
    exact_line, = ax.plot(x, y2, 'k--', label="exact")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title(label)
    ax.grid(True)
    ax.legend(frameon=False)
    return fig, ax, num_line, exact_line

def _update_plot(line1, line2, y1, y2, t, label):
    line1.set_ydata(y1)
    line2.set_ydata(y2)
    ymin, ymax = min(np.min(y1), np.min(y2)), max(np.max(y1), np.max(y2))
    ax = line1.axes
    ax.set_ylim(ymin * 1.1, ymax * 1.1)
    ax.set_title(f"{label} at t = {t:.3f}")
    ax.figure.canvas.draw()
    ax.figure.canvas.flush_events()
    plt.pause(0.05)

def wave1d_routine(label):
    def decorator(func):
        def wrapper(*args, draw_solution=False, plot_freq=10, **kwargs):
            cfg.optimizations.static_condensation = False
            func(*args, **kwargs)
            cfg.fem.order = 5
            if cfg.fem.name == "hdg":
                cfg.optimizations.static_condensation = True
            cfg.initialize()

            uh = cfg.fem.get_fields("phi").phi
            ue_func = get_analytic_solution(cfg.time.timer.t, cfg.convection_velocity[0])
            xcoor = np.linspace(x0, x1, ne * cfg.fem.order, dtype=float)

            if draw_solution:
                fig, ax, num_line, exact_line = _init_plot(
                    xcoor, uh(mesh(xcoor)), ue_func(mesh(xcoor)), label
                )

            qorder = 10
            data = np.zeros((nt, 2), dtype=float)

            for i, t in enumerate(cfg.time.start_solution_routine(True)):
                err = np.sqrt(ngs.Integrate((ue_func - uh) ** 2, mesh, order=qorder))
                data[i] = [cfg.time.timer.t.Get(), err]

                if draw_solution and i % plot_freq == 0:
                    _update_plot(num_line, exact_line, uh(mesh(xcoor)), ue_func(mesh(xcoor)), t, label)

            if draw_solution:
                plt.pause(1.0)
                plt.close(fig)
            
            u_final = uh(mesh(xcoor))  # Evaluate at final time
            return data, xcoor, u_final
        wrapper.label = label
        return wrapper
    return decorator



# Specialized routines.
@wave1d_routine("implicit_euler(hdg)")
def implicit_euler_hdg():
    cfg.fem = "hdg"
    cfg.fem.scheme = "implicit_euler"
    
@wave1d_routine("implicit_euler(dg)")
def implicit_euler_dg():
    cfg.fem = "dg"
    cfg.fem.scheme = "implicit_euler"

@wave1d_routine("bdf2(hdg)")
def bdf2_hdg():
    cfg.fem = "hdg"
    cfg.fem.scheme = "bdf2"
@wave1d_routine("bdf2(dg)")
def bdf2_dg():
    cfg.fem = "dg"
    cfg.fem.scheme = "bdf2"

@wave1d_routine("sdirk22(hdg)")
def sdirk22_hdg():
    cfg.fem = "hdg"
    cfg.fem.scheme = "sdirk22"
@wave1d_routine("sdirk22(dg)")
def sdirk22_dg():
    cfg.fem = "dg"
    cfg.fem.scheme = "sdirk22"

@wave1d_routine("sdirk33(hdg)")
def sdirk33_hdg():
    cfg.fem = "hdg"
    cfg.fem.scheme = "sdirk33"
@wave1d_routine("sdirk33(dg)")
def sdirk33_dg():
    cfg.fem = "dg"
    cfg.fem.scheme = "sdirk33"

@wave1d_routine("imex_rk_ars443(dg)")
def imex_rk_ars443_dg():
    cfg.fem = "dg"
    cfg.fem.scheme = "imex_rk_ars443"

@wave1d_routine("ssprk3(dg)")
def ssprk3_dg():
    cfg.fem = "dg"
    cfg.fem.scheme = "ssprk3"

@wave1d_routine("crk4(dg)")
def crk4_dg():
    cfg.fem = "dg"
    cfg.fem.scheme = "crk4"

@wave1d_routine("explicit_euler(dg)")
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
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.grid(True)
    return fig,ax


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


# Figure 2: logaraithmic plot of the relative error in the L2-norm.
fig2, ax2 = setup_fig("Time", r"L2-norm[Ue - Uh]", "Time Evolution of Relative L2-norm[Ue - Uh]")

# Figure 3: linear plot for the final solution.
fig3, ax3 = setup_fig("x", "u(x, t_final)", f"Final Solution at t = {tf}")

# Run each simulation.
for routine in routines:
    print(f"Running {routine.label}...")
    data, xcoor, usol = routine(draw_solution=True)
    style = get_plot_style(routine.label, method_colors)

    ax2.semilogy(data[:, 0], data[:, 1], **style)
    ax3.plot(xcoor, usol, **style)


# Final touches.
ax2.legend(loc="upper right", frameon=False)
fig2.tight_layout()

# Plot analytic solution at the final step.
ue_final = get_analytic_solution(data[-1,0], cfg.convection_velocity[0])
ax3.plot(xcoor, ue_final(mesh(xcoor)), 'k-')

ax3.legend(loc="upper right", frameon=False)
ax3.set_xlim(x0, x1)
fig3.tight_layout()
plt.show()



