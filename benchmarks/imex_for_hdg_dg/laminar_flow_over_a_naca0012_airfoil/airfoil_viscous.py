# Import preliminary modules.
from dream.compressible import Initial, CompressibleFlowSolver, flowfields, FarField, Outflow, InterfaceBC, AdiabaticWall, IsothermalWall, dimensionalfields
from dream.time import SynchronizedIMEXTimeRoutine
import ngsolve as ngs 
import numpy as np
from geometry import get_airfoil_ogrid
from dream.io import BoundarySensor, PointSensor
from dream.compressible import SpongeLayer
from dream.mesh import BufferCoord, SpongeFunction

ngs.ngsglobals.msg_level = 0 
ngs.SetNumThreads(64)


# In the following, Re is based on the chord length and velocity magnitude.
class SimulationParam_NACA0012:
    """
    Simulation parameters and derived freestream properties for a NACA0012 airfoil.
    Re is based on chord length and velocity magnitude.
    """
    def __init__(self, 
                 R: float = 287.058, # [J/(kg*K)]
                 gamma: float = 1.4,
                 L: float = 1.0,     # [m]
                 Pr: float = 0.72,
                 Ma: float = 0.5,
                 Re: float = 5000,
                 aoa: float = 0.0,    # [deg]
                 display_info: bool = False):
        self.R = R
        self.gamma = gamma
        self.L = L
        self.Pr = Pr
        self.Ma = Ma
        self.Re = Re
        self.aoa = aoa

        # Freestream static conditions (default assumed).
        self.T = 288.15 # [K]
        self.p = 101325 # [Pa]

        # Derived properties.
        self.rho = self.p / (self.R * self.T)          # [kg/m3]
        self.a = np.sqrt(self.gamma * self.R * self.T) # [m/s]
        
        # Velocity components.
        aoa_rad = np.radians(self.aoa)       # [rad]
        self.umag = self.Ma * self.a         # [m]
        self.u = self.umag * np.cos(aoa_rad) # [m] 
        self.v = self.umag * np.sin(aoa_rad) # [m]

        # Dynamic and kinematic viscosity.
        self.mu = (self.rho * self.umag * self.L) / self.Re # [Kg/(m*s)]
        self.nu = self.mu / self.rho                        # [m2/s]

        # Specific heats.
        self.cp = self.gamma * self.R / (self.gamma - 1.0)  # [J/(kg*K)]
        self.cv = self.cp / self.gamma                      # [J/(kg*K)]

        # Thermal properties.
        self.kappa = self.mu * self.cp / self.Pr       # conductivity: [Kg*m/(K*s3)]
        self.alpha = self.kappa / (self.rho * self.cp) #  diffusivity: [m2/s]

        # Display the flow conditions.
        if display_info:
            print(self, flush=True)

    def set_dimensional_fields(self, cfg: CompressibleFlowSolver) -> None:
        cfg.equation_of_state.heat_capacity_ratio = self.gamma
        cfg.dimensional_fields = dimensionalfields(rho_inf=self.rho, 
                                                   u_inf=self.umag, 
                                                   T_inf=self.T,
                                                   mu_inf=self.mu,
                                                   k_inf=self.kappa, 
                                                   c_p=self.cp, 
                                                   L=self.L)

    def get_freestream_conditions(self, cfg: CompressibleFlowSolver) -> flowfields:
        U = flowfields()
        U.rho = self.rho / cfg.scaling.reference_density
        U.u = ngs.CF( (self.u, self.v) ) / cfg.scaling.reference_velocity
        U.p = self.p / cfg.scaling.reference_pressure
        return U
    
    def __repr__(self):
        return (f"Re = {self.Re:.5e} [-]\n"
                f"Ma = {self.Ma:.3f} [-]\n"
                f"Pr = {self.Pr:.3f} [-]\n"
                f"AoA = {self.aoa:.2f} [deg]\n"
                f"T = {self.T:.2f} [K]\n"
                f"P = {self.p:.2f} [Pa]\n"
                f"rho = {self.rho:.4f} [kg/m^3]\n"
                f"a = {self.a:.2f} [m/s]\n"
                f"umag = {self.umag:.2f} [m/s]\n"
                f"u = {self.u:.2f} [m/s]\n"
                f"v = {self.v:.2f} [m/s]\n"
                f"mu = {self.mu:.5f} [Pa*s]\n"
                f"nu = {self.nu:.5f} [m^2/s]\n"
                f"R = {self.R:.3f} [J/kg*K]\n"
                f"cp = {self.cp:.3f} [J/kg*K]\n"
                f"cv = {self.cv:.3f} [J/kg*K]\n"
                f"kappa = {self.kappa:.5f} [W/(m*K)]\n"
                f"alpha = {self.alpha:.5f} [m^2/s]")




class NumericalParameters:
    riemann_solver: str = "lax_friedrich"

# Class containing time parameters.
class TimeInfo:
    def __init__(self, t0, tf, dt):
        self.t0 = t0
        self.tf = tf
        self.dt = dt

class SpongeParameters:
    r0 = 5.5 
    r1 = 7.0
    x0 = 1.0
    y0 = 0.0
    sm = 2.0
    sp = 2
    buffer_coord = BufferCoord.polar(r0=r0, rn=r1, shift=(x0,y0))
    sponge_func = SpongeFunction.polynomial(weight=sm,
                                            x=buffer_coord,
                                            order=sp)

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



# Function that displays relevant information for this simulation.
def report_statistics(cfg_imp: CompressibleFlowSolver, cfg_exp: CompressibleFlowSolver) -> None:
    
    nelem_imp = cfg_imp.mesh.ne; nelem_exp = cfg_exp.mesh.ne
    ne = nelem_imp + nelem_exp
    print( f"Number of elements: (imp, exp) = ({nelem_imp}, {nelem_exp})... total = {ne}", flush=True )



# Function that visualizes the data using VTK format.
def init_vtk_stream(cfg, plot_sol_vtk=False, nOutput=100, filename="output"):
    if plot_sol_vtk:
        
        # Extract the specific heat ratio.
        gamma = cfg.equation_of_state.heat_capacity_ratio 

        # Get the local flow fields.
        uh = cfg.get_solution_fields("grad_u", "grad_p", "grad_T", "grad_rho")
        fields = uh
        
        # Reference values.
        u_ref = cfg.scaling.reference_velocity
        p_ref = cfg.scaling.reference_pressure
        T_ref = cfg.scaling.reference_temperature
        rho_ref = cfg.scaling.reference_density
        Re_ref = cfg.scaling.reference_reynolds_number 
        
        # Dimensional values.
        p_dim = cfg.pressure(uh) * p_ref
        T_dim = cfg.temperature(uh) * T_ref
        a_dim = cfg.speed_of_sound(uh) * u_ref
        u_dim = cfg.velocity(uh) * u_ref
        rho_dim  = cfg.density(uh) * rho_ref

        fields["Dim[Velocity]"] = u_dim 
        fields["Dim[Pressure]"] = p_dim
        fields["Dim[Temperature]"] = T_dim
        fields["Dim[Density]"]  = rho_dim
        fields["Dim[Entropy]"]  = ngs.log( p_dim / rho_dim**gamma )
        fields["Dim[SpeedSound]"] = a_dim
        fields["SpeedSound"] = cfg.speed_of_sound(uh)
        fields["Mach"] = cfg.get_local_mach_number(uh)
        fields["Re"] = cfg.get_local_reynolds_number(uh) * Re_ref
        fields["Vorticity"] = cfg.vorticity(uh, uh)
        
        cfg.io.vtk.fields = fields 
        cfg.io.vtk.enable = True
        cfg.io.vtk.rate = nOutput
        cfg.io.vtk.subdivision = 2
        cfg.io.vtk.path = "airfoil_viscous"
        cfg.io.vtk.filename = filename

        # Save mesh file.
        cfg.io.ngsmesh.enable = True
        cfg.io.ngsmesh.filename = "mesh_" + filename
        
        # Save solution file.
        cfg.io.gfu.enable = True
        cfg.io.gfu.path = filename
        cfg.io.gfu.filename = "solution"
        cfg.io.gfu.rate = 1e8 
        cfg.io.gfu.time_level_rate = 1e8



# Function that sets the HDG configuration.
def init_hdg_cfg(cfg, param, time_info, multizone_time, nPoly, nStage, nOutput, plot_sol_vtk=False):

    cfg.dynamic_viscosity = "constant"
    cfg.scaling = "aerodynamic"

    cfg.riemann_solver = NumericalParameters.riemann_solver
    cfg.fem = "conservative_hdg"
    cfg.fem.order = nPoly
    cfg.fem.viscous_treatment = "mixed_strain_temperature_gradient"
    #cfg.fem.viscous_treatment = "interior_penalty"
    #cfg.fem.viscous_treatment.interior_penalty_coefficient = 10.0
    
    cfg.time = "transient"
    cfg.fem.scheme = get_imex_scheme_implicit(nStage)
    cfg.time.timer.interval = (time_info.t0, time_info.tf)
    cfg.time.timer.step = time_info.dt
    
    cfg.fem.solver.method = "newton"
    cfg.fem.solver.method.damping_factor = 1
    cfg.fem.solver.method.max_iterations = 10
    cfg.fem.solver.method.convergence_criterion = 1e-10
    
    # Get freestream conditions.
    param.set_dimensional_fields(cfg)
    Uinf = param.get_freestream_conditions(cfg)
    
    # Boundary conditions (non-interface).
    cfg.bcs['airfoil'] = AdiabaticWall()
    cfg.dcs['implicit'] = Initial(fields=Uinf)

    cfg.fem.initialize_finite_element_spaces()
    cfg.fem.initialize_trial_and_test_functions()
    cfg.fem.initialize_gridfunctions()
    cfg.fem.initialize_time_scheme_gridfunctions()

    # Set up VTK stream.
    init_vtk_stream(cfg, plot_sol_vtk, nOutput, "hdg_nse")



# Function that sets the SDG configuration.
def init_sdg_cfg(cfg, param, time_info, multizone_time, nPoly, nStage, nOutput, plot_sol_vtk=False):

    cfg.dynamic_viscosity = "constant"
    cfg.scaling = "aerodynamic"

    cfg.riemann_solver = NumericalParameters.riemann_solver
    cfg.fem = "conservative_dg"
    cfg.fem.order = nPoly
    cfg.fem.viscous_treatment = "interior_penalty"
    cfg.fem.viscous_treatment.interior_penalty_coefficient = 1.0

    cfg.time = "transient"
    cfg.fem.scheme = get_imex_scheme_explicit(nStage)
    cfg.time.timer.interval = (time_info.t0, time_info.tf)
    cfg.time.timer.step = time_info.dt
    
    cfg.fem.bonus_int_order = nPoly
    
    # Get freestream conditions.
    param.set_dimensional_fields(cfg)
    Uinf = param.get_freestream_conditions(cfg)
    
    # Boundary conditions (non-interface).
    cfg.bcs['farfield'] = FarField(fields=Uinf)
    cfg.dcs['explicit|buffer'] = Initial(fields=Uinf)

    # Sponge layer configuration. 
    sponge_func = SpongeParameters.sponge_func
    sponge_order = SpongeParameters.sp
    cfg.dcs['buffer'] = SpongeLayer(target_state=Uinf,
                                    function=sponge_func,
                                    order=sponge_order)

    cfg.fem.initialize_finite_element_spaces()
    cfg.fem.initialize_trial_and_test_functions()
    cfg.fem.initialize_gridfunctions()
    cfg.fem.initialize_time_scheme_gridfunctions()

    # Set up VTK stream.
    init_vtk_stream(cfg, plot_sol_vtk, nOutput, "sdg_nse")


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

    cfg_exp.fem.scheme.compile = True
    cfg_imp.fem.scheme.compile = True


# Number of stages in the IMEX scheme.
nStage = 3

# Time information.
time_info = TimeInfo( t0=0.0, tf=20.0, dt=0.001 )


# Polynomial order.
nPoly = 3

# File-writing frequency (every nOutput iterations).
nOutput = 100

# Initialize simulation parameters.
param = SimulationParam_NACA0012(display_info=True)


# Deduce parameters.
farfield_center = (SpongeParameters.x0, SpongeParameters.y0)
buffer_radius = SpongeParameters.r0
farfield_radius = SpongeParameters.r1


# Generate a simple grid.
mesh_imp, mesh_exp = get_airfoil_ogrid(target_sim=param,
                                       n_poly=nPoly,
                                       naca_code="0012",
                                       target_dyplus=0.7,
                                       offset_distance=0.2,
                                       farfield_radius=farfield_radius,
                                       farfield_center=farfield_center,
                                       buffer_radius=buffer_radius,
                                       interface_maxh=0.075,
                                       buffer_interface_maxh=1.0,
                                       grading=0.2,
                                       inner_maxh=0.07,
                                       wake_maxh=0.075,
                                       outer_maxh=2.0,
                                       target_dxplus=0.7,
                                       ratio_inflation=1.2,
                                       n_boundary_layers=5,
                                       inner_quad_dominated=False,
                                       outer_quad_dominated=False,
                                       show_preview=False)




# # 
# Solver configuration: Compressible (inviscid) flow.
# # 

cfg_hdg = CompressibleFlowSolver(mesh_imp)
cfg_sdg = CompressibleFlowSolver(mesh_exp)

# Create a multizone time strategy IMEX class.
time = SynchronizedIMEXTimeRoutine(cfg_implicit=cfg_hdg, cfg_explicit=cfg_sdg)

# Initialize the non-interface configurations.
init_hdg_cfg(cfg_hdg, param, time_info, time, nPoly, nStage, nOutput, plot_sol_vtk=True)
init_sdg_cfg(cfg_sdg, param, time_info, time, nPoly, nStage, nOutput, plot_sol_vtk=True)

# Initialize the interface and remaining configuration.
set_interface_conditions(cfg_hdg, cfg_sdg)


# Display statistical information.
report_statistics(cfg_imp=cfg_hdg, cfg_exp=cfg_sdg)



# # #
# Add sensors and probes.
# # 

# Extract the numerical and reference solution.
Uh = cfg_hdg.get_solution_fields("T", "p", "grad_T", "grad_u", "grad_p", "grad_rho")
Uinf = param.get_freestream_conditions(cfg_hdg)

#probe = PointSensor.from_boundary({'c_p': cfg_hdg.pressure_coefficient(Uh, Uinf),
#                                   'T':   cfg_hdg.temperature(Uh),
#                                   'grad_p': cfg_hdg.pressure_gradient(Uh, Uh)},
#                                    mesh_imp, 'airfoil')
#probe.name = "sensor_airfoil_surface"
#probe.rate = 1

sensor = BoundarySensor({"c_l": cfg_hdg.lift_coefficient(Uh, Uh, Uinf, (0, 1)), 
                         "c_d": cfg_hdg.drag_coefficient(Uh, Uh, Uinf, (1, 0))}, 
                         mesh_imp, 'airfoil')

# Additional info on the surface sensor.
sensor.name = "surface_coefficients"
sensor.rate = 10
sensor.integration_order = 10

cfg_hdg.io.sensor.enable = True
#cfg_hdg.io.sensor.add(probe)
cfg_hdg.io.sensor.add(sensor)





# # #
# Run simulation.
# # 

with ngs.TaskManager(): 
    time.solve()
        
        

