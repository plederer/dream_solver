from netgen.occ import OCCGeometry, WorkPlane
from ngsolve import *
from dream import *
from netgen.meshing import IdentificationType
from ngsolve.meshes import MakeStructured2DMesh
from ngsolve.internal import *

ngsglobals.msg_level = 0
SetNumThreads(8)


periodic = False
circle = False
structured = False
maxh = 0.1

cfg = SolverConfiguration()
cfg.formulation = "conservative"
cfg.fem = "hdg"
# cfg.dynamic_viscosity = "constant"
# cfg.dynamic_viscosity = None
# cfg.mixed_method = "strain_heat"
# cfg.mixed_method = None
cfg.scaling = "aerodynamic"
cfg.riemann_solver = 'farfield'

cfg.Reynolds_number = 150
cfg.Mach_number = 0.03
cfg.Prandtl_number = 0.72
cfg.heat_capacity_ratio = 1.4

cfg.order = 4
cfg.bonus_int_order_bnd = cfg.order
cfg.bonus_int_order_vol = cfg.order

cfg.time.simulation = "transient"
cfg.time.scheme = "BDF2"
cfg.time.step = 1e-2
cfg.time.interval = (0, 200)

cfg.linear_solver = "pardiso"
cfg.damping_factor = 1
cfg.max_iterations = 10
cfg.convergence_criterion = 1e-10

cfg.compile_flag = False
cfg.static_condensation = True

if circle:

    R = 0.5
    face = WorkPlane().MoveTo(0, -R).Arc(R, 180).Arc(R, 180).Face()
    bnds = ['right', 'left']

    # face = WorkPlane().MoveTo(0, -1).Arc(1, 40).Arc(1, 100).Arc(1,40).Arc(1, 180).Face()
    # bnds = ['right', 'nscbc', 'right', 'left']
    for bc, edge in zip(bnds, face.edges):
        edge.name = bc


    # for vertex in face.vertices:
    #     vertex.maxh = 0.01
    mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=maxh))#, grading=0.1))
    # mesh.Curve(cfg.order)

else:

    if structured:
        N = int(1 / maxh)
        mesh = MakeStructured2DMesh(False, N, N, periodic_y=periodic, mapping=lambda x, y: (x - 0.5, y - 0.5))
    else:
        face = WorkPlane().RectangleC(1, 3).Face()
        # face = WorkPlane().MoveTo(-0.5, 0).RectangleC(2, 2).Face()
        face = WorkPlane().RectangleC(1, 1).Face()
        # face = WorkPlane().RectangleC(1, 2).Face()
        wp = WorkPlane()
        # face = wp.MoveTo(-0.5, -0.5).LineTo(0.5, -0.51, 'bottom').LineTo(0.5, 0.51, 'right').LineTo(-0.5, 0.5,'top').LineTo(-0.5, -0.5, 'left').Face()
        # face = wp.MoveTo(-0.5, -1).LineTo(0.5, -1.01, 'bottom').LineTo(0.5, 1.01, 'right').LineTo(-0.5, 1,'top').LineTo(-0.5, -1, 'left').Face()

        for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
            edge.name = bc
        if periodic:
            periodic_edge = face.edges[0]
            periodic_edge.Identify(face.edges[2], "periodic", IdentificationType.PERIODIC)

        # face.edges[1].maxh = 0.07
        mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=maxh))
# mesh.Refine()

gamma = cfg.heat_capacity_ratio
M = cfg.Mach_number

farfield = INF.farfield((cos(pi/6), sin(pi/6)), cfg)
farfield = INF.farfield((1, 0), cfg)
u_inf = CF(farfield.velocity)
rho_inf = farfield.density
p_inf = farfield.pressure
T_inf = farfield.temperature
c = INF.speed_of_sound(cfg)


# Vortex Pirozzoli
Mt = 0.01
R = 0.1
r = sqrt((x-0.2)**2 + y**2)

if cfg.scaling is cfg.scaling.AERODYNAMIC:
    vt = Mt/cfg.Mach_number
elif cfg.scaling is cfg.scaling.ACOUSTIC:
    vt = Mt
elif cfg.scaling is cfg.scaling.AEROACOUSTIC:
    vt = Mt/(1 + cfg.Mach_number)

psi = vt * R * exp((1 - (r/R)**2)/2)
u_0 = u_inf + CF((psi.Diff(y), -psi.Diff(x)))
p_0 = p_inf * (1 - (gamma - 1)/2 * Mt**2 * exp(1 - (r/R)**2))**(gamma/(gamma - 1))
rho_0 = rho_inf * (1 - (gamma - 1)/2 * Mt**2 *  exp(1 - (r/R)**2))**(1/(gamma - 1))
initial = State(u_0, rho_0, p_0)
# initial = State(u_inf, rho_inf, p_inf)

p_00 = p_inf * (1 - (gamma - 1)/2 * Mt**2 * exp(1))**(gamma/(gamma - 1))


solver = CompressibleHDGSolver(mesh, cfg)
solver.boundary_conditions.set(bcs.FarField(farfield, Qform=True), "top|bottom|left")
# solver.boundary_conditions.set(bcs.FarField(farfield, Qform=True), "top|bottom")
sigma = State(pressure=0.01, velocity=1, temperature=1)

solver.boundary_conditions.set(bcs.NSCBC(farfield, "yoo", sigma, tangential_flux=True, glue=False), "right")
# solver.boundary_conditions.set(bcs.GFarField(farfield, convective_tangential_flux=False, glue=False, sigma=State(pressure=0.01)), "top|bottom")
solver.boundary_conditions.set(bcs.GFarField(farfield, type="gfarfield", relaxation="farfield", convective_tangential_flux=False, viscous_fluxes=False,  glue=False, sigma=State(velocity=1, pressure=1)), "left|top|bottom")
solver.boundary_conditions.set(bcs.GFarField(farfield, type="gfarfield", relaxation="farfield", convective_tangential_flux=False, viscous_fluxes=False, glue=False, sigma=State(velocity=0.1, pressure=0.1)), "top|bottom")
# solver.boundary_conditions.set(bcs.GFarField(farfield, relaxation="outflow", convective_tangential_flux=True, glue=True, sigma=State(1, 1, 0.5, 1, 1)), "right")
solver.boundary_conditions.set(bcs.GFarField(farfield, type="gfarfield", relaxation="farfield", convective_tangential_flux=True, viscous_fluxes=True, glue=False, sigma=State(velocity=0.1, pressure=0.1)), "right")

if periodic:
    solver.boundary_conditions.set(bcs.Periodic(), "top|bottom")
solver.domain_conditions.set(dcs.Initial(initial))


with TaskManager():
    solver.setup()

    solver.drawer.draw(energy=True)
    solver.drawer.draw_particle_velocity(u_inf, autoscale=True, min=-1e-4, max=1e-4)
    solver.drawer.draw_acoustic_pressure(p_inf, autoscale=True, min=-1e-2, max=1e-2)
    Draw((solver.formulation.density() - 1), mesh, "rho*",  autoscale=False, min=-1e-4, max=1e-4)
    Draw((solver.formulation.temperature() - farfield.temperature), mesh, "T*",  autoscale=False, min=-1e-4, max=1e-4)
    Draw((solver.formulation.pressure() - p_inf)/(p_00 - p_inf), mesh, "p*",  autoscale=False, min=-1e-4, max=1e-4)

    visoptions.deformation = 1
    visoptions.vecfunction = 0
    # visoptions.subdivisions = cfg.order

    solver.solve_transient()
