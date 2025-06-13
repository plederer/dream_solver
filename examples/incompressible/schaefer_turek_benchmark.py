import ngsolve as ngs
import netgen.occ as occ
from dream.incompressible import IncompressibleSolver, Inflow, Wall, flowfields


shape = occ.Rectangle(20, 4.1).Circle(2, 2, 0.5).Reverse().Face()
shape.edges.name = "cyl"
shape.edges.Min(occ.X).name = "inlet"
shape.edges.Max(occ.X).name = "outlet"
shape.edges.Min(occ.Y).name = "wall"
shape.edges.Max(occ.Y).name = "wall"
mesh = ngs.Mesh(occ.OCCGeometry(shape, dim=2).GenerateMesh(maxh=0.7))

# Set up the solver configuration
cfg = IncompressibleSolver(mesh)
cfg.fem = "taylor-hood"
# cfg.fem = "hdivhdg"
cfg.time = "stationary"

# cfg.io.log.level = 10

cfg.fem.order = 3
cfg.fem.scheme = "stationary"
cfg.convection = False

cfg.reynolds_number = 100
cfg.dynamic_viscosity = "constant"

mesh.Curve(cfg.fem.order)

# Set up the boundary conditions
cfg.bcs['inlet'] = Inflow(velocity=(1.5*4*ngs.y*(4.1-ngs.y)/(4.1*4.1), 0))  # flowfields(velocity=(1, 0)))
cfg.bcs['wall|cyl'] = "wall"
cfg.bcs['outlet'] = "outflow"

# Initialize the finite element spaces, trial and test functions, gridfunctions, and boundary conditions
cfg.initialize()

fields = cfg.get_solution_fields('velocity', default_fields=False)
cfg.io.draw(fields)

with TaskManager():
    cfg.solve()

cfg.time = "transient"
cfg.fem.scheme = "imex"
cfg.time.timer.interval = (0, 100)
cfg.time.timer.step = 0.01
cfg.convection = True

cfg.fem.initialize_time_scheme_gridfunctions()
cfg.fem.initialize_symbolic_forms()

# SetNumThreads(12)
with TaskManager():
    cfg.solve()
