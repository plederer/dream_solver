import sys
sys.path.append('.')
from main import *
from ngsolve import *
from netgen.occ import *

wp = WorkPlane()
outer = WorkPlane().Circle(0, 0, 4*H).Face()
outer.edges.maxh = 0.5
outer.edges[0].name = "outer"
outer.name = "outer"
outer.maxh = 0.5

mesh = Mesh(OCCGeometry(Glue([face, outer]), dim=2).GenerateMesh(grading=0.15))
mesh.Curve(cfg.order)

@test(name)
def reference():
    """ Exact solution for pressure pulse! """

    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.CBC(farfield, sigma=State(pressure=0.01, velocity=0.01)), "outer")
    
    tree.directory_name = f"Ma{cfg.Mach_number.Get()}/alpha{alpha}"

    solver.get_saver(tree).save_mesh('mesh_exact')

    return solver


if __name__ == '__main__':

    reference()