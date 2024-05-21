import sys
sys.path.append('.')
from main import *
from ngsolve import *
from netgen.occ import *

wp = WorkPlane()
circle = WorkPlane().Circle(0, 0, 6).Face()
circle.edges.name = "outer"
circle.edges.maxh = 0.3
circle.name = "outer"
circle.maxh = 0.3

mesh = Mesh(OCCGeometry(Glue([circle, face]), dim=2).GenerateMesh())
mesh.Curve(cfg.order)

@test(name)
def exact():
    """ Exact solution for pressure pulse! """

    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.FarField(farfield, Qform=True), "outer")

    tree.directory_name = f"Ma{cfg.Mach_number.Get()}/alpha{alpha}"
    return solver


if __name__ == '__main__':

    exact()