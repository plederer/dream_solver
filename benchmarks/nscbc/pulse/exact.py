import sys
sys.path.append('.')
from main import *
from ngsolve import *
from netgen.occ import *

wp = WorkPlane()
outer = WorkPlane().Circle(0, 0, 3).Face()
outer.edges.maxh = 0.3
outer.name = "outer"
outer.maxh = 0.3

sponge = WorkPlane().Circle(0, 0, 8).Face()
sponge.edges[0].name = "outer"
sponge.edges.maxh = 1
sponge.name = "sponge"
sponge.maxh = 1

mesh = Mesh(OCCGeometry(Glue([face, outer, sponge]), dim=2).GenerateMesh(grading=0.15))
mesh.Curve(cfg.order)

@test(name)
def exact():
    """ Exact solution for pressure pulse! """

    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.GFarField(farfield, sigma=State(pressure=0.01, velocity=0.01)), "outer")
    
    r_ = BufferCoordinate.polar(3, 8)
    sponge = SpongeFunction.penta_smooth(r_)
    solver.domain_conditions.set(dcs.SpongeLayer(farfield, sponge), 'sponge')

    tree.directory_name = f"Ma{cfg.Mach_number.Get()}/alpha{alpha}"

    solver.get_saver(tree).save_mesh('mesh_exact')

    return solver


if __name__ == '__main__':

    exact()