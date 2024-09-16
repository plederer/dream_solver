import sys
sys.path.append('.')
from main import *
from ngsolve import *
from netgen.occ import *

wp = WorkPlane()
right = wp.MoveTo(W/2, -2*maxh).Rectangle(3.5*W, 4*maxh).Face()
for bc, edge in zip(['bottom', 'outflow', 'top', 'default'], right.edges):
    edge.name = bc
right_edge = right.edges[0]
right_edge.Identify(right.edges[2], "periodic_right", IdentificationType.PERIODIC)

left = wp.MoveTo(-W/2-3.5*W, -2*maxh).Rectangle(3.5*W, 4*maxh).Face()
for bc, edge in zip(['bottom', 'default', 'top', 'inflow'], left.edges):
    edge.name = bc
left_edge = left.edges[0]
left_edge.Identify(left.edges[2], "periodic_right", IdentificationType.PERIODIC)

mesh = Mesh(OCCGeometry(Glue([face, left, right]), dim=2).GenerateMesh(maxh=maxh))

@test(name)
def grcbc_farfield_reference():
    """ Exact solution for pressure pulse! """

    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.CBC(farfield, sigma=State(pressure=1e-5, velocity=1e-5)), "inflow|outflow")

    solver.get_saver(tree).save_mesh('mesh_exact')

    return solver


if __name__ == '__main__':

    grcbc_farfield_reference()