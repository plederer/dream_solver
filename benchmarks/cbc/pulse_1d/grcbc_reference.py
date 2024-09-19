import sys
sys.path.append('.')
from netgen.occ import *
from ngsolve import *
from main import *


wp = WorkPlane()
buffer = wp.RectangleC(14*W, 4*maxh).Face()
for edge, bc in zip(buffer.edges, ['bottom', 'outflow', 'top', 'inflow']):
    edge.name = bc

for edge, bc in zip(face.edges, ['bottom', 'default', 'top', 'default']):
    edge.name = bc

geo = Glue([buffer, face])

for i, k in zip([1, 6, 9], [3, 4, 11]):
    geo.edges[i].Identify(geo.edges[k], f"periodic{i}_{k}", IdentificationType.PERIODIC)

mesh = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=maxh))


@test(name)
def grcbc_farfield_reference():
    """ Exact solution for pressure pulse! """

    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.CBC(farfield, sigma=State(pressure=1e-5, velocity=1e-5)), "inflow|outflow")

    solver.get_saver(tree).save_mesh('mesh_exact')

    return solver


if __name__ == '__main__':
    grcbc_farfield_reference()
