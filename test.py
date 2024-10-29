from ngsolve import *
from dream.solver import SolverConfiguration

mesh = Mesh(unit_square.GenerateMesh(maxh=1))

cfg = SolverConfiguration(mesh)

fes = L2(mesh, order=1)
fes = FacetFESpace(mesh, order=1)**2
u, v = fes.TnT()
# blf = BilinearForm(fes)
# blf += u * v * dx(element_boundary=True)
# blf.Assemble()

# mat = blf.mat
# # mat = fes.Mass(10)
# print(mat.ToDense())

def test(i = 10):
    yield None

    # for t in range(i):
    #     yield t

    #     print(t)

for h in test():
    print(h)