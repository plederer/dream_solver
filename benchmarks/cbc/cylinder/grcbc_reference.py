#%%
import sys
sys.path.append('.')
from main import *

wp.MoveTo(0, bl*R).Direction(-1, 0).Arc(bl*R, 180)
wp.LineTo(200*R, -bl*R)
wp.LineTo(200*R, bl*R)
wp.LineTo(0, bl*R)
wake = wp.Face()
# wake = wake - (wake - wp.MoveTo(0,0).Circle(0, 0, R*200).Face())
wake.faces.maxh = 1.5
wake -= dom
wake -= cyl


sound = wp.MoveTo(0,0).RectangleC(R*400, R*400).Face()
sound.faces.maxh = 15
sound.faces.name = "sound"
for edge, name_ in zip(sound.edges, ['planar', 'outflow', 'planar', 'inflow']):
    edge.name = name_
sound -= cyl
sound = Glue([wake, sound])


dom.edges.name = "default"
dom = Glue([dom, sound])
dom.edges[4].name = "cyl"
geo = OCCGeometry(dom, dim=2)

mesh = Mesh(geo.GenerateMesh(maxh=200, grading=0.23))
mesh.Curve(order=cfg.order)

@test(name)
def grcbc_reference():
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.CBC(farfield, sigma=State(velocity=0.01, pressure=0.01)), 'inflow')
    solver.boundary_conditions.set(bcs.CBC(farfield, sigma=State(velocity=0.01, pressure=0.01)), 'planar')
    solver.boundary_conditions.set(bcs.CBC(farfield, relaxation="outflow", convective_tangential_flux=True, viscous_fluxes=True, sigma=State(velocity=1, pressure=0.01)), 'outflow')

    solver.get_saver(tree).save_mesh('mesh_gfarfield_reference')

    return solver


if __name__ == "__main__":
    grcbc_reference()
