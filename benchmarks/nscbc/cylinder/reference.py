#%%
import sys
sys.path.append('.')
from main import *
from netgen.webgui import Draw

wp.MoveTo(0, bl*R).Direction(-1, 0).Arc(bl*R, 180)
wp.LineTo(200*R, -bl*R)
wp.LineTo(200*R, bl*R)
wp.LineTo(0, bl*R)
wake = wp.Face()
wake = wake - (wake - wp.MoveTo(0,0).Circle(0, 0, R*200).Face())
wake.faces.maxh = 0.5
# wake.faces.maxh = 1
wake -= dom
wake -= cyl


sound = wp.MoveTo(0,0).Circle(0, 0, R*200).Circle(0, 0, R).Reverse().Face()
sound.faces.maxh = 5
sound.faces.name = "sound"
sound = Glue([wake, sound])
sponge = wp.MoveTo(0,0).Circle(0, 0, R*3000).Circle(0, 0, R*200).Reverse().Face()
sponge.faces.name = "sponge"
sponge.edges[0].name = "farfield"



# dom = Glue([dom, sound, sponge])
dom = Glue([sponge, sound, dom])
geo = OCCGeometry(dom, dim=2)

mesh = Mesh(geo.GenerateMesh(maxh=100, grading=0.2))

@test(name)
def reference():
    solver = CompressibleHDGSolver(mesh, cfg, tree)
    solver.boundary_conditions.set(bcs.FarField(farfield, Qform=True), 'farfield')

    r_ = BufferCoordinate.polar(200*R, 3000*R)
    sponge = SpongeFunction.penta_smooth(r_)

    solver.domain_conditions.set(dcs.SpongeLayer(farfield, sponge), 'sponge')

    return solver
#%%
if __name__ == "__main__":
    reference()

#%%
from ngsolve import *
fes = FacetFESpace(Mesh(unit_square.GenerateMesh(maxh=1)), order=2)
b = fes.Mass()
# %%
