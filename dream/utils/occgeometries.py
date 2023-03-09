from netgen.occ import WorkPlane, OCCGeometry, Axes, Glue
from ngsolve import *


def Rectangle(p1, p2, wp=None, bottom="bottom", right="right", top="top", left="left") -> WorkPlane:

    if wp is None:
        wp = WorkPlane()

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    wp.MoveTo(*p1)
    wp.Line(dx, name=bottom).Rotate(90)
    wp.Line(dy, name=right).Rotate(90)
    wp.Line(dx, name=top).Rotate(90)
    wp.Line(dy, name=left).Rotate(90)

    return wp


def HalfSplitCircle(center, radius, wp=None, left="left", right="right", mid="mid") -> WorkPlane:

    wp = WorkPlane(Axes())
    wp.MoveTo(center[0], center[1]+radius).Rotate(180)
    wp.Arc(radius, 180)
    wp.LineTo(center[0], center[1]+radius)
    wp.LineTo(center[0], center[1]-radius)
    wp.Arc(radius, 180)

    geo = OCCGeometry(wp.Face(), dim=2)
    mesh = Mesh(geo.GenerateMesh(maxh=0.2))
    mesh.ngmesh.SetBCName(0, "left")
    mesh.ngmesh.SetBCName(1, "mid")
    mesh.ngmesh.SetBCName(2, "mid")
    mesh.ngmesh.SetBCName(3, "right")

    return mesh

import sys
sys.path.insert(1, './')
from geometries import MakeOCCSpongeCircle

geo = MakeOCCSpongeCircle((0,0), 2, 1)
mesh = Mesh(geo.GenerateMesh(maxh=0.2))
mesh.Curve(2)
print(mesh.GetBoundaries())
print(mesh.GetMaterials())

n = specialcf.normal(2)
t = specialcf.tangential(2)

fes = VectorL2(mesh, order=2, dirichlet=".*")
gfu = GridFunction(fes)

gfu.Set((2,-2), definedon=mesh.Materials("Outer"))

Draw(gfu, mesh, "gfu")
