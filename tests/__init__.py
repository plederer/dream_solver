import unittest
import numpy.testing as nptest
import netgen.occ as occ
import ngsolve as ngs


def unit_square(maxh=0.25, periodic: bool = False) -> ngs.Mesh:
    wp = occ.WorkPlane()
    faces = []
    for i, x_ in enumerate([-0.375, -0.125, 0.125, 0.375]):
        face = wp.MoveTo(x_, 0).RectangleC(0.25, 1).Face()

        face.name = f"layer_{i}"

        for edge, bnd in zip(face.edges, ("bottom", "right", "top", "left")):
            edge.name = bnd

        if periodic:
            periodic_edge = face.edges[0]
            periodic_edge.Identify(face.edges[2], f"periodic_{i}", occ.IdentificationType.PERIODIC)

        faces.append(face)

    face = occ.Glue(faces)
    return ngs.Mesh(occ.OCCGeometry(face, dim=2).GenerateMesh(maxh=maxh))


def unit_circle(maxh=0.125, shift=(0, 0)) -> ngs.Mesh:
    wp = occ.WorkPlane()
    faces = []
    for i, r_ in enumerate([0.125, 0.25, 0.375, 0.5]):
        face = wp.Circle(*shift, r_).Face()

        face.name = f"layer_{i}"
        faces.append(face)

    face = occ.Glue(faces)
    return ngs.Mesh(occ.OCCGeometry(face, dim=2).GenerateMesh(maxh=maxh))


def simplex(maxh=1, dim: int = 2) -> ngs.Mesh:
    if dim == 2:
        wp = occ.WorkPlane()
        wp = wp.LineTo(1, 0).LineTo(0, 1).LineTo(0, 0).Face()
    elif dim == 3:
        wp = occ.Box(occ.Pnt(0, 0, 0), occ.Pnt(1, 1, 1))
        wp -= occ.WorkPlane(occ.Axes(p=(1, 0, 0), n=(1, 1, 1), h=(-1, 1, 0)), ).RectangleC(3, 3).Face().Extrude(3)
    else:
        raise ValueError(f"{dim}-simplex not supported!")

    return ngs.Mesh(occ.OCCGeometry(wp, dim=dim).GenerateMesh(maxh=maxh))
