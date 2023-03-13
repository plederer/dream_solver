import numpy as np
from netgen.occ import WorkPlane, OCCGeometry, Glue
from netgen.meshing import Element0D, Element1D, Element2D, MeshPoint, FaceDescriptor
from netgen.meshing import Mesh as ng_Mesh
from netgen.csg import *
from math import cos, sin, pi
from typing import Optional


def circular_cylinder_mesh(radius: float = 0.5,
                           sponge_layer: bool = False,
                           boundary_layer_levels: int = 5,
                           boundary_layer_thickness: float = 0.0,
                           transition_layer_levels: int = 5,
                           transition_layer_growth: float = 1.4,
                           transition_radial_factor: float = 6,
                           farfield_radial_factor: float = 50,
                           sponge_radial_factor: float = 60,
                           wake_maxh: float = 2,
                           farfield_maxh: float = 4,
                           sponge_maxh: float = 4,
                           bc_inflow: str = "inflow",
                           bc_outflow: str = "outflow",
                           bc_cylinder: str = "cylinder",
                           bc_sponge: str = "sponge",
                           curve_layers: bool = False):

    if boundary_layer_thickness < 0:
        raise ValueError(f"Boundary Layer Thickness needs to be greater equal Zero!")
    if not sponge_layer:
        sponge_radial_factor = farfield_radial_factor
        sponge_maxh = farfield_maxh
    elif sponge_radial_factor < farfield_radial_factor and sponge_layer:
        raise ValueError("Sponge Radial Factor must be greater than Farfield Radial Factor")

    bl_radius = radius + boundary_layer_thickness
    tr_radius = transition_radial_factor * radius
    ff_radius = farfield_radial_factor * radius
    sp_radius = sponge_radial_factor * radius

    wp = WorkPlane()

    # Cylinder
    cylinder = wp.Circle(radius).Face()
    cylinder.edges[0].name = bc_cylinder

    # Viscous regime
    if boundary_layer_thickness > 0:
        bl_maxh = boundary_layer_thickness/boundary_layer_levels
        bl_radial_levels = np.linspace(radius, bl_radius, int(boundary_layer_levels) + 1)
        bl_faces = [wp.Circle(r).Face() for r in np.flip(bl_radial_levels[1:])]
        for bl_face in bl_faces:
            bl_face.maxh = bl_maxh
        boundary_layer = Glue(bl_faces) - cylinder

    # Transition regime
    tr_layer_growth = np.linspace(0, 1, transition_layer_levels+1)**transition_layer_growth
    tr_radial_levels = bl_radius + (tr_radius - bl_radius) * tr_layer_growth
    tr_maxh = np.diff(tr_radial_levels)
    tr_faces = [wp.Circle(r).Face() for r in np.flip(tr_radial_levels[1:])]
    for tr_face, maxh in zip(tr_faces, tr_maxh):
        tr_face.maxh = maxh
    transition_regime = Glue(tr_faces) - cylinder

    # Farfield region
    farfield = wp.MoveTo(0, 0).Circle(ff_radius).Face()
    farfield.maxh = farfield_maxh

    # Wake region
    wake_radius = tr_radius + maxh
    wp.MoveTo(0, wake_radius).Direction(-1, 0)
    wp.Arc(wake_radius, 180)
    wp.LineTo(ff_radius, -wake_radius)
    wp.LineTo(ff_radius, wake_radius)
    wp.LineTo(0, wake_radius)
    wake = wp.Face() - transition_regime - cylinder
    wake = wake * farfield
    wake.maxh = wake_maxh

    # Outer region (if defined)
    wp.MoveTo(0, sp_radius).Direction(-1, 0)
    wp.Arc(sp_radius, 180)
    wp.Arc(sp_radius, 180)
    outer = wp.Face()

    for edge, bc in zip(outer.edges, [bc_inflow, bc_outflow]):
        edge.name = bc

    if sponge_layer:
        for face in outer.faces:
            face.name = bc_sponge
        outer = outer - farfield
        outer.maxh = sponge_maxh
        outer = Glue([outer, farfield])

    sound = Glue([outer - wake, wake * outer]) - transition_regime - cylinder

    geo = Glue([sound, transition_regime])
    if boundary_layer_thickness > 0:
        geo = Glue([geo, boundary_layer])

    geo = OCCGeometry(geo, dim=2)
    mesh = geo.GenerateMesh(maxh=sponge_maxh)

    if not curve_layers:

        geo = outer - cylinder
        geo = OCCGeometry(geo, dim=2)

        new_mesh = ng_Mesh()
        new_mesh.dim = 2
        new_mesh.SetGeometry(geo)

        edge_map = set(elem.edgenr for elem in mesh.Elements1D())

        new_mesh.Add(FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))
        new_mesh.Add(FaceDescriptor(surfnr=2, domin=1, domout=0, bc=2))
        new_mesh.Add(FaceDescriptor(surfnr=3, domin=1, domout=0, bc=3))

        new_mesh.SetBCName(0, bc_inflow)
        new_mesh.SetBCName(1, bc_outflow)
        new_mesh.SetBCName(2, bc_cylinder)

        idx_dom = new_mesh.AddRegion("default", dim=2)
        new_mesh.SetMaterial(idx_dom, 'default')

        if sponge_layer:
            new_mesh.SetBCName(3, "default")
            sponge_dom = new_mesh.AddRegion(bc_sponge, dim=2)
            new_mesh.SetMaterial(sponge_dom, bc_sponge)
            edge_map = {1: (0, 1), 2: (1, 2), 3: (2, 4),  4: (2, 4), 5: (2, 4), max(edge_map): (3, 3)}
        else:
            edge_map = {6: (0, 1), 8: (1, 2), 7: (1, 2), 5: (1, 2), 1: (1, 2), max(edge_map): (2, 3)}

        for point in mesh.Points():
            new_mesh.Add(point)

        for elem in mesh.Elements2D():
            if sponge_layer and elem.index == 1:
                new_mesh.Add(Element2D(sponge_dom, elem.vertices))
            else:
                new_mesh.Add(Element2D(idx_dom, elem.vertices))

        for elem in mesh.Elements1D():
            if elem.edgenr in edge_map:
                edgenr, index = edge_map[elem.edgenr]
                new_mesh.Add(Element1D(elem.points, elem.surfaces, index, edgenr))

        mesh = new_mesh

    return mesh


def angular_cylinder_mesh(radius: float = 0.5,
                          sponge_layer: bool = False,
                          boundary_layer_levels: int = 5,
                          boundary_layer_thickness: float = 0.0,
                          transition_layer_levels: int = 5,
                          transition_layer_growth: float = 1.4,
                          transition_radial_factor: float = 6,
                          farfield_radial_factor: float = 50,
                          sponge_radial_factor: float = 60,
                          wake_factor: float = 1,
                          wake_maxh: float = 2,
                          farfield_maxh: float = 4,
                          sponge_maxh: float = 4,
                          bc_inflow: str = "inflow",
                          bc_outflow: str = "outflow",
                          bc_cylinder: str = "cylinder",
                          bc_wake: str = "outflow",
                          bc_sponge: str = "sponge",
                          curve_layers: bool = False):

    if boundary_layer_thickness < 0:
        raise ValueError(f"Boundary Layer Thickness needs to be greater equal Zero!")
    if not sponge_layer:
        sponge_radial_factor = farfield_radial_factor
        sponge_maxh = farfield_maxh
    elif sponge_radial_factor < farfield_radial_factor and sponge_layer:
        raise ValueError("Sponge Radial Factor must be greater than Farfield Radial Factor")

    bl_radius = radius + boundary_layer_thickness
    tr_radius = transition_radial_factor * radius
    ff_radius = farfield_radial_factor * radius
    sp_radius = sponge_radial_factor * radius
    wake_length = ff_radius * wake_factor

    wp = WorkPlane()

    # Cylinder
    cylinder = wp.Circle(radius).Face()
    cylinder.edges[0].name = bc_cylinder

    # Viscous regime
    if boundary_layer_thickness > 0:
        bl_maxh = boundary_layer_thickness/boundary_layer_levels
        bl_radial_levels = np.linspace(radius, bl_radius, int(boundary_layer_levels) + 1)
        bl_faces = [wp.Circle(r).Face() for r in np.flip(bl_radial_levels[1:])]
        for bl_face in bl_faces:
            bl_face.maxh = bl_maxh
        boundary_layer = Glue(bl_faces) - cylinder

    # Transition regime
    tr_layer_growth = np.linspace(0, 1, transition_layer_levels+1)**transition_layer_growth
    tr_radial_levels = bl_radius + (tr_radius - bl_radius) * tr_layer_growth
    tr_maxh = np.diff(tr_radial_levels)
    tr_faces = [wp.Circle(r).Face() for r in np.flip(tr_radial_levels[1:])]
    for tr_face, maxh in zip(tr_faces, tr_maxh):
        tr_face.maxh = maxh
    transition_regime = Glue(tr_faces) - cylinder

    # Farfield region
    wp.MoveTo(0, ff_radius).Direction(-1, 0)
    wp.Arc(ff_radius, 180)
    wp.LineTo(wake_length, -ff_radius)
    wp.LineTo(wake_length, ff_radius)
    wp.LineTo(0, ff_radius)
    farfield = wp.Face()
    farfield.maxh = farfield_maxh

    # Wake region
    wake_radius = tr_radius + maxh
    wp.MoveTo(0, wake_radius).Direction(-1, 0)
    wp.Arc(wake_radius, 180)
    wp.LineTo(wake_length, -wake_radius)
    wp.LineTo(wake_length, wake_radius)
    wp.LineTo(0, wake_radius)
    wake = wp.Face() - transition_regime - cylinder
    wake.maxh = wake_maxh

    # Outer region (if defined)
    sponge_length = sp_radius - ff_radius
    wp.MoveTo(0, sp_radius).Direction(-1, 0)
    wp.Arc(sp_radius, 180)
    wp.LineTo(wake_length + sponge_length, -sp_radius)
    wp.LineTo(wake_length + sponge_length, sp_radius)
    wp.LineTo(0, sp_radius)
    outer = wp.Face()

    for edge, bc in zip(outer.edges, [bc_inflow, bc_wake, bc_outflow, bc_wake]):
        edge.name = bc

    if sponge_layer:
        outer = Glue([outer - farfield, farfield])

        zone1 = wp.MoveTo(0, -sp_radius).Rectangle(wake_length + sponge_length, sponge_length).Face()
        zone2 = wp.MoveTo(0, sp_radius - sponge_length).Rectangle(wake_length + sponge_length, sponge_length).Face()
        zone3 = wp.MoveTo(wake_length, -sp_radius).Rectangle(sponge_length, 2*sp_radius).Face()
        outer = Glue([outer, zone1, zone2, zone3])

        sponge_boundaries = [f"{bc_sponge}_{bc}"
                             for bc in ('inflow', 'wake_top', 'wake_bot', 'corner_top', 'corner_bot', 'outflow')]
        for face, bc in zip(outer.faces, sponge_boundaries):
            face.name = bc
            face.maxh = sponge_maxh

    sound = Glue([outer - wake, wake * outer]) - transition_regime - cylinder

    geo = Glue([sound, transition_regime])
    if boundary_layer_thickness > 0:
        geo = Glue([geo, boundary_layer])

    geo = OCCGeometry(geo, dim=2)
    mesh = geo.GenerateMesh(maxh=sponge_maxh)

    if not curve_layers:

        geo = outer - cylinder
        geo = OCCGeometry(geo, dim=2)

        new_mesh = ng_Mesh()
        new_mesh.dim = 2
        new_mesh.SetGeometry(geo)

        edge_map = set(elem.edgenr for elem in mesh.Elements1D())

        new_mesh.Add(FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))
        new_mesh.Add(FaceDescriptor(surfnr=2, domin=1, domout=0, bc=2))
        new_mesh.Add(FaceDescriptor(surfnr=3, domin=1, domout=0, bc=3))
        new_mesh.Add(FaceDescriptor(surfnr=4, domin=1, domout=0, bc=4))

        new_mesh.SetBCName(0, bc_inflow)
        new_mesh.SetBCName(1, bc_outflow)
        new_mesh.SetBCName(2, bc_cylinder)
        new_mesh.SetBCName(3, bc_wake)

        if sponge_layer:
            new_mesh.Add(FaceDescriptor(surfnr=5, domin=1, domout=0, bc=5))
            new_mesh.Add(FaceDescriptor(surfnr=6, domin=1, domout=0, bc=6))
            new_mesh.Add(FaceDescriptor(surfnr=7, domin=1, domout=0, bc=7))
            new_mesh.Add(FaceDescriptor(surfnr=8, domin=1, domout=0, bc=8))
            new_mesh.SetBCName(4, "default")

            face_map = []
            for index, bc in enumerate(sponge_boundaries):
                sponge_dom = new_mesh.AddRegion(bc, dim=2)
                new_mesh.SetMaterial(sponge_dom, bc)
                face_map.append(sponge_dom)

            inflow_map = {4: (3, 1)}
            outflow_map = {12: (11, 2), 15: (14, 2), 17: (16, 2)}
            wake_map = {5: (4, 4), 10: (9, 4), 11: (10, 4), 16: (15, 4)}
            cylinder_map = {max(edge_map): (18, 3)}
            default_map = {
                1: (0, 5),
                2: (1, 5),
                3: (2, 5),
                6: (5, 5),
                7: (6, 5),
                8: (7, 5),
                9: (8, 5),
                13: (12, 5),
                14: (13, 5),
                18: (17, 5),
                19: (17, 5),
                20: (17, 5)}
            edge_map = {**inflow_map, **outflow_map, **wake_map, **cylinder_map, **default_map}
        else:
            edge_map = {8: (0, 1), 1: (1, 4), 7: (2, 4),  6: (3, 2), 9: (3, 2), 2: (3, 2), max(edge_map): (4, 3)}

        idx_dom = new_mesh.AddRegion("default", dim=2)
        new_mesh.SetMaterial(idx_dom, 'default')

        for point in mesh.Points():
            new_mesh.Add(point)

        for elem in mesh.Elements2D():
            if sponge_layer:
                if elem.index in face_map:
                    new_mesh.Add(Element2D(elem.index, elem.vertices))
                else:
                    new_mesh.Add(Element2D(idx_dom, elem.vertices))
            else:
                new_mesh.Add(Element2D(idx_dom, elem.vertices))

        for elem in mesh.Elements1D():
            if elem.edgenr in edge_map:
                edgenr, index = edge_map[elem.edgenr]
                new_mesh.Add(Element1D(elem.points, elem.surfaces, index, edgenr))

        mesh = new_mesh

    return mesh


def Get_Omesh(r, R, N, L, geom=1):
    if (N % 4) > 0:
        print("N must be a multiplicative of 4!")
        quit()
    if (L % 2) > 0:
        print("L must be a multiplicative of 2!")
        quit()
    if L > int(N/2):
        print("L > N/2!!! Boundary conditions need to be updated... TODO")
        quit()

    mesh = ng_Mesh()
    mesh.dim = 2

    top = Plane(Pnt(0, 0, 0), Vec(0, 0, 1))
    bot = Plane(Pnt(0, 0, 1), Vec(0, 0, -1))
    ring = Cylinder(Pnt(0, 0, 0), Pnt(0, 0, 1), R)
    inner = Cylinder(Pnt(0, 0, 0), Pnt(0, 0, 1), r)
    geo = CSGeometry()
    geo.SetBoundingBox(Pnt(-R, -R, -R), Pnt(R, R, R))
    geo.Add(top)
    geo.Add(inner)
    geo.Add(ring)

    mesh.SetGeometry(geo)

    pnums = []
    for j in range(L+1):
        for i in range(N):
            phi = pi/N * j
            px = cos(2 * pi * i/N + phi)
            py = sin(2 * pi * i/N + phi)

            ri = (R - r) * (j/L)**geom + r

            pnums.append(mesh.Add(MeshPoint(Pnt(ri * px, ri * py, 0))))

    # print(pnums)
    mesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    mesh.Add(FaceDescriptor(surfnr=2, domin=1, domout=0, bc=1))
    mesh.Add(FaceDescriptor(surfnr=3, domin=1, domout=0, bc=2))

    idx_dom = 1

    for j in range(L):
        # print("j=",j)
        for i in range(N-1):
            # print("i=",i)
            # offset =
            mesh.Add(Element2D(idx_dom, [pnums[i + j * (N)], pnums[i + (j+1) * (N)], pnums[i + 1 + j * (N)]]))
            mesh.Add(Element2D(idx_dom, [pnums[i + (j+1) * (N)], pnums[i + (j+1) * (N) + 1], pnums[i + 1 + j * (N)]]))

        mesh.Add(Element2D(idx_dom, [pnums[N - 1 + j * (N)], pnums[N-1 + (j+1) * (N)], pnums[j * (N)]]))
        mesh.Add(Element2D(idx_dom, [pnums[0 + j * (N)], pnums[N-1 + (j+1) * (N)], pnums[(j+1) * (N)]]))

    for i in range(N-1):
        mesh.Add(Element1D([pnums[i], pnums[i+1]], [0, 1], 1))
    mesh.Add(Element1D([pnums[N-1], pnums[0]], [0, 1], 1))

    offset = int(-L/2 + N/4)

    for i in range(0, offset):
        mesh.Add(Element1D([pnums[i + L * N], pnums[i + L * N + 1]], [0, 2], index=3))

    for i in range(offset, int(N/2)+offset):
        mesh.Add(Element1D([pnums[i + L * N], pnums[i + L * N + 1]], [0, 2], index=2))

    for i in range(int(N/2)+offset, N-1):
        mesh.Add(Element1D([pnums[i + L * N], pnums[i + L * N + 1]], [0, 2], index=3))
    mesh.Add(Element1D([pnums[L*N], pnums[N - 1 + L * N]], [0, 2], index=3))

    mesh.SetBCName(0, "cylinder")
    mesh.SetBCName(1, "inflow")
    mesh.SetBCName(2, "outflow")

    return (mesh)


if __name__ == "__main__":
    from ngsolve import *
    R = 1
    R_farfield = R * 2
    mesh = Mesh(Get_Omesh(R, R_farfield, 36, 18, geom=2))
    mesh.Curve(4)
    print(mesh.GetBoundaries())
    Draw(mesh)
    V = H1(mesh, dirichlet=".*")
    u = GridFunction(V)
    u.Set(1, BND, definedon=mesh.Boundaries("inflow"))
    Draw(u)
