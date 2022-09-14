from netgen.geom2d import unit_square, MakeCircle, SplineGeometry

from netgen.meshing import Element0D, Element1D, Element2D, MeshPoint, FaceDescriptor
from netgen.meshing import Mesh as ng_Mesh
from netgen.csg import *

from math import cos, sin, pi


def Get_Omesh(r,R,N,L, geom=1):
    mesh = ng_Mesh()
    mesh.dim = 2

    top   = Plane (Pnt(0,0,0), Vec(0,0,1) )
    bot   = Plane (Pnt(0,0,1), Vec(0,0,-1) )
    ring  = Cylinder ( Pnt(0, 0, 0), Pnt(0, 0, 1), R)
    inner  = Cylinder ( Pnt(0, 0, 0), Pnt(0, 0, 1), r)
    geo = CSGeometry()
    geo.SetBoundingBox(Pnt(-R,-R,-R),Pnt(R,R,R))
    geo.Add(top)
    geo.Add(inner)
    geo.Add(ring)
    
    mesh.SetGeometry(geo)

    pnums = []
    for i in range(N):
        px = cos(2 * pi * i/N)
        py = sin(2 * pi * i/N)
        for j in range(L+1):
            ri = (R - r)* (j/L)**geom + r
            pnums.append(mesh.Add(MeshPoint(Pnt(ri * px, ri * py, 0))))

    mesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=1))
    mesh.Add(FaceDescriptor(surfnr=2, domin=1, domout=0, bc=1))
    mesh.Add(FaceDescriptor(surfnr=3, domin=1, domout=0, bc=2))

    
    idx_dom = 1 
    
    for i in range(N-1):
        for j in range(L):
            mesh.Add(Element2D(idx_dom, [pnums[i * (L+1) + j], pnums[i * (L+1) + j + 1], pnums[(i+1) * (L+1) + j]]))
            mesh.Add(Element2D(idx_dom, [pnums[(i+1) * (L+1) + j], pnums[i * (L+1) + j + 1], pnums[(i+1) * (L+1) + j+1]]))

    for j in range(L):
            mesh.Add(Element2D(idx_dom, [pnums[(N - 1) * (L+1) + j], pnums[(N-1) * (L+1) + j + 1], pnums[j]]))
            mesh.Add(Element2D(idx_dom, [pnums[j], pnums[(N-1) * (L+1) + j + 1], pnums[j+1]]))

    for i in range(N-1):
        mesh.Add(Element1D([pnums[i*(L+1)],pnums[(i+1)*(L+1)]], [0,1], 1))

    mesh.Add(Element1D([pnums[(N-1)*(L+1)], pnums[0]], [0,1], 1))

    for i in range(N-1):
        mesh.Add(Element1D([pnums[(i+1)*(L+1) + L],pnums[i*(L+1) + L]], [0,2], index=2))
    mesh.Add(Element1D([pnums[(N-1)*(L+1)+ L], pnums[L]], [0,2], index=2))

    mesh.SetBCName(0, "cyl")
    mesh.SetBCName(1, "inflow")

    return(mesh)

if __name__ == "__main__":
    from ngsolve import *
    R = 1   
    R_farfield = R * 2
    mesh = Mesh(Get_Omesh(R, R_farfield, 12, 3))
    mesh.Curve(2)
    print(mesh.GetBoundaries())
    Draw(mesh)