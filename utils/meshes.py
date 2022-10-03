from netgen.geom2d import unit_square, MakeCircle, SplineGeometry

from netgen.meshing import Element0D, Element1D, Element2D, MeshPoint, FaceDescriptor
from netgen.meshing import Mesh as ng_Mesh
from netgen.csg import *

from math import cos, sin, pi


def Get_Omesh(r,R,N,L, geom=1):
    if (N%4) > 0:
        print("N must be a multiplicative of 4!")
        quit()
    if (L%2) > 0:
        print("L must be a multiplicative of 2!")
        quit()
    if L > int(N/2):
        print("L > N/2!!! Boundary conditions need to be updated... TODO")
        quit()
    

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
        
        mesh.Add(Element2D(idx_dom, [pnums[N -1 + j * (N)], pnums[N-1 + (j+1) * (N)], pnums[j * (N)]]))
        mesh.Add(Element2D(idx_dom, [pnums[0 + j * (N)], pnums[N-1 + (j+1) * (N)], pnums[(j+1) * (N)]]))

    for i in range(N-1):
        mesh.Add(Element1D([pnums[i],pnums[i+1]], [0,1], 1))
    mesh.Add(Element1D([pnums[N-1], pnums[0]], [0,1], 1))


    offset = int(-L/2 + N/4)

    for i in range(0, offset):
        mesh.Add(Element1D([pnums[i + L * N],pnums[i + L * N + 1]], [0,2], index=3))
    

    for i in range(offset, int(N/2)+offset):
        mesh.Add(Element1D([pnums[i + L * N], pnums[i + L * N + 1]], [0,2], index=2))

    for i in range(int(N/2)+offset, N-1):
        mesh.Add(Element1D([pnums[i + L * N], pnums[i + L * N + 1]], [0,2], index=3))
    mesh.Add(Element1D([pnums[L*N], pnums[N - 1 + L * N]], [0,2], index=3))

    mesh.SetBCName(0, "cyl")
    mesh.SetBCName(1, "inflow")
    mesh.SetBCName(2, "outflow")

    return(mesh)

if __name__ == "__main__":
    from ngsolve import *
    R = 1   
    R_farfield = R * 2
    mesh = Mesh(Get_Omesh(R, R_farfield, 36, 18, geom = 2))
    mesh.Curve(4)
    print(mesh.GetBoundaries())
    Draw(mesh)
    V = H1(mesh, dirichlet=".*")
    u = GridFunction(V)
    u.Set(1, BND, definedon=mesh.Boundaries("inflow"))
    Draw(u)