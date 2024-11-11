from ngsolve import *
from netgen.occ import OCCGeometry, WorkPlane
from netgen.meshing import IdentificationType
from ngsolve.meshes import MakeStructured2DMesh



# Function that generates a simple grid.
def CreateSimpleGrid(gridparam):

    # Unpack the grid specification.
    isCircle      = gridparam[0]
    isStructured  = gridparam[1] 
    isPeriodic    = gridparam[2]
    maxElemSize   = gridparam[3] 


    # For now, there is two choices: circle and a unit-square domain.
    if isCircle:
    
        face = WorkPlane().MoveTo(0, -0.5).Arc(0.5, 180).Arc(0.5, 180).Face()
        for bc, edge in zip(['right', 'left'], face.edges):
            edge.name = bc
        mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=maxElemSize))
    
    else:
    
        if isStructured:
            N = int(1 / maxElemSize)
            mesh = MakeStructured2DMesh(False, N, N, periodic_y=isPeriodic, mapping=lambda x, y: (x - 0.5, y - 0.5))
        else:
            face = WorkPlane().RectangleC(1, 1).Face()
    
            for bc, edge in zip(['bottom', 'right', 'top', 'left'], face.edges):
                edge.name = bc
            if isPeriodic:
                periodic_edge = face.edges[0]
                periodic_edge.Identify(face.edges[2], "periodic", IdentificationType.PERIODIC)
            mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=maxElemSize))


    # Return the grid.
    return mesh



# Function that checks if the boundary is curved.
def CurvedBoundary(gridparam):
    
    # For now, curvature is only needed for a circle grid.
    if gridparam[0]:
        return True
    else:
        return False
