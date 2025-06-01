from ngsolve import *
from netgen.occ import OCCGeometry, WorkPlane
from netgen.meshing import IdentificationType
from ngsolve.meshes import MakeStructured2DMesh



# Function that generates a simple rectangular grid.
def CreateSimpleGrid(nElem1D, xLength, yLength):

    # Estimated element length, per dimension.
    hx = xLength/float(nElem1D)
    hy = yLength/float(nElem1D)

    # Select a common element size.
    hsize = min( hx, hy ) 

    # This is the domain. 
    domain = WorkPlane().RectangleC(xLength, yLength).Face()

    # Assign the name of the internal solution in the domain.
    domain.name = 'internal'

    # For convenience, extract and name each of the edges consistently.
    bottom = domain.edges[0]; bottom.name = 'bottom'
    right  = domain.edges[1]; right.name  = 'right'
    top    = domain.edges[2]; top.name    = 'top'
    left   = domain.edges[3]; left.name   = 'left'

    # Initialize a rectangular 2D geometry.
    geo = OCCGeometry(domain, dim=2)

    # Discretize the domain.
    mesh = Mesh(geo.GenerateMesh(maxh=hsize, quad_dominated=True))

    # To refine the mesh by a factor of two, uncomment.
    #mesh.Refine()

    # Return the grid.
    return mesh



