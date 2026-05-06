import ngsolve as ngs
import netgen.occ as occ
import numpy as np
from dream.mesh import get_chord_naca_4digit_series_coordinates
import matplotlib.pyplot as plt
from netgen.meshing import MeshingParameters

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def intersect_point_from_point_slope(p1, m1, p2, m2):
    """Intersection of two lines given points and slopes."""
    x1, y1 = p1
    x2, y2 = p2
    if np.isclose(m1, m2):
        return None  # parallel
    x_c = (y2 - y1 + m1 * x1 - m2 * x2) / (m1 - m2)
    y_c = y1 + m1 * (x_c - x1)
    return np.array([x_c, y_c])


def get_normal_slope(dx, dy):
    """Return (m_normal, m_tangent) given line components."""
    if abs(dx) < 1e-12:
        m_t = np.inf
    else:
        m_t = dy / dx
    if np.isinf(m_t):
        m_n = 0.0
    elif abs(m_t) < 1e-12:
        m_n = np.inf
    else:
        m_n = -1.0 / m_t
    return m_n, m_t


def circle_arc_points(center, radius, p1, p2, n_points=50, clockwise=False):
    """Return arc points between p1 and p2 along a circle."""
    cx, cy = center
    t1 = np.arctan2(p1[1] - cy, p1[0] - cx)
    t2 = np.arctan2(p2[1] - cy, p2[0] - cx)
    if clockwise and t2 > t1:
        t2 -= 2 * np.pi
    elif not clockwise and t2 < t1:
        t2 += 2 * np.pi
    theta = np.linspace(t1, t2, n_points)
    return np.column_stack([cx + radius * np.cos(theta),
                            cy + radius * np.sin(theta)])


def estimate_Cf_laminar(Rex) -> float:
    if Rex > 5.0e4:
        raise ValueError(f"Warning, Re={Rex} is not strictly laminar.")
    return 0.664/np.sqrt(Rex) # Blasius

def estimate_Cf_turbulent(Rex) -> float:
    if Rex < 5.0e4:
        raise ValueError(f"Warning, Re={Rex} is not strictly turbulent.")
    return 0.0592/Rex**(1/5) # Schlichting


def estimate_distance_for_dyplus(target_sim, n_poly, dyplus=1.0) -> float:

    Rex = target_sim.Re
    if Rex > 5.0e4:
        Cf = estimate_Cf_turbulent(Rex)
    else:
        Cf = estimate_Cf_laminar(Rex)
    
    # Get the required free-stream conditions.
    Uinf = target_sim.umag
    nu   = target_sim.nu

    # Calculate the skin-friction velocity.
    utau = Uinf * np.sqrt( Cf/2 )

    # Estimate the element resolution based on the wall-units.
    dy1 = dyplus * nu / utau
    
    # Roughly account for the DOFs inside the element.
    dy1 *= n_poly
    
    # Ensure the polynomial order is greater than 0.
    if n_poly < 1:
        raise ValueError(f"Polynomial order must be greater than 0.")

    # And voila, we're done.
    return dy1


# ---------------------------------------------------------------------
# Geometry construction helpers
# ---------------------------------------------------------------------

def round_trailing_edge(naca, tol=2e-2, n_arc=51):
    """
    Round the trailing edge of a NACA airfoil using a circular arc
    smoothly connecting upper and lower surfaces.
    """
    xy = np.asarray(naca)[:, :2]
    xmax = np.max(xy[:, 0])
    te_idx = np.where(np.abs(xy[:, 0] - xmax) < tol)[0]
    xy_te = xy[te_idx]

    # Separate upper/lower TE regions
    idx_upper = xy_te[:, 1] > 0
    idx_lower = ~idx_upper
    x_upper, y_upper = xy_te[idx_upper, 0], xy_te[idx_upper, 1]
    x_lower, y_lower = xy_te[idx_lower, 0], xy_te[idx_lower, 1]

    # Two points per side near TE
    xu, yu = x_upper[np.argsort(x_upper)[:2]], y_upper[np.argsort(x_upper)[:2]]
    xl, yl = x_lower[np.argsort(x_lower)[:2]], y_lower[np.argsort(x_lower)[:2]]

    # Slopes
    m_upper = get_normal_slope(xu[1] - xu[0], yu[1] - yu[0])
    m_lower = get_normal_slope(xl[1] - xl[0], yl[1] - yl[0])

    # Intersect normals to find circle center
    p1, p2 = (xu[0], yu[0]), (xl[0], yl[0])
    center = intersect_point_from_point_slope(p1, m_upper[0], p2, m_lower[0])
    if center is None:
        return xy  # fallback: nothing to round

    radius = np.linalg.norm(center - p1)
    arc_points = circle_arc_points(center, radius, p1, p2, n_points=n_arc, clockwise=True)

    # Replace TE region with rounded arc
    profile = np.delete(xy, te_idx, axis=0)
    profile = np.vstack((profile, arc_points))

    upper = profile[profile[:, 1] > 0]
    lower = profile[profile[:, 1] <= 0]
    upper = upper[np.argsort(-upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]
    naca_rounded = np.vstack((upper, lower))

    if not np.allclose(naca_rounded[0], naca_rounded[-1]):
        naca_rounded = np.vstack((naca_rounded, naca_rounded[0]))

    return naca_rounded


def extrude_airfoil_normals(points, offset):
    """Extrude a closed 2D airfoil outward along its local normals."""
    pts = np.asarray(points)
    if pts.shape[1] == 3:
        pts = pts[:, :2]
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    n = len(pts)
    tangents = np.zeros_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[-2]
    tangents[-1] = tangents[0]
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]

    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    # ensure outward normals (CCW expected)
    area = 0.5 * np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1])
    if area > 0:  # CW -> flip
        normals = -normals

    offset_pts = pts + offset * normals
    offset_pts[-1] = offset_pts[0]
    return offset_pts


def make_spline_wire(points):
    """Convert numpy array to OCC spline wire."""
    pts = np.asarray(points)
    if pts.shape[1] > 2:
        pts = pts[:, :2]
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    occ_points = [occ.gp_Pnt(float(x), float(y), 0.0) for x, y in pts]
    edge = occ.SplineApproximation(occ_points, continuity=occ.ShapeContinuity.C2)
    return occ.Wire(edge)


def smooth_stretch(layer_points, x0, x1, y0, y1, sx=None, sy=None, stretch_y=False):
    """
    Smoothly stretches/compresses the (x, y) coordinates of `layer_points`
    with C1 continuity.

    Parameters
    ----------
    layer_points : (N, 2) array_like
        Original coordinates (columns: x, y).
    x0, x1 : float
        Start and end of x-direction stretch.
    y0, y1 : float
        Center and target half-height for symmetric y stretch.
    sx, sy : float, optional
        Smoothness parameters (>=2) for x and y, default uses cubic Hermite.
    stretch_y : bool, optional
        If False, y coordinates are left unchanged.

    Returns
    -------
    stretched_points : (N, 2) ndarray
        Array of (xnew, ynew) coordinates after stretching.
    """

    pts = np.asarray(layer_points)
    x, y = pts[:, 0], pts[:, 1]

    # ---- X-direction stretch ----
    xmax = np.max(x)
    xi = np.clip((x - x0) / (xmax - x0 + 1e-12), 0.0, 1.0)

    if sx is None:
        wx = 3*xi**2 - 2*xi**3
    else:
        k = max(2.0, sx)
        wx = xi**k * (3 - 2*xi)

    xnew = np.where(
        x <= x0, x,
        np.where(x >= xmax, x1, x + (x1 - x) * wx)
    )

    # ---- Y-direction symmetric stretch ----
    if stretch_y:
        y_shifted = y - y0
        yabs = np.abs(y_shifted)
        ymax = np.max(yabs)

        eta = np.clip(yabs / (ymax + 1e-12), 0.0, 1.0)

        if sy is None:
            wy = 3*eta**2 - 2*eta**3
        else:
            k = max(2.0, sy)
            wy = eta**k * (3 - 2*eta)

        # Map magnitudes smoothly from [0, ymax] → [0, |y1 - y0|]
        ynew_abs = yabs + (abs(y1 - y0) - yabs) * wy

        # Restore sign and shift back by y0
        ynew = np.sign(y_shifted) * ynew_abs + y0
    else:
        ynew = y.copy()

    # ---- Return combined coordinates ----
    return np.column_stack((xnew, ynew))



def make_farfield_surface(radius, kind="circle", center=(0.5, 0), aspect_ratio=0.8):

    x0, y0 = center

    wp = occ.WorkPlane().MoveTo(x0, y0)

    if kind.lower() == "circle":
        farfield = wp.Circle(radius).Face()
    elif kind.lower() == "ellipse":
        rmajor = radius
        rminor = rmajor * aspect_ratio
        farfield = wp.Ellipse(rmajor, rminor).Face()
    else:
        raise ValueError(f"Unknown farfield kind '{kind}'. Expected 'circle' or 'ellipse'.")

    return farfield

def make_buffer_surface(radius, kind="circle", center=(0.5, 0), aspect_ratio=0.8):

    x0, y0 = center

    wp = occ.WorkPlane().MoveTo(x0, y0)

    if kind.lower() == "circle":
        buffer = wp.Circle(radius).Face()
    elif kind.lower() == "ellipse":
        rmajor = radius
        rminor = rmajor * aspect_ratio
        buffer = wp.Ellipse(rmajor, rminor).Face()
    else:
        raise ValueError(f"Unknown farfield kind '{kind}'. Expected 'circle' or 'ellipse'.")

    return buffer


def make_wake_surface(dx, dy, lx, ly, wake_maxh):

    # Add a wake region, to better resolve the vortex shedding.
    wake = occ.WorkPlane().MoveTo(dx, dy).RectangleC(lx, ly).Face()
    wake.maxh = wake_maxh
    return wake

def make_boundary_layers(naca_rounded, n_layers=5, dy0=0.015, r=1.8, maxh=0.2):
    
    # --- Compute layer offsets ---
    offsets = [dy0*sum((r ** i) for i in range(k)) for k in range(1,n_layers+1)]

    # --- Extrude layers ---
    layers_points = [extrude_airfoil_normals(naca_rounded, offset=dy) for dy in offsets]

    # --- Create OCC faces ---
    layers_faces = []
    for pts in layers_points:
        face = occ.Face(make_spline_wire(pts))
        face.maxh = maxh
        layers_faces.append(face)

    # Add the interface layer.
    face = occ.WorkPlane().MoveTo(0.5, 0).RectangleC(1.4, 0.5).Face()
    face.maxh = maxh
    layers_faces.append(face)
    
    # --- Glue all together ---
    glued_face = occ.Glue(layers_faces)
    
    return glued_face, layers_points



# ---------------------------------------------------------------------
# Main O-grid generator
# ---------------------------------------------------------------------

def get_airfoil_ogrid(
    target_sim,
    n_poly,
    naca_code="0012",
    n_points=800,
    inner_maxh=0.02,
    outer_maxh=0.2,
    target_dxplus=1.0,
    interface_maxh=0.05,
    wake_maxh=0.1,
    buffer_interface_maxh=0.5,
    grading=0.3,
    offset_distance=0.1,
    farfield_radius=5.0,
    farfield_center=(1.0,0.0),
    buffer_radius=4.0,
    target_dyplus=1.0,
    ratio_inflation=1.1,
    n_boundary_layers=5,
    inner_quad_dominated=False,
    outer_quad_dominated=False, 
    show_preview=False,
):
    """Generate an O-grid mesh around a NACA airfoil using NGSolve/Netgen."""
    naca = get_chord_naca_4digit_series_coordinates(naca_code, n=n_points, nodal_distribution='cosine')
    if not np.allclose(naca[0], naca[-1]):
        naca = np.vstack([naca, naca[0]])

    naca_rounded = round_trailing_edge(naca, tol=1e-3, n_arc=121)
    offset_curve = extrude_airfoil_normals(naca_rounded, offset=offset_distance)

    # --- Geometry construction ---
    airfoil_face = occ.Face(make_spline_wire(naca)); airfoil_face.name = "implicit"
    
    # Estimate first layer height.
    dy0 = estimate_distance_for_dyplus(target_sim, n_poly, dyplus=target_dyplus) 

    # Generate ogrid layers, until (and including) the interface.
    inflation_maxh = inner_maxh * (ratio_inflation ** n_boundary_layers)
    ogrid_faces, layer_points = make_boundary_layers(naca_rounded, n_layers=n_boundary_layers, dy0=dy0, r=ratio_inflation, maxh=inflation_maxh)

    # Display boundary layers.
    if show_preview:
        plt.figure()
        naca = np.asarray(naca)
        plt.plot(naca[:, 0], naca[:, 1], 'k-', label='Original')
        plt.plot(naca_rounded[:, 0], naca_rounded[:, 1], 'g-', label='Rounded')
        plt.plot(offset_curve[:, 0], offset_curve[:, 1], 'r--', label='Offset')
        
        for pts in layer_points:
            plt.plot(pts[:, 0], pts[:, 1], 'b', label='Layers')
        
        plt.axis('equal'); plt.legend(); plt.grid(True)
        plt.show()



    inner_domain = ogrid_faces - airfoil_face
    inner_domain.faces.name = "implicit"
    inner_domain.faces.maxh = inner_maxh


    farfield = make_farfield_surface(radius=farfield_radius,
                                     kind="circle",
                                     center=farfield_center)

    buffer_layer = make_buffer_surface(radius=buffer_radius,
                                 kind="circle",
                                 center=farfield_center)
    
    wake = make_wake_surface(dx=3.6, dy=0.0, lx=4.8, ly=0.5, wake_maxh=wake_maxh)
    wake_farfield = occ.Glue([wake, farfield])
    grid = occ.Glue([wake_farfield, buffer_layer])

    outer_domain = grid - ogrid_faces 
    outer_domain.faces[0].name = "explicit" # wake
    outer_domain.faces[1].name = "buffer"
    outer_domain.faces[2].name = "explicit" # surrounding region
   
    # To estimate the effective dxplus on the airfoil, we remove the 
    # estimated yplus resolution first, then use the wall-units.
    dx0 = (target_dxplus / target_dyplus) * dy0
    for edge in airfoil_face.edges:
        edge.name, edge.maxh = "airfoil", dx0

    outer_domain.edges[4].name = "farfield"
    outer_domain.edges[4].maxh = outer_maxh

    outer_domain.edges[6].name = "buffer_interface"
    outer_domain.edges[6].maxh = buffer_interface_maxh 

    outer_domain.edges[8].name = "wake_right"
    #outer_domain.edges[8].maxh = wake_maxh
    outer_domain.edges[9].name = "wake_top"
    #outer_domain.edges[8].maxh = wake__maxh
    outer_domain.edges[7].name = "wake_bottom"
    #outer_domain.edges[8].maxh = wake_maxh
    
    outer_domain.edges[3].name  = "interface" # imex interface right/wake_left
    outer_domain.edges[3].maxh  = interface_maxh
    outer_domain.edges[10].name = "interface" # imex interface top
    outer_domain.edges[10].maxh = interface_maxh
    outer_domain.edges[12].name = "interface" # imex interface bottom
    outer_domain.edges[12].maxh = interface_maxh
    outer_domain.edges[11].name = "interface" # imex interface left
    outer_domain.edges[11].maxh = interface_maxh

    # Report target wall-unit quantities.
    print( f"target (dx0, dy0): {dx0,dy0}", flush=True )

    # Ensure the leading edge point is in the mesh.
    le_pnt = occ.Vertex(occ.Pnt(0, 0, 0))
 
    inner_domain.faces.quad_dominated = inner_quad_dominated
    outer_domain.faces.quad_dominated = outer_quad_dominated
    #geo = occ.OCCGeometry(occ.Glue([inner_domain, outer_domain, le_pnt]), dim=2)
    geo = occ.OCCGeometry(occ.Glue([inner_domain, outer_domain]), dim=2)
    # Choose specific refinement points, often in the wake.    
    mp = MeshingParameters()
    #mp.RestrictH(x=-0.0045, y=0, z=0, h=0.002)
    mp.RestrictH(x=1.006, y=+0.007, z=0, h=dy0*0.5)
    mp.RestrictH(x=1.006, y=-0.007, z=0, h=dy0*0.5)
    mesh = ngs.Mesh(geo.GenerateMesh(mp=mp, grading=grading))

    implicit = ngs.Mesh(mesh.ngmesh.GetSubMesh(faces="implicit"))
    explicit = ngs.Mesh(mesh.ngmesh.GetSubMesh(faces="explicit|buffer"))
    implicit.Curve(3); explicit.Curve(3)
    return implicit, explicit







