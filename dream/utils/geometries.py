from __future__ import annotations
from ngsolve import *
from netgen.geom2d import CSG2d, Circle, Rectangle, EdgeInfo as EI, PointInfo as PI, Solid2d
from netgen.occ import WorkPlane, OCCGeometry, Glue
from dataclasses import dataclass, field
from netgen.meshing import IdentificationType
from math import pi, atan2
from enum import Enum
from ..region import BufferCoordinate


class BNDS:

    @dataclass
    class BND:
        DEFAULT: str
        NAME: str = None

        def __post_init__(self):
            if self.NAME is None:
                self.NAME = self.DEFAULT

        @property
        def is_default(self) -> bool:
            return self.NAME == self.DEFAULT

    @property
    def names(self):
        return tuple(bnd.NAME for bnd in self)

    @property
    def defaults(self):
        return tuple(bnd.DEFAULT for bnd in self)

    def __iter__(self) -> BND:
        for bnd in vars(self).values():
            yield bnd


class Domain:

    def __init__(self,
                 mat: str = "default",
                 loc: list[float] = None,
                 maxh: float = 0.5,
                 main: tuple[float, float] = (0, 0, 0)
                 ) -> None:

        self.mat = mat
        self.loc = list(loc)
        self.maxh = maxh
        self.main = tuple(main)


class RectangleDomain(Domain):

    @dataclass
    class RectangleBNDS(BNDS):
        bottom: BNDS.BND = field(default_factory=lambda: BNDS.BND("bottom"))
        right: BNDS.BND = field(default_factory=lambda: BNDS.BND("right"))
        top: BNDS.BND = field(default_factory=lambda: BNDS.BND("top"))
        left: BNDS.BND = field(default_factory=lambda: BNDS.BND("left"))
        bottom_inner: BNDS.BND = field(default_factory=lambda: BNDS.BND("bottom"))
        right_inner: BNDS.BND = field(default_factory=lambda: BNDS.BND("right"))
        top_inner: BNDS.BND = field(default_factory=lambda: BNDS.BND("top"))
        left_inner: BNDS.BND = field(default_factory=lambda: BNDS.BND("left"))

        @property
        def outer(self) -> tuple[BNDS.BND]:
            return (self.bottom, self.right, self.top, self.left)

        @property
        def inner(self) -> tuple[BNDS.BND]:
            return (self.bottom_inner, self.right_inner, self.top_inner, self.left_inner)

    def __init__(self,
                 W: float = 1,
                 H: float = 1,
                 Wi: float = 0,
                 Hi: float = 0,
                 mat: str = "default",
                 bnds=None,
                 loc: tuple[float, float] = (0.0, 0.0),
                 maxh: float = 0.5) -> None:
        self.W = W
        self.H = H

        self.Wi_ring = []
        self.Hi_ring = []

        if Wi != 0 and Hi == 0:
            Hi = Wi
        elif Hi != 0 and Wi == 0:
            Wi = Hi

        self.Wi = Wi
        self.Hi = Hi

        if bnds is None:
            self.bnds = self.RectangleBNDS()

        super().__init__(mat, loc, maxh, main=loc)

    @property
    def x_(self) -> BufferCoordinate:

        if self.Wi == 0 and self.Hi == 0:
            if self.loc[0] >= self.main[0]:
                start = self.loc[0] - self.W/2
                end = self.loc[0] + self.W/2
            elif self.loc[0] < self.main[0]:
                start = self.loc[0] + self.W/2
                end = self.loc[0] - self.W/2
        else:
            start = self.loc[0] + self.Wi/2
            end = self.loc[0] + self.W/2

        return BufferCoordinate.x(start, end, offset=self.loc[0])

    @property
    def y_(self) -> BufferCoordinate:

        if self.Wi == 0 and self.Hi == 0:
            if self.loc[1] >= self.main[1]:
                start = self.loc[1] - self.H/2
                end = self.loc[1] + self.H/2
            elif self.loc[1] < self.main[1]:
                start = self.loc[1] + self.H/2
                end = self.loc[1] - self.H/2
        else:
            start = self.loc[1] + self.Hi/2
            end = self.loc[1] + self.H/2

        return BufferCoordinate.y(start, end, offset=self.loc[1])

    @property
    def xy_(self) -> BufferCoordinate:

        x, y = self.x_, self.y_
        start = (x.start, y.start)
        end = (x.end, y.end)
        offset = (x.offset, y.offset)

        return BufferCoordinate.xy(start, end, offset=offset)

    def get_face(self, wp=None):
        if wp is None:
            wp = WorkPlane()

        wp.MoveTo(self.loc[0], self.loc[1])
        rectangle = wp.RectangleC(self.W, self.H).Face()
        rectangle.name = self.mat

        for edge, name in zip(rectangle.edges, self.bnds.outer):
            edge.name = name.NAME

        if self.Wi > 0 and self.Hi > 0:
            size_W = (self.W - self.Wi)/2
            size_H = (self.H - self.Hi)/2
            W_p, W_m = self.loc[0] + (self.Wi + self.W)/4, self.loc[0] - (self.Wi + self.W)/4
            H_p, H_m = self.loc[1] + (self.Hi + self.H)/4, self.loc[1] - (self.Hi + self.H)/4

            faces = [
                wp.MoveTo(self.loc[0], H_m).RectangleC(self.W, size_H).Face(),
                wp.MoveTo(W_p, self.loc[1]).RectangleC(size_W, self.H).Face(),
                wp.MoveTo(self.loc[0], H_p).RectangleC(self.W, size_H).Face(),
                wp.MoveTo(W_m, self.loc[1]).RectangleC(size_W, self.H).Face()
            ]

            boundaries = [
                (self.bnds.bottom, self.bnds.right, self.bnds.top_inner, self.bnds.left),
                (self.bnds.bottom, self.bnds.right, self.bnds.top, self.bnds.left_inner),
                (self.bnds.bottom_inner, self.bnds.right, self.bnds.top, self.bnds.left),
                (self.bnds.bottom, self.bnds.right_inner, self.bnds.top, self.bnds.left)
            ]

            for face, bnds in zip(faces, boundaries):
                for edge, name in zip(face.edges, bnds):
                    edge.name = name.NAME
                face.name = self.mat

            rectangle = Glue(faces)

            W_r, W_l = self.loc[0] + self.Wi/2,  self.loc[0] - self.Wi/2
            H_t, H_b = self.loc[1] + self.Hi/2,  self.loc[1] - self.Hi/2

            default_edges = [
                (W_p, H_t),  (W_p, H_b),
                (W_m, H_t),  (W_m, H_b),
                (W_r, H_p),  (W_l, H_p),
                (W_r, H_m),  (W_l, H_m),
            ]
            for edge in default_edges:
                rectangle.edges.Nearest(edge).name = "default"

            if self.Wi_ring and self.Hi_ring:

                H_points = [(W_p, self.loc[1]), (W_m, self.loc[1])]
                W_points = [(self.loc[0], H_m), (self.loc[0], H_p)]

                subfaces = []
                for point in H_points:
                    face = rectangle.faces.Nearest(point)
                    wp.MoveTo(*point)
                    for Hi in self.Hi_ring[-2::-1]:
                        cut = wp.RectangleC(size_W, Hi).Face()
                        face = Glue([face, cut])
                    subfaces.append(face)

                for point in W_points:
                    face = rectangle.faces.Nearest(point)
                    wp.MoveTo(*point)
                    for Wi in self.Wi_ring[-2::-1]:
                        cut = wp.RectangleC(Wi, size_H).Face()
                        face = Glue([face, cut])
                    subfaces.append(face)

                corners = ((W_p, H_p), (W_p, H_m), (W_m, H_p), (W_m, H_m))
                corners = [rectangle.faces.Nearest(corner) for corner in corners]

                rectangle = Glue(corners + subfaces)
                for face in rectangle.faces:
                    face.name = self.mat

        rectangle.maxh = self.maxh

        return rectangle


class CircularDomain(Domain):

    @dataclass
    class CircularBNDS(BNDS):
        outer_I: BNDS.BND = field(default_factory=lambda: BNDS.BND("outer"))
        outer_II: BNDS.BND = field(default_factory=lambda: BNDS.BND("outer"))
        outer_III: BNDS.BND = field(default_factory=lambda: BNDS.BND("outer"))
        outer_IV: BNDS.BND = field(default_factory=lambda: BNDS.BND("outer"))
        inner_I: BNDS.BND = field(default_factory=lambda: BNDS.BND("inner"))
        inner_II: BNDS.BND = field(default_factory=lambda: BNDS.BND("inner"))
        inner_III: BNDS.BND = field(default_factory=lambda: BNDS.BND("inner"))
        inner_IV: BNDS.BND = field(default_factory=lambda: BNDS.BND("inner"))

        @property
        def outer(self) -> tuple[BNDS.BND]:
            return (self.outer_I, self.outer_II, self.outer_III, self.outer_IV)

        @property
        def inner(self) -> tuple[BNDS.BND]:
            return (self.inner_I, self.inner_II, self.inner_III, self.inner_IV)

    def __init__(self,
                 R: float = 1,
                 Ri: float = 0,
                 mat: str = "default",
                 bnds=None,
                 loc: tuple[float, float] = (0.0, 0.0),
                 maxh: float = 0.5) -> None:
        self.R = R
        self.Ri = Ri

        if bnds is None:
            self.bnds = self.CircularBNDS()
        super().__init__(mat, loc, maxh, main=loc)

    @property
    def r_(self):
        return BufferCoordinate.polar(self.Ri, self.R, self.loc)

    def get_face(self, wp=None):
        if wp is None:
            wp = WorkPlane()
        wp.Direction(0, 1)
        wp.MoveTo(self.loc[0] + self.R, self.loc[1])
        wp.Arc(r=self.R, ang=90)
        wp.Arc(r=self.R, ang=90)
        wp.Arc(r=self.R, ang=90)
        wp.Arc(r=self.R, ang=90)

        outer = wp.Face()
        outer.name = self.mat
        for edge, bc in zip(outer.edges, self.bnds.outer):
            edge.name = bc.NAME

        if self.Ri > 0:
            wp.Direction(0, 1)
            wp.MoveTo(self.loc[0] + self.Ri, self.loc[1])
            wp.Arc(r=self.Ri, ang=90)
            wp.Arc(r=self.Ri, ang=90)
            wp.Arc(r=self.Ri, ang=90)
            wp.Arc(r=self.Ri, ang=90)

            inner = wp.Face()
            inner.name = self.mat

            for edge, bc in zip(inner.edges, self.bnds.inner):
                edge.name = bc.NAME

            outer = outer - inner

        outer.maxh = self.maxh

        return outer


class Grid:

    class Direction(Enum):
        X = "x"
        Y = "y"
        Z = "z"
        R = "r"

    def __init__(self, center: Domain) -> None:
        self._center = center
        self.periodic = []

    def add_periodic(self, master: str, minion: str):
        self.periodic.append((master, minion))

    def _add_domain(self, domain: RectangleDomain, direction: list):
        if domain.mat == "default":
            domain.mat = self._center.mat

        domain.main = self._center.main

        direction.append(domain)

    def _swap_boundaries(self, first: BNDS.BND, last: BNDS.BND, mat: str):
        if first.is_default and last.is_default:
            first.NAME = mat + "_facet"
            last.NAME = mat + "_facet"
        elif not last.is_default:
            first.NAME = last.NAME
        elif not first.is_default:
            last.NAME = first.NAME
        else:
            raise ValueError(f"Can not decide between: {first.NAME} and {last.NAME}")


class Rectangle1DGrid(Grid):

    def __init__(self, center: RectangleDomain, direction="x") -> None:
        if not isinstance(center, RectangleDomain):
            raise TypeError()
        self.dir = self.Direction(direction.lower())
        self.front = []
        self.back = []
        super().__init__(center)

    @property
    def center(self) -> RectangleDomain:
        return self._center

    def add_front(self, domain: RectangleDomain):

        swap = self.center
        if self.front:
            swap = self.front[-1]

        if self.dir is self.Direction.X:
            domain.H = self.center.H
            domain.loc[1] = self.center.loc[1]
            domain.loc[0] = swap.loc[0] + (swap.W + domain.W)/2
            self._swap_boundaries(swap.bnds.right, domain.bnds.left, swap.mat)

        elif self.dir is self.Direction.Y:
            domain.W = self.center.W
            domain.loc[0] = self.center.loc[0]
            domain.loc[1] = swap.loc[1] + (swap.H + domain.H)/2
            self._swap_boundaries(swap.bnds.top, domain.bnds.bottom, swap.mat)
        self._add_domain(domain, self.front)

    def add_back(self, domain: RectangleDomain):
        swap = self.center
        if self.back:
            swap = self.back[-1]

        if self.dir is self.Direction.X:
            domain.H = self.center.H
            domain.loc[1] = self.center.loc[1]
            domain.loc[0] = swap.loc[0] - (swap.W + domain.W)/2
            self._swap_boundaries(swap.bnds.left, domain.bnds.right, swap.mat)

        elif self.dir is self.Direction.Y:
            domain.W = self.center.W
            domain.loc[0] = self.center.loc[0]
            domain.loc[1] = swap.loc[1] - (swap.H + domain.H)/2
            self._swap_boundaries(swap.bnds.bottom, domain.bnds.top, swap.mat)

        self._add_domain(domain, self.back)

    def get_face(self, wp=None):
        if wp is None:
            wp = WorkPlane()

        rectangle = self.center.get_face(wp)
        front = [domain.get_face(wp) for domain in self.front]
        back = [domain.get_face(wp) for domain in self.back]

        rectangle = Glue([rectangle] + front + back)

        for bnds in self.periodic:
            master = [edge for edge in rectangle.edges if edge.name == bnds[0]]
            minion = [edge for edge in rectangle.edges if edge.name == bnds[1]]

            for b, t in zip(master, minion):
                b.Identify(t, f"{bnds[0]}_{bnds[1]}_periodic", IdentificationType.PERIODIC)

        return rectangle


class CircularGrid(Grid):

    def __init__(self, center: CircularDomain) -> None:
        if not isinstance(center, CircularDomain):
            raise TypeError()
        self.dir = self.Direction('r')
        self.rings = []
        super().__init__(center)

    @property
    def center(self) -> CircularDomain:
        return self._center

    def add_ring(self, domain: CircularDomain):
        domain.loc = self.center.loc

        swap = self.center
        if self.rings:
            swap = self.rings[-1]

        domain.Ri = swap.R
        domain.R += domain.Ri

        for first, last in zip(swap.bnds.outer, domain.bnds.inner):
            self._swap_boundaries(first, last, swap.mat)

        self._add_domain(domain, self.rings)

    def get_face(self, wp=None):
        if wp is None:
            wp = WorkPlane()

        circle = [self.center.get_face(wp)]
        rings = [domain.get_face(wp) for domain in self.rings]

        circle = Glue(circle + rings)

        for bnds in self.periodic:
            master = [edge for edge in circle.edges if edge.name == bnds[0]]
            minion = [edge for edge in circle.edges if edge.name == bnds[1]]

            for b, t in zip(master, minion):
                b.Identify(t, "periodic", IdentificationType.PERIODIC)

        return circle


class RectangleGrid(Grid):

    def __init__(self, center: RectangleDomain) -> None:
        if not isinstance(center, RectangleDomain):
            raise TypeError()
        self.rings = []
        super().__init__(center)

    @property
    def center(self) -> RectangleDomain:
        return self._center

    def add_ring(self, domain: RectangleDomain):
        domain.loc = self.center.loc

        swap = self.center
        if self.rings:
            swap = self.rings[-1]

        domain.Wi = swap.W
        domain.Hi = swap.H
        domain.W += domain.Wi
        domain.H += domain.Hi

        domain.Wi_ring.extend(swap.Wi_ring + [domain.Wi])
        domain.Hi_ring.extend(swap.Hi_ring + [domain.Hi])

        for first, last in zip(swap.bnds.outer, domain.bnds.inner):
            self._swap_boundaries(first, last, swap.mat)

        self._add_domain(domain, self.rings)

    def get_face(self, wp=None):
        if wp is None:
            wp = WorkPlane()

        rectangle = [self.center.get_face(wp)]
        rings = [domain.get_face(wp) for domain in self.rings]

        rectangle = Glue(rectangle + rings)

        for bnds in self.periodic:
            master = [edge for edge in rectangle.edges if edge.name == bnds[0]]
            minion = [edge for edge in rectangle.edges if edge.name == bnds[1]]

            for b, t in zip(master, minion):
                b.Identify(t, "periodic", IdentificationType.PERIODIC)

        return rectangle


def MakeSmoothRectangle(geo, p1, p2, r, bc=None, bcs=None, **args):
    p1x, p1y = p1
    p2x, p2y = p2
    p1x, p2x = min(p1x, p2x), max(p1x, p2x)
    p1y, p2y = min(p1y, p2y), max(p1y, p2y)

    if not bcs:
        bcs = 4*[bc]

    pts = [geo.AppendPoint(*p) for p in [(p1x, p1y), (p1x+r, p1y), (p2x-r, p1y),
                                         (p2x, p1y), (p2x, p1y+r), (p2x, p2y-r),
                                         (p2x, p2y), (p2x-r, p2y), (p1x+r, p2y),
                                         (p1x, p2y), (p1x, p2y-r), (p1x, p1y+r)]]

    for p1, p2, bc in [(1, 2, bcs[0]), (4, 5, bcs[1]), (7, 8, bcs[2]), (10, 11, bcs[3])]:
        geo.Append(["line", pts[p1], pts[p2]], bc=bc, **args)

    geo.Append(["spline3", pts[11], pts[0], pts[1]], bc=bc, **args)
    geo.Append(["spline3", pts[2], pts[3], pts[4]], bc=bc, **args)
    geo.Append(["spline3", pts[5], pts[6], pts[7]], bc=bc, **args)
    geo.Append(["spline3", pts[8], pts[9], pts[10]], bc=bc, **args)


def MakeOCCRectangle(p1, p2, bottom="bottom", right="right", top="top", left="left") -> OCCGeometry:

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    wp = WorkPlane()
    wp.MoveTo(*p1)
    wp.Line(dx, name=bottom).Rotate(90)
    wp.Line(dy, name=right).Rotate(90)
    wp.Line(dx, name=top).Rotate(90)
    wp.Line(dy, name=left).Rotate(90)

    geo = OCCGeometry(wp.Face(), dim=2)

    return geo


def MakeOCCCircle(center, radius, br="right", tr="right", tl="left", bl="left") -> OCCGeometry:

    dx, dy = center

    wp = WorkPlane()
    wp.MoveTo(dx, dy-radius)
    wp.Arc(r=radius, ang=90)
    wp.Arc(r=radius, ang=90)
    wp.Arc(r=radius, ang=90)
    wp.Arc(r=radius, ang=90)
    face = wp.Face()
    face.edges[0].name = br
    face.edges[1].name = tr
    face.edges[2].name = tl
    face.edges[3].name = bl

    geo = OCCGeometry(face, dim=2)

    return geo


def MakeOCCCirclePlane(center, radius, bottom="bottom", right="right", top="top", left="left") -> OCCGeometry:

    dx, dy = center

    wp = WorkPlane()
    wp.MoveTo(dx, dy+radius).Rotate(180)
    wp.Arc(r=radius, ang=180)
    wp.Line(radius, name=bottom).Rotate(90)
    wp.Line(2*radius, name=right).Rotate(90)
    wp.Line(radius, name=top).Rotate(90)
    face = wp.Face()
    face.edges[0].name = left

    geo = OCCGeometry(face, dim=2)

    return geo


def MakeRectangle(geo, p1, p2, p3, p4, bc=None, bcs=None, **args):
    # p1x, p1y = p1
    # p2x, p2y = p2
    # p1x,p2x = min(p1x,p2x), max(p1x, p2x)
    # p1y,p2y = min(p1y,p2y), max(p1y, p2y)

    if not bcs:
        bcs = 4*[bc]

    pts = [geo.AppendPoint(*p) for p in [p1, p2, p3, p4]]

    for p1, p2, bc in [(0, 1, bcs[0]), (1, 2, bcs[1]), (2, 3, bcs[2]), (3, 0, bcs[3])]:
        geo.Append(["line", pts[p1], pts[p2]], bc=bc, **args)


def MakePlate(geo, L=1, loc_maxh=0.01):

    ps = [(0, 0), (L/2, 0)]
    pts = [geo.AppendPoint(*p) for p in ps]

    pts += [geo.AppendPoint(3*L, 0, maxh=loc_maxh)]
    ps = [(3*L, L/2), (0, L/2)]
    pts += [geo.AppendPoint(*p) for p in ps]

    for p1, p2, bc in [(0, 1, "sym"), (3, 4, "top"), (4, 0, "inflow")]:
        geo.Append(["line", pts[p1], pts[p2]], bc=bc)
    # , (2, 3, "outflow")
    geo.Append(["line", pts[1], pts[2]], bc="ad_wall", maxh=loc_maxh)
    geo.Append(["line", pts[2], pts[3]], bc="outflow")  # , maxh=loc_maxh)


# z + n*b / (z - n*b) = (zeta + b / (zeta - b) ) ** n
# this gives
# z = kb * [ (zeta + b)**k + (zeta - b)**k] / [ (zeta + b)**k - (zeta - b)**k]
def profile(Mx, My, r, k, b, t, scale=1):
    zeta = [r * cos(2*pi*t) + Mx, r * sin(2*pi*t) + My]

    zeta_p_b = [zeta[0] + b, zeta[1]]
    zeta_m_b = [zeta[0] - b, zeta[1]]

    Aphi_zeta_p_b = [sqrt(zeta_p_b[0]**2 + zeta_p_b[1]**2), atan2(zeta_p_b[1],  zeta_p_b[0])]
    Aphi_zeta_m_b = [sqrt(zeta_m_b[0]**2 + zeta_m_b[1]**2), atan2(zeta_m_b[1],  zeta_m_b[0])]

    h1 = [Aphi_zeta_p_b[0]**k, Aphi_zeta_p_b[1] * k]
    h2 = [Aphi_zeta_m_b[0]**k, Aphi_zeta_m_b[1] * k]

    h3 = [h1[0] * cos(h1[1]), h1[0] * sin(h1[1])]
    h4 = [h2[0] * cos(h2[1]), h2[0] * sin(h2[1])]

    top = [h3[0] + h4[0], h3[1] + h4[1]]
    bott = [h3[0] - h4[0], h3[1] - h4[1]]
    z = [scale * k * b * (top[0] * bott[0] + top[1] * bott[1]) / (bott[0]**2 + bott[1]**2),
         scale * k * b * (top[1] * bott[0] - top[0] * bott[1]) / (bott[0]**2 + bott[1]**2)]

    return z[0], z[1]


def Make_C_type(geo, r, R, L, maxh_cyl):
    pts = [geo.AppendPoint(*p) for p in [(-R, R), (-R, 0), (-R, -R),
                                         (0, -R), (L, -R), (L, R),
                                         (0, R)]]

    geo.Append(["spline3", pts[6], pts[0], pts[1]], bc="inflow")
    geo.Append(["spline3", pts[1], pts[2], pts[3]], bc="inflow")

    geo.Append(["line", pts[3], pts[4]], bc="outflow")
    geo.Append(["line", pts[4], pts[5]], bc="outflow")
    geo.Append(["line", pts[5], pts[6]], bc="outflow")

    geo.AddCircle((0, 0), r=r, leftdomain=0, rightdomain=1, bc="cyl", maxh=maxh_cyl)


def Make_Circle(geo, R, R_farfield, quadlayer=False, delta=1, HPREF=1, loch=1):

    ip = [(0, R), (-R, R), (-R, 0), (-R, -R),
          (0, -R), (R, -R), (R, 0), (R, R)]

    op = [(0, R_farfield), (-R_farfield, R_farfield), (-R_farfield, 0), (-R_farfield, -R_farfield),
          (0, -R_farfield), (R_farfield, -R_farfield), (R_farfield, 0),
          (R_farfield, R_farfield)]

    ps = op + ip
    rd = 1

    if quadlayer:
        mp = [(0, R+delta), (-R-delta, R+delta), (-R-delta, 0), (-R-delta, -R-delta),
              (0, -R-delta), (R+delta, -R-delta), (R+delta, 0), (R+delta, R+delta)]
        ps = ps + mp

    pts = [geo.AppendPoint(*p) for p in ps]

    geo.Append(["spline3", pts[0], pts[1], pts[2]], leftdomain=1, rightdomain=0, bc="inflow")
    geo.Append(["spline3", pts[2], pts[3], pts[4]], leftdomain=1, rightdomain=0, bc="inflow")
    geo.Append(["spline3", pts[4], pts[5], pts[6]], leftdomain=1, rightdomain=0, bc="outflow")
    geo.Append(["spline3", pts[6], pts[7], pts[0]], leftdomain=1, rightdomain=0, bc="outflow")

    if quadlayer:
        rd = 2
    c1 = geo.Append(["spline3", pts[8], pts[9], pts[10]], leftdomain=0,
                    rightdomain=rd, bc="cyl", hpref=HPREF, maxh=loch)
    c2 = geo.Append(["spline3", pts[10], pts[11], pts[12]], leftdomain=0,
                    rightdomain=rd, bc="cyl", hpref=HPREF, maxh=loch)
    c3 = geo.Append(["spline3", pts[12], pts[13], pts[14]], leftdomain=0,
                    rightdomain=rd, bc="cyl", hpref=HPREF, maxh=loch)
    c4 = geo.Append(["spline3", pts[14], pts[15], pts[8]], leftdomain=0,
                    rightdomain=rd, bc="cyl", hpref=HPREF, maxh=loch)

    if quadlayer:
        geo.Append(["spline3", pts[16], pts[17], pts[18]], leftdomain=2, rightdomain=1, bc="cyl2", copy=c1)
        geo.Append(["spline3", pts[18], pts[19], pts[20]], leftdomain=2, rightdomain=1, bc="cyl2", copy=c2)
        geo.Append(["spline3", pts[20], pts[21], pts[22]], leftdomain=2, rightdomain=1, bc="cyl2", copy=c3)
        geo.Append(["spline3", pts[22], pts[23], pts[16]], leftdomain=2, rightdomain=1, bc="cyl2", copy=c4)

        # geo.SetDomainQuadMeshing(2,True)


def Make_Circle_Channel(geo, R, R_farfield, R_channel, maxh, maxh_cyl, maxh_channel):
    if R+2*maxh_cyl > R_channel:
        raise Exception("reduce maxh_cyl")
    cyl = Solid2d([(0, -1),
                   EI((1,  -1), bc="cyl", maxh=maxh_cyl),  # control point for quadratic spline
                   (1, 0),
                   EI((1,  1), bc="cyl", maxh=maxh_cyl),  # spline with maxh
                   (0, 1),
                   EI((-1,  1), bc="cyl", maxh=maxh_cyl),
                   (-1, 0),
                   EI((-1, -1), bc="cyl", maxh=maxh_cyl),  # spline with bc
                   ])

    cyl_layer = Circle(center=(0, 0), radius=R+2*maxh_cyl, bc="inner")
    cyl.Scale(R)

    circle_FF = Solid2d([(0, -1),
                         EI((1,  -1), bc="outflow"),  # control point for quadratic spline
                         (1, 0),
                         EI((1,  1), bc="outflow"),  # spline with maxh
                         (0, 1),
                         EI((-1,  1), bc="inflow"),
                         (-1, 0),
                         EI((-1, -1), bc="inflow"),  # spline with bc
                         ])
    circle_FF.Scale(R_farfield)

    cyl_2 = Circle(center=(0, 0), radius=R_channel, bc="inner")
    rect = Rectangle(pmin=(0, -R_channel), pmax=(R_farfield + 1, R_channel), bc="inner")

    layer = cyl_layer - cyl
    layer.Maxh(maxh_cyl)

    dom1 = (cyl_2 + rect)
    channel = dom1 * circle_FF - cyl_layer
    channel.Maxh(maxh_channel)

    outer = (circle_FF - rect) - cyl_2
    outer.Maxh(maxh)
    geo.Add(outer)
    geo.Add(channel)
    geo.Add(layer)


def Make_HalfCircle_Channel(geo, R, R_farfield, R_channel, maxh, maxh_cyl, maxh_channel):
    if R+2*maxh_cyl > R_channel:
        raise Exception("reduce maxh_cyl")
    cyl = Solid2d([(0, -1),
                   EI((1,  -1), bc="cyl", maxh=maxh_cyl),  # control point for quadratic spline
                   (1, 0),
                   EI((1,  1), bc="cyl", maxh=maxh_cyl),  # spline with maxh
                   (0, 1),
                   EI((-1,  1), bc="cyl", maxh=maxh_cyl),
                   (-1, 0),
                   EI((-1, -1), bc="cyl", maxh=maxh_cyl),  # spline with bc
                   ])

    cyl_layer = Circle(center=(0, 0), radius=R+2*maxh_cyl, bc="inner")
    cyl.Scale(R)

    circle_FF_1 = Solid2d([(0, -1),
                          EI((1,  -1), bc="outflow"),  # control point for quadratic spline
                          (1, 0),
                          EI((1,  1), bc="outflow"),  # spline with maxh
                          (0, 1),
                           EI((-1,  1), bc="inflow"),
                           (-1, 0),
                           EI((-1, -1), bc="inflow"),  # spline with bc
                           ])
    circle_FF_1.Scale(R_farfield)

    # rect_FF = Rectangle( pmin=(0,-R_farfield), pmax=(2 * R_farfield, R_farfield), bc = "outflow")
    rect_FF = Solid2d(
        [(0, -R_farfield),
         (2 * R_farfield, -R_farfield - 2),
         (2 * R_farfield, R_farfield + 2),
         (0, R_farfield)],
        bc="outflow")

    circle_FF = circle_FF_1 + rect_FF

    cyl_2 = Circle(center=(0, 0), radius=R_channel, bc="inner")
    rect = Rectangle(pmin=(0, -R_channel), pmax=(R_farfield + 1, R_channel), bc="inner")

    layer = cyl_layer - cyl
    layer.Maxh(maxh_cyl)

    dom1 = (cyl_2 + rect)
    channel = dom1 * circle_FF - cyl_layer
    channel.Maxh(maxh_channel)

    outer = (circle_FF - rect) - cyl_2
    outer.Maxh(maxh)
    geo.Add(outer)
    geo.Add(channel)
    geo.Add(layer)


def MakeCircle(geo, R_farfield, addrect=False):
    circle_FF_1 = Solid2d([(0, -1),
                          EI((1,  -1), bc="outflow"),  # control point for quadratic spline
                          (1, 0),
                          EI((1,  1), bc="outflow"),  # spline with maxh
                          (0, 1),
                           EI((-1,  1), bc="inflow"),
                           (-1, 0),
                           EI((-1, -1), bc="inflow"),  # spline with bc
                           ])
    circle_FF_1.Scale(R_farfield)
    if addrect:
        L_rect = 10
        rect_FF = Solid2d(
            [(0, -R_farfield),
             (L_rect, -R_farfield),
             (L_rect, R_farfield),
             (0, R_farfield)],
            bc="outflow")
        circle_FF = circle_FF_1 + rect_FF
    else:
        circle_FF = circle_FF_1

    geo.Add(circle_FF)
