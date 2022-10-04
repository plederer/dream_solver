from ngsolve import *
from netgen.geom2d import CSG2d, Circle, Rectangle, EdgeInfo as EI, PointInfo as PI, Solid2d


import math
from math import pi, atan2

def MakeSmoothRectangle (geo, p1, p2,r, bc=None, bcs=None, **args):
    p1x, p1y = p1
    p2x, p2y = p2
    p1x,p2x = min(p1x,p2x), max(p1x, p2x)
    p1y,p2y = min(p1y,p2y), max(p1y, p2y)

    if not bcs: bcs=4*[bc]

    pts = [geo.AppendPoint(*p) for p in [(p1x,p1y), (p1x+r,p1y),(p2x-r,p1y),
                                              (p2x,p1y), (p2x, p1y+r), (p2x, p2y-r),
                                              (p2x,p2y), (p2x-r, p2y), (p1x+r, p2y),
                                              (p1x,p2y), (p1x, p2y-r), (p1x, p1y+r)]]
    
    for p1,p2,bc in [(1,2,bcs[0]), (4,5, bcs[1]), (7, 8, bcs[2]), (10, 11, bcs[3])]:
        geo.Append( ["line", pts[p1], pts[p2]], bc=bc, **args)

    geo.Append( ["spline3", pts[11], pts[0], pts[1]],bc = bc, **args)
    geo.Append( ["spline3", pts[2], pts[3], pts[4]],bc = bc, **args)
    geo.Append( ["spline3", pts[5], pts[6], pts[7]],bc = bc, **args)
    geo.Append( ["spline3", pts[8], pts[9], pts[10]],bc = bc, **args)


def MakeRectangle(geo, p1, p2, p3, p4, bc=None, bcs=None, **args):
    # p1x, p1y = p1
    # p2x, p2y = p2
    # p1x,p2x = min(p1x,p2x), max(p1x, p2x)
    # p1y,p2y = min(p1y,p2y), max(p1y, p2y)

    if not bcs: bcs=4*[bc]

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
    geo.Append(["line", pts[2], pts[3]], bc="outflow") #, maxh=loc_maxh)



    
# z + n*b / (z - n*b) = (zeta + b / (zeta - b) ) ** n
# this gives
# z = kb * [ (zeta + b)**k + (zeta - b)**k] / [ (zeta + b)**k - (zeta - b)**k]
def profile(Mx,My,r,k,b,t, scale = 1):
    zeta = [r * cos(2*pi*t) + Mx,r * sin(2*pi*t) + My]
    
    zeta_p_b = [zeta[0] + b, zeta[1]]
    zeta_m_b = [zeta[0] - b, zeta[1]]

    Aphi_zeta_p_b = [ sqrt(zeta_p_b[0]**2 + zeta_p_b[1]**2), atan2( zeta_p_b[1] ,  zeta_p_b[0])]
    Aphi_zeta_m_b = [ sqrt(zeta_m_b[0]**2 + zeta_m_b[1]**2), atan2( zeta_m_b[1] ,  zeta_m_b[0])]

    h1 = [Aphi_zeta_p_b[0]**k, Aphi_zeta_p_b[1] * k] 
    h2 = [Aphi_zeta_m_b[0]**k, Aphi_zeta_m_b[1] * k]

    h3 = [h1[0] * cos(h1[1]), h1[0] * sin(h1[1])]
    h4 = [h2[0] * cos(h2[1]), h2[0] * sin(h2[1])]

    top = [h3[0] + h4[0], h3[1] + h4[1]]
    bott = [h3[0] - h4[0], h3[1] - h4[1]]
    z = [scale * k * b * (top[0] * bott[0] + top[1] * bott[1]) / (bott[0]**2 + bott[1]**2),
         scale * k * b * (top[1] * bott[0] - top[0] * bott[1]) / (bott[0]**2 + bott[1]**2)]
    
    return z[0],z[1]



def Make_C_type(geo, r, R, L, maxh_cyl):
    pts = [geo.AppendPoint(*p) for p in [(-R, R), (-R, 0),(-R, -R),
                                         (0, -R), (L,-R), (L, R),
                                         (0, R)]]
    
    geo.Append( ["spline3", pts[6], pts[0], pts[1]],bc="inflow")
    geo.Append( ["spline3", pts[1], pts[2], pts[3]],bc="inflow")

    geo.Append( ["line", pts[3], pts[4]], bc="outflow")
    geo.Append( ["line", pts[4], pts[5]], bc="outflow")
    geo.Append( ["line", pts[5], pts[6]], bc="outflow")

    geo.AddCircle ( (0, 0), r=r, leftdomain=0, rightdomain=1, bc="cyl", maxh=maxh_cyl)


def Make_Circle(geo, R, R_farfield, quadlayer=False, delta=1, HPREF = 1, loch = 1):

    ip = [(0, R), (-R, R), (-R, 0),(-R, -R),
          (0, -R), (R,-R), (R, 0), (R, R)]

    op = [(0, R_farfield), (-R_farfield, R_farfield), (-R_farfield, 0),(-R_farfield, -R_farfield),
                                         (0, -R_farfield), (R_farfield,-R_farfield), (R_farfield, 0),
                                         (R_farfield, R_farfield)]

    ps = op + ip
    rd = 1

    if quadlayer:
        mp = [(0, R+delta), (-R-delta, R+delta), (-R-delta, 0),(-R-delta, -R-delta),
          (0, -R-delta), (R+delta,-R-delta), (R+delta, 0), (R+delta, R+delta)]
        ps = ps + mp

    pts = [geo.AppendPoint(*p) for p in ps]
    
    geo.Append( ["spline3", pts[0], pts[1], pts[2]], leftdomain=1, rightdomain=0, bc="inflow")
    geo.Append( ["spline3", pts[2], pts[3], pts[4]], leftdomain=1, rightdomain=0, bc="inflow")
    geo.Append( ["spline3", pts[4], pts[5], pts[6]], leftdomain=1, rightdomain=0, bc="outflow")
    geo.Append( ["spline3", pts[6], pts[7], pts[0]], leftdomain=1, rightdomain=0, bc="outflow")

    if quadlayer:
        rd = 2
    c1 = geo.Append (["spline3", pts[8],pts[9],pts[10]], leftdomain=0, rightdomain=rd, bc="cyl", hpref = HPREF, maxh=loch)
    c2 = geo.Append (["spline3", pts[10],pts[11],pts[12]], leftdomain=0, rightdomain=rd, bc="cyl", hpref = HPREF, maxh=loch)
    c3 = geo.Append (["spline3", pts[12],pts[13],pts[14]], leftdomain=0, rightdomain=rd, bc="cyl", hpref = HPREF, maxh=loch)
    c4 = geo.Append (["spline3", pts[14],pts[15],pts[8]], leftdomain=0, rightdomain=rd, bc="cyl", hpref = HPREF, maxh=loch)

    if quadlayer:
        geo.Append (["spline3", pts[16],pts[17],pts[18]], leftdomain=2, rightdomain=1, bc="cyl2", copy = c1)
        geo.Append (["spline3", pts[18],pts[19],pts[20]], leftdomain=2, rightdomain=1, bc="cyl2", copy = c2)
        geo.Append (["spline3", pts[20],pts[21],pts[22]], leftdomain=2, rightdomain=1, bc="cyl2", copy = c3)
        geo.Append (["spline3", pts[22],pts[23],pts[16]], leftdomain=2, rightdomain=1, bc="cyl2", copy = c4)

        # geo.SetDomainQuadMeshing(2,True)


def Make_Circle_Channel(geo, R, R_farfield, R_channel, maxh, maxh_cyl, maxh_channel):
    cyl = Solid2d( [(0, -1),
                          EI(( 1,  -1), bc="cyl", maxh=maxh_cyl), # control point for quadratic spline
                          (1,0),
                          EI(( 1,  1), bc="cyl", maxh=maxh_cyl), # spline with maxh
                          (0,1),
                          EI((-1,  1), bc="cyl", maxh=maxh_cyl),
                          (-1,0),
                          EI((-1, -1), bc="cyl", maxh=maxh_cyl), # spline with bc
                          ])
    cyl_layer = cyl.Copy().Scale(R+2*maxh_cyl)
    cyl.Scale(R)
    

    circle_FF = Solid2d( [(0, -1),
                          EI(( 1,  -1), bc="outflow"), # control point for quadratic spline
                          (1,0),
                          EI(( 1,  1), bc="outflow"), # spline with maxh
                          (0,1),
                          EI((-1,  1), bc="inflow"),
                          (-1,0),
                          EI((-1, -1), bc="inflow"), # spline with bc
                          ])
    circle_FF.Scale(R_farfield)

    cyl_2 = Circle( center=(0,0), radius=R_channel)
    rect = Rectangle( pmin=(0,-R_channel), pmax=(R_farfield + 1,R_channel))                 
    
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
