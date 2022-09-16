from ngsolve import *

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


def Make_Circle(geo, R):
    pts = [geo.AppendPoint(*p) for p in [(0, R), (-R, R), (-R, 0),(-R, -R),
                                         (0, -R), (R,-R), (R, 0),
                                         (R, R)]]
    
    geo.Append( ["spline3", pts[0], pts[1], pts[2]],bc="inflow")
    geo.Append( ["spline3", pts[2], pts[3], pts[4]],bc="inflow")

    geo.Append( ["spline3", pts[4], pts[5], pts[6]],bc="outflow")
    geo.Append( ["spline3", pts[6], pts[7], pts[0]],bc="outflow")


# def Make_FlatPlate()