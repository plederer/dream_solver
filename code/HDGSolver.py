from netgen.geom2d import unit_square, SplineGeometry
from ngsolve import *
from ngsolve.internal import visoptions, viewoptions
import math
import sys

from ngsolve.nonlinearsolvers import Newton, NewtonSolver

from flow_utils import *

class compressibleHDGsolver():
    def __init__(self, mesh, order, ff_data, bnd_data, viscid=True, stationary=True):
        self.mesh = mesh

        # bnd names and conditions :
        # "inflow" far-field, subsonic inflow
        # "outflow" subsonic outflow (pressure outflow)
        # "ad_wall" adiabatic wall
        # "iso_wall" isothermal wall
        # "inv_wall" invscid wall
        # "dirichlet" dirichlet
        
        self.bnd_data = bnd_data

        self.FU = FlowUtils(ff_data)
        self.viscid = viscid
        self.stationary = stationary
        self.order = order
        #################################################################################
        
    def SetUp(self, force=CoefficientFunction((0, 0, 0, 0)), condense=False):
        self.condense = condense
        bi_vol = 0
        bi_bnd = 0
        self.V1 = L2(self.mesh, order=self.order)
        self.V2 = FacetFESpace(self.mesh, order=self.order)
        self.V3 = VectorL2(self.mesh, order=self.order)
        
        if self.viscid:
            self.fes = self.V1**4 * self.V2**4 * self.V3**4
            u, uhat, q = self.fes.TrialFunction()
            v, vhat, r = self.fes.TestFunction()
        else:
            self.fes = self.V1**4 * self.V2**4
            u, uhat = self.fes.TrialFunction()
            v, vhat = self.fes.TestFunction()
                                                   
        # u = CoefficientFunction((u1,u2,u3,u4))
        # v = CoefficientFunction((v1,v2,v3,v4))
 
        # uhat = CoefficientFunction((uhat1,uhat2,uhat3,uhat4))
        # vhat = CoefficientFunction((vhat1,vhat2,vhat3,vhat4))
 
        # uhat_trace = CoefficientFunction((uhat1.Trace(),uhat2.Trace(),uhat3.Trace(),uhat4.Trace()))
        # vhat_trace = CoefficientFunction((vhat1.Trace(),vhat2.Trace(),vhat3.Trace(),vhat4.Trace()))

        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)
        # gfu_old = GridFunction(fes)

        # gfu_old = CoefficientFunction((gfu_old.components[0],gfu_old.components[1],gfu_old.components[2],gfu_old.components[3]))

        h = specialcf.mesh_size
        n = specialcf.normal(2)

        # V3 = FacetFESpace(self.mesh, order=0, dirichlet=self.dirichlet)
        # psi = GridFunction(V3)
        # psi.Set(1, BND)
        
        # Bilinearform
        self.a = BilinearForm(self.fes, condense=self.condense)

        # time derivative
        #a += (u - gfu_old) * v * dx()
        #a += u * v * dx()

        # q = nabla u 
        if self.viscid:
            self.a += (InnerProduct(q, r) + InnerProduct(u, div(r))).Compile() \
                * dx()
            self.a += -InnerProduct(uhat, r*n).Compile() \
                * dx(element_boundary=True)

        #  konv flux
        self.a += -InnerProduct(self.FU.Flux(u), grad(v)).Compile() * dx()
        self.a += InnerProduct(self.FU.numFlux(uhat, u, uhat, n), v).Compile() \
            * dx(element_boundary=True)
        self.a += InnerProduct(self.FU.numFlux(uhat, u, uhat, n), vhat).Compile() \
            * dx(element_boundary=True)

        # subtract integrals that were added in the line above
        # if "dirichlet" in self.bnd_data:  #self.bnd_data["dirichlet"][0] != "":
        self.a += -InnerProduct(self.FU.numFlux(uhat, u, uhat, n), vhat).Compile() \
                * ds(skeleton=True) #, definedon=self.mesh.Boundaries(self.bnd_data["dirichlet"][0]))


        # if self.stationary:
        self.a += 1/self.FU.dt * InnerProduct(u - self.gfu_old.components[0], v) * dx 


        #  diff flux
        if self.viscid:
            self.a += -InnerProduct(self.FU.diffFlux(u, q ), grad(v)).Compile() \
                * dx(bonus_intorder=bi_vol)
            self.a += InnerProduct(self.FU.numdiffFlux(u, uhat, q, n), v).Compile() \
                * dx(element_boundary=True, bonus_intorder=bi_bnd)
            self.a += InnerProduct(self.FU.numdiffFlux(u, uhat, q, n), vhat).Compile() \
                * dx(element_boundary=True, bonus_intorder=bi_bnd)
            #if "dirichlet" in self.bnd_data:
            self.a += -InnerProduct(self.FU.numdiffFlux(u, uhat, q, n), vhat).Compile() \
                    * ds(skeleton=True) #, definedon=self.mesh.Boundaries(self.bnd_data["dirichlet"][0]))

        # bnd fluxes

        # for bnd_name in 
        # a += (  (Aplus(uhat,n) * (u-uhat) + Aminus(uhat, n) * (cf0-uhat)) * vhat) \
        #     * ds(skeleton = True, definedon = mesh.Boundaries("left|right|top|bottom"))

        # print(self.bnd_data)
        if "dirichlet" in self.bnd_data:
            Bhat = self.bnd_data["dirichlet"][1]
            self.a += ((Bhat-uhat) * vhat) \
                * ds(skeleton = True, definedon=self.mesh.Boundaries(self.bnd_data["dirichlet"][0]))
        
        if "inflow" in self.bnd_data:
            Bhat = self.FU.Aplus(uhat,n) * (u-uhat)
            Bhat += self.FU.Aminus(uhat, n) * (self.bnd_data["inflow"][1]-uhat)
            self.a += (Bhat * vhat) \
                * ds(skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["inflow"][0]))

        if "inv_wall" in self.bnd_data:
            # if self.viscid and "inv_wall" in self.bnd_data:
            #     raise Exception("inv_val boundary only for inviscid flows available")
            self.a += (InnerProduct(self.FU.reflect(u,n)-uhat,vhat)) \
                * ds(skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["inv_wall"]))

        if "iso_wall" in self.bnd_data:
            if not self.viscid:
                raise Exception("iso_val boundary only for viscid flows available")
            T_w = self.bnd_data["iso_wall"][1]
            rho_E = u[0] * T_w/self.FU.gamma
            U = CF((u[0], u[1], u[2], rho_E))
            self.a += (-InnerProduct(uhat-U,vhat)) \
                * ds(skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["iso_wall"][0]))
        
        if "ad_wall" in self.bnd_data:
            if not self.viscid:
                raise Exception("ad_val boundary only for viscid flows available")
            tau_d = 1/((self.FU.gamma - 1) * self.FU.Minf**2 * self.FU.Re * self.FU.Pr)
            rho_E = self.FU.mu/(self.FU.Re*self.FU.Pr) * self.FU.gradT(u, q) * n - tau_d * (u[3] - uhat[3])
            B = CF((u[0] - uhat[0], u[1], u[2], rho_E))
            self.a += (InnerProduct(B,vhat)) \
                * ds(skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["ad_wall"]))


        if "outflow" in self.bnd_data:
            p_out = self.bnd_data["outflow"][1]
            rho_E = p_out/(self.FU.gamma - 1) + u[0]/2 * ((u[1]/u[0])**2 + (u[2]/u[0])**2)
            U = CF((u[0], u[1], u[2], rho_E))
            # U = u
            self.a += (-InnerProduct(uhat-U,vhat)) \
                * ds(skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["outflow"][0]))

        # self.a += (   (u-uhat) * vhat) * ds(skeleton = True, \
        # definedon = self.mesh.Boundaries(self.dirichlet))

        self.f = LinearForm(self.fes)
        self.f += InnerProduct(force, v) * dx(bonus_intorder=bi_vol)
        self.f.Assemble()
                
        #######################################################################
        
    def SetInitial(self, U_init, Q_init = None):        
        # Initial conditions
        # set to zero
        # for i in range(4):
        #    gfu.components[0].Set(1/)
        #    gfu.components[i+4].Set(1e-6,BND)
        #    gfu.components[i+8].Set(1e-6)
        #    gfu.components[i+12].Set(1e-6)
        
        if self.viscid:
            u, uhat, q = self.fes.TrialFunction()
            v, vhat, r = self.fes.TestFunction()
        else:
            u, uhat = self.fes.TrialFunction()
            v, vhat = self.fes.TestFunction()

        self.m = BilinearForm(self.fes)
        if self.viscid:
            self.m += InnerProduct(q,r) * dx()

        self.m += u * v * dx()
        self.m += uhat * vhat * dx(element_boundary=True)
        self.m.Assemble()
        
        minv = self.m.mat.Inverse(self.fes.FreeDofs(), inverse="sparsecholesky")

        t0 = LinearForm(self.fes)
        if self.viscid:
            t0 += InnerProduct(Q_init, r) * dx()
        t0 += U_init * v * dx()
        t0 += U_init * vhat * dx(element_boundary=True)
        # if "ad_wall" in self.bnd_data:
        #     t0 += -InnerProduct(U_init, vhat) \
        #             * ds(skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["ad_wall"]))
        
        t0.Assemble()

        self.gfu.vec.data = minv * t0.vec
        #################################################################################

    def Solve(self, maxit=10, maxerr=1e-8, dampfactor=1, solver="pardiso", printing=True):
        res = self.gfu.vec.CreateVector()
        w = self.gfu.vec.CreateVector()

        # solver = NewtonSolver(a=self.a, u=self.gfu, rhs=self.f, freedofs=None,
        #          inverse="pardiso", solver=None)
        
        # solver.Solve(maxit=maxit, maxerr=maxerr, dampfactor=dampfactor,
        #              printing=True, callback=None, linesearch=False,
        #              printenergy=False, print_wrong_direction=False)
        
        for it in range(maxit):
            if self.stationary == True:
                self.gfu_old.vec.data = self.gfu.vec
            # if it < 10:
            #     dampfactor = 0.001

            if self.stationary == True:
                if (it%10 == 0) and (it > 0) and (self.FU.dt.Get() < 1):
                    c_dt = self.FU.dt.Get() * 10
                    self.FU.dt.Set(c_dt)
                    print("new dt = ", c_dt)

            if printing:
                print ("Newton iteration", it)

            self.a.Apply(self.gfu.vec, res)
            
            
            self.a.AssembleLinearization(self.gfu.vec)
            res.data -= self.f.vec
            # print(Norm(res))
            inv = self.a.mat.Inverse(self.fes.FreeDofs(self.a.condense), inverse = solver) 
            if self.a.condense:
                # print("condense")
                res.data += self.a.harmonic_extension_trans * res
                w.data = inv * res
                w.data += self.a.harmonic_extension * w
                w.data += self.a.inner_solve * res
            else:
                w.data = inv * res

            self.gfu.vec.data -= dampfactor *  w

            err = sqrt(InnerProduct (w,res)**2)
            if printing:
                print("err = ", err)
            if err < maxerr:
                break

            Redraw()
        if not self.stationary:
            self.gfu_old.vec.data = self.gfu.vec
        #     # input()
        #################################################################################

    @property
    def density(self):
        return self.gfu.components[0][0]

    @property
    def velocity(self):
        return CoefficientFunction((self.gfu.components[0][1]/self.gfu.components[0][0],self.gfu.components[0][2]/self.gfu.components[0][0]))

    @property
    def grad_velocity(self):
        return self.FU.gradvel(self.gfu.components[0], self.gfu.components[2])

    @property
    def pressure(self):
        # return (self.FU.gamma - 1) * (self.gfu.components[0][3] - self.gfu.components[0][0]/2 * InnerProduct(self.velocity,self.velocity))
        return self.FU.R * (self.gfu.components[0][3] - self.gfu.components[0][0]/2 * InnerProduct(self.velocity,self.velocity))

    @property
    def temperature(self):        
        # return self.ff_data["gamma"] * self.ff_data["Minf"]**2 * self.pressure / self.density
        return  self.pressure / self.density / self.FU.R

    @property
    def energy(self):
        return self.gfu.components[0][3] / self.gfu.components[0][0]

    @property
    def c(self):
        return sqrt(self.FU.gamma * self.FU.R * self.temperature)
        
    @property
    def M(self):
        return sqrt(self.velocity[0]**2 + self.velocity[1]**2)/self.c