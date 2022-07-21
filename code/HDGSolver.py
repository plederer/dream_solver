from netgen.geom2d import unit_square, SplineGeometry
from ngsolve import *
from ngsolve.internal import visoptions, viewoptions
import math
import sys

from ngsolve.nonlinearsolvers import Newton, NewtonSolver

from flow_utils import *

class compressibleHDGsolver():
    def __init__(self, mesh, order, ff_data, bnd_data, bnd_names):
        self.mesh = mesh

        # boundary conditions, bnd_names is a set
        self.inflow = bnd_names["inflow"]  # far-field, subsonic inflow
        self.ss_outflow = bnd_names["ss_outflow"]  # super sonic outflow (pressure outflow)
        self.ad_wall = bnd_names["ad_wall"]  # adiabatic wall
        self.iso_wall = bnd_names["iso_wall"]  # isothermal wall
        self.inv_wall = bnd_names["inv_wall"]  # invscid wall
        self.dirichlet = bnd_names["dirichlet"]  # invscid wall
        
        self.FU = FlowUtils(ff_data)

        self.order = order
        self.bnd_data = bnd_data
        self.ff_data = ff_data
        #################################################################################
        
    def SetUp(self, force=CoefficientFunction((0, 0, 0, 0)), condense=False):
        self.condense = condense
        bi_vol = 0
        bi_bnd = 0
        self.V1 = L2(self.mesh, order=self.order)
        self.V2 = FacetFESpace(self.mesh, order=self.order)
        self.V3 = VectorL2(self.mesh, order=self.order)
        
        self.fes = self.V1**4 * self.V2**4 * self.V3**4

        u, uhat, q = self.fes.TrialFunction()
        v, vhat, r = self.fes.TestFunction()
                                                   
        # u = CoefficientFunction((u1,u2,u3,u4))
        # v = CoefficientFunction((v1,v2,v3,v4))
 
        # uhat = CoefficientFunction((uhat1,uhat2,uhat3,uhat4))
        # vhat = CoefficientFunction((vhat1,vhat2,vhat3,vhat4))
 
        # uhat_trace = CoefficientFunction((uhat1.Trace(),uhat2.Trace(),uhat3.Trace(),uhat4.Trace()))
        # vhat_trace = CoefficientFunction((vhat1.Trace(),vhat2.Trace(),vhat3.Trace(),vhat4.Trace()))

        self.gfu = GridFunction(self.fes)
        # gfu_old = GridFunction(fes)

        # gfu_old = CoefficientFunction((gfu_old.components[0],gfu_old.components[1],gfu_old.components[2],gfu_old.components[3]))

        h = specialcf.mesh_size
        n = specialcf.normal(2)

        V3 = FacetFESpace(self.mesh, order=0, dirichlet=self.dirichlet)
        psi = GridFunction(V3)
        psi.Set(1, BND)
        
        # Bilinearform
        self.a = BilinearForm(self.fes, condense=self.condense)

        # time derivative
        #a += (u - gfu_old) * v * dx()
        #a += u * v * dx()

        # q = nabla u 
        self.a += (InnerProduct(q, r) + InnerProduct(u, div(r))).Compile() \
            * dx()
        self.a += -InnerProduct(uhat, r*n).Compile() \
            * dx(element_boundary=True)

        #  konv flux
        self.a += -InnerProduct(self.FU.Flux(u), grad(v)).Compile() * dx()

        self.a += InnerProduct(self.FU.numFlux(uhat, u, uhat, n), v).Compile() \
            * dx(element_boundary=True)
        self.a += (1-psi) * InnerProduct(self.FU.numFlux(uhat, u, uhat, n), vhat).Compile() \
            * dx(element_boundary=True)
        
        # diff flux
        self.a += -InnerProduct(self.FU.diffFlux(u, q ), grad(v)).Compile() \
            * dx(bonus_intorder=bi_vol)
        self.a += InnerProduct(self.FU.numdiffFlux(u, uhat, q, n), v).Compile() \
            * dx(element_boundary=True, bonus_intorder=bi_bnd)
        self.a += (1-psi) * InnerProduct(self.FU.numdiffFlux(u, uhat, q, n), vhat).Compile() \
            * dx(element_boundary=True, bonus_intorder=bi_bnd)

        # bnd fluxes
        # a += (  (Aplus(uhat,n) * (u-uhat) + Aminus(uhat, n) * (cf0-uhat)) * vhat) * ds(skeleton = True, definedon = mesh.Boundaries("left|right|top|bottom"))

        # ubnd = CoefficientFunction((rho_ex,rho_ex * u_ex,rho_ex * v_ex, rho_ex * E_ex))
        
        # ubnd = CoefficientFunction((u[0],rho_ex * u_ex,rho_ex * v_ex, rho_ex * E_ex))

        #self.a += (   (self.bnd_data-uhat) * vhat) * ds(skeleton = True, \
        # definedon = self.mesh.Boundaries(self.dirichlet))
        #self.a += ( psi *   (self.bnd_data-uhat) * vhat) \
        # * dx(element_boundary = True) #ds(skeleton = True, \
        # definedon = self.mesh.Boundaries(self.dirichlet))

        self.a += ( (self.bnd_data-uhat.Trace()) * vhat.Trace()) \
            * ds(definedon=self.mesh.Boundaries(self.dirichlet))
        
        # self.a += (   (u-uhat) * vhat) * ds(skeleton = True, \
        # definedon = self.mesh.Boundaries(self.dirichlet))

        self.f = LinearForm(self.fes)
        self.f += InnerProduct(force, v) * dx(bonus_intorder=bi_vol)
        self.f.Assemble()
                
        #################################################################################
        
    def SetInitial(self, U_init, Q_init):        
        # Initial conditions
        # set to zero
        # for i in range(4):
        #    gfu.components[0].Set(1/)
        #    gfu.components[i+4].Set(1e-6,BND)
        #    gfu.components[i+8].Set(1e-6)
        #    gfu.components[i+12].Set(1e-6)
        
        u, uhat, q = self.fes.TrialFunction()
        v, vhat, r = self.fes.TestFunction()

        self.m = BilinearForm(self.fes)

        self.m += InnerProduct(q,r) * dx()
        self.m += u * v * dx()
        self.m += uhat * vhat * dx(element_boundary=True)
        self.m.Assemble()
        
        minv = self.m.mat.Inverse(self.fes.FreeDofs(), inverse="sparsecholesky")

        t0 = LinearForm(self.fes)
        t0 += InnerProduct(Q_init, r) * dx()
        t0 += U_init * v * dx()
        t0 += U_init * vhat * dx(element_boundary=True)
        t0.Assemble()

        self.gfu.vec.data = minv * t0.vec
        #################################################################################

    def Solve(self, maxit=10, maxerr=1e-8, dampfactor=1, solver = "pardiso", printing = True):                       
        res = self.gfu.vec.CreateVector()
        w = self.gfu.vec.CreateVector()


        # solver = NewtonSolver(a=self.a, u=self.gfu, rhs=self.f, freedofs=None,
        #          inverse="pardiso", solver=None)
        
        # solver.Solve(maxit=maxit, maxerr=maxerr, dampfactor=dampfactor,
        #              printing=True, callback=None, linesearch=False,
        #              printenergy=False, print_wrong_direction=False)
        
        for it in range(maxit):
            # if it < 10:
            #     dampfactor = 0.2
            if printing:
                print ("Newton iteration", it)
            self.a.Apply(self.gfu.vec, res)
            # print(Norm(res))
            
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

            # print(Norm(w))
            self.gfu.vec.data -= dampfactor *  w
            Redraw()
            
            err = sqrt(InnerProduct (w,res)**2)
            if printing:
                print("err = ", err)
            if err < maxerr:
                break
            # input()
        #################################################################################

    @property
    def density(self):
        return self.gfu.components[0][0]

    @property
    def velocity(self):
        return CoefficientFunction((self.gfu.components[0][1]/self.gfu.components[0][0],self.gfu.components[0][2]/self.gfu.components[0][0]))

    @property
    def pressure(self):
        return (self.ff_data["gamma"] - 1) * (self.gfu.components[0][3] - self.gfu.components[0][0]/2 * InnerProduct(self.velocity,self.velocity))

    @property
    def temperature(self):        
        # return self.ff_data["gamma"] * self.ff_data["Minf"]**2 * self.pressure / self.density
        return self.ff_data["R"] * self.pressure / self.density

    @property
    def energy(self):
        return self.gfu.components[0][3] / self.gfu.components[0][0]

    @property
    def c(self):
        return sqrt(self.ff_data["gamma"] * self.temperature)
        
    @property
    def M(self):
        return sqrt(self.velocity[0]**2 + self.velocity[1]**2)/self.c

    @property
    def GFU(self):
        return self.gfu

