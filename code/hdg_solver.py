from tracemalloc import StatisticDiff
from netgen.geom2d import unit_square, SplineGeometry
from ngsolve import *
from ngsolve.internal import visoptions, viewoptions
from configuration import SolverConfiguration, DynamicViscosity, MixedMethods
from formulations import formulation_factory, Formulation
import boundary_conditions as bc
from datetime import datetime
from ngsolve.nonlinearsolvers import Newton, NewtonSolver

from flow_utils import *

import os
import sys


class compressibleHDGsolver():
    def __init__(self, mesh, solver_configuration: SolverConfiguration):

        self.mesh = mesh
        self.solver_configuration = solver_configuration
        self._formulation = formulation_factory(mesh, solver_configuration)
        self._bcs = self.formulation.bcs

    @property
    def boundary_conditions(self) -> bc.BoundaryConditions:
        return self._bcs

    @property
    def formulation(self) -> Formulation:
        return self._formulation

    def _set_linearform(self, force):

        fes = self.formulation.fes
        TnT = self.formulation.TnT

        bonus_int_order = self.solver_configuration.bonus_int_order_vol

        (_, _, _), (V, _, _) = TnT

        self.f = LinearForm(fes)
        if force is not None:
            self.f += InnerProduct(force, V) * dx(bonus_intorder=bonus_int_order)
        self.f.Assemble()

    def _set_bilinearform(self):

        fes = self.formulation.fes
        TnT = self.formulation.TnT
        mesh = self.formulation.mesh
        time_scheme = self.formulation.time_scheme

        condense = self.solver_configuration.static_condensation
        viscosity = self.solver_configuration.dynamic_viscosity
        mixed_method = self.solver_configuration.mixed_method
        bonus_vol = self.solver_configuration.bonus_int_order_vol

        h = specialcf.mesh_size
        n = specialcf.normal(2)
        t = specialcf.tangential(2)

        (U, Uhat, Q), (V, Vhat, P) = TnT

        self.a = BilinearForm(fes, condense=condense)
        self.formulation.set_time_bilinearform(self.a, self.gfu_old)

        self.formulation.set_convective_bilinearform(self.a)

        if viscosity is not DynamicViscosity.INVISCID:
            self.formulation.set_diffusive_bilinearform(self.a)

        if mixed_method is not MixedMethods.NONE:
            self.formulation.set_mixed_bilinearform(self.a)

        self.formulation.set_boundary_conditions_bilinearform(self.a)

        # bnd fluxes

        if "iso_wall" in self.bnd_data:
            if not self.viscid:
                raise Exception("iso_val boundary only for viscid flows available")
            T_w = self.bnd_data["iso_wall"][1]
            rho_E = u[0] * T_w/self.FU.gamma
            U = CF((u[0], u[1], u[2], rho_E))
            Bhat = U - uhat
            self.a += (InnerProduct(Bhat, vhat)).Compile(self.compile_flag) \
                * ds(skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["iso_wall"][0]))

        if "ad_wall" in self.bnd_data:
            if not self.viscid:
                raise Exception("ad_val boundary only for viscid flows available")
            tau_d = self.FU.tau_d * 1/self.FU.Re.Get()  # 1/((self.FU.gamma - 1) * self.FU.Minf**2 * self.FU.Re * self.FU.Pr)
            rho_E = self.FU.mu.Get()/(self.FU.Re.Get()*self.FU.Pr.Get()
                                      ) * self.FU.gradT(u, q) * n - tau_d * (u[3] - uhat[3])
            Bhat = CF((u[0] - uhat[0], uhat[1], uhat[2], rho_E))
            self.a += (InnerProduct(Bhat, vhat)).Compile(self.compile_flag) \
                * ds(skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["ad_wall"]))

        # self.a += (   (u-uhat) * vhat) * ds(skeleton = True, \
        # definedon = self.mesh.Boundaries(self.dirichlet))
        #######################################################################

    def setup(self, force: CF = None):

        num_temporary_vectors = self.formulation.time_scheme.num_temporary_vectors

        self.gfu = GridFunction(self.formulation.fes)
        self.gfu_old = tuple(GridFunction(self.formulation.fes) for num in range(num_temporary_vectors))

        self._set_linearform(force)
        self._set_bilinearform()

    def SetInitial(self, U_init, Q_init=None):
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
            self.m += InnerProduct(q, r) * dx()

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
        #     t0 += InnerProduct(U_init_hat, vhat) \
        #             * ds(skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["ad_wall"]))

        t0.Assemble()

        self.gfu.vec.data = minv * t0.vec
        #################################################################################

    def Solve(self, maxit=10, maxerr=1e-8, dampfactor=1, solver="pardiso", printing=True, stop=False, max_dt=1, stat_step=10):
        res = self.gfu.vec.CreateVector()
        w = self.gfu.vec.CreateVector()
        # solver = NewtonSolver(a=self.a, u=self.gfu, rhs=self.f, freedofs=None,
        #          inverse="pardiso", solver=None)

        # solver.Solve(maxit=maxit, maxerr=maxerr, dampfactor=dampfactor,
        #              printing=True, callback=None, linesearch=False,
        #              printenergy=False, print_wrong_direction=False)
        # if not self.stationary:
        #     self.gfu_old.vec.data = self.gfu.vec
        #     self.gfu_old_2.vec.data = self.gfu_old.vec

        for it in range(maxit):
            if self.stationary:
                self.gfu_old.vec.data = self.gfu.vec
            # if it < 10:
            #     dampfactor = 0.001

            if self.stationary:
                if (it % stat_step == 0) and (it > 0) and (self.FU.dt.Get() < max_dt):
                    c_dt = self.FU.dt.Get() * 10
                    self.FU.dt.Set(c_dt)
                    print("new dt = ", c_dt)

            if printing:
                print("Newton iteration", it)

            self.a.Apply(self.gfu.vec, res)

            res.data -= self.f.vec

            try:
                self.a.AssembleLinearization(self.gfu.vec)
            except Exception as e:
                print("\nThis did not work!! Try smaller time step \n")
                raise e

            inv = self.a.mat.Inverse(self.fes.FreeDofs(self.a.condense), inverse=solver)
            if self.a.condense:
                # print("condense")
                res.data += self.a.harmonic_extension_trans * res
                w.data = inv * res
                w.data += self.a.harmonic_extension * w
                w.data += self.a.inner_solve * res
            else:
                w.data = inv * res

            self.gfu.vec.data -= dampfactor * w

            if self.stationary:
                # res.data = self.gfu_old.vec - self.gfu.vec
                # err = sqrt(InnerProduct (res,res)/InnerProduct(self.gfu_old.vec,self.gfu_old.vec))
                err = sqrt(InnerProduct(w, res)**2)
            else:
                err = sqrt(InnerProduct(w, res)**2)
            if printing:
                print("err = ", err)
            if err < maxerr:
                break

            if self.stationary:
                Redraw()
            if stop:
                input()

        Redraw()
        # if not self.stationary:
        #     self.gfu_old.vec.data = self.gfu.vec
        #     # input()
        #################################################################################

    @property
    def density(self):
        return self.FU.rho(self.gfu.components[0])

    @property
    def velocity(self):
        return self.FU.vel(self.gfu.components[0])

    @property
    def vorticity(self):
        # dux_minus_dvy = 0.5 * (self.gfu.components[2][0] - self.gfu.components[2][0])
        # duy_plus_dvx = self.gfu.components[2][1]
        q = CF(Grad(self.gfu.components[0]), dims=(4, 2))

        # u_x = 1/self.density * (q[1,0] - q[0,0] * self.velocity[0])
        u_y = 1/self.density * (q[1, 1] - q[0, 1] * self.velocity[0])
        v_x = 1/self.density * (q[2, 0] - q[0, 0] * self.velocity[1])
        # v_y = 1/self.density * (q[2,1] - q[0,1] * self.velocity[1])

        vort = u_y - v_x
        return vort

    @property
    def grad_velocity(self):
        return self.FU.gradvel(self.gfu.components[0], self.gfu.components[2])

    @property
    def pressure(self):
        return self.FU.p(self.gfu.components[0])

    @property
    def temperature(self):
        return self.FU.T(self.gfu.components[0])

    @property
    def energy(self):
        return self.FU.E(self.gfu.components[0])

    @property
    def c(self):
        return self.FU.c(self.gfu.components[0])

    @property
    def M(self):
        return self.FU.M(self.gfu.components[0])

    def DrawSolutions(self):
        Draw(self.velocity, self.mesh, "u")
        Draw(self.vorticity, self.mesh, "omega")
        Draw(self.pressure, self.mesh, "p")
        Draw(self.c, self.mesh, "c")
        Draw(self.M, self.mesh, "M")
        Draw(self.temperature, self.mesh, "T")
        Draw(self.energy, self.mesh, "E")
        Draw(self.density, self.mesh, "rho")
        visoptions.scalfunction = 'u:0'

    def CalcForces(self, surf, scale=1):
        q = self.gfu.components[2]
        sigma_visc = self.FU.mu.Get()/self.FU.Re.Get() * CF((q[0], q[1], q[1], q[2]), dims=(2, 2))

        sigma_bnd = BoundaryFromVolumeCF(sigma_visc - self.pressure * Id(2))
        n = specialcf.normal(2)
        forces = Integrate(sigma_bnd * n, self.mesh, definedon=self.mesh.Boundaries(surf))

        fd = scale * forces[0]
        fl = scale * forces[1]

        return fd, fl

    def SaveForces(self, t, surf, scale=1, init=False):
        if self.base_dir is None:
            self.InitializeDir()
        if init:
            outfile = open(os.path.join(self.forces_dir, "force_file"), 'w')
        else:
            outfile = open(os.path.join(self.forces_dir, "force_file"), 'a')
        fd, fl = self.CalcForces(surf, scale)
        outfile.write("{}\t{}\t{}\n".format(t, fd, fl))
        outfile.close
