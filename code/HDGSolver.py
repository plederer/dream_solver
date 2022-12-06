from tracemalloc import StatisticDiff
from netgen.geom2d import unit_square, SplineGeometry
from ngsolve import *
from ngsolve.internal import visoptions, viewoptions
import math
import sys
from datetime import datetime
import pickle

from ngsolve.nonlinearsolvers import Newton, NewtonSolver

from flow_utils import *

import os, sys


class compressibleHDGsolver():
    def __init__(self, mesh, order, ff_data, bnd_data, viscid=True, stationary=False, time_solver="IE", force=None):
        self.mesh = mesh

        # bnd names and conditions :
        # "inflow" far-field, subsonic inflow
        # "outflow" subsonic outflow (pressure outflow)
        # "ad_wall" adiabatic wall
        # "iso_wall" isothermal wall
        # "inv_wall" invscid wall
        # "dirichlet" dirichlet

        self.bnd_data = bnd_data

        self.dom_bnd = ""
        for key in self.bnd_data.keys():
            bnd = self.bnd_data[key]
            if type(bnd) == list:
                self.dom_bnd += bnd[0] + "|"
            else:
                self.dom_bnd += bnd + "|"
        self.dom_bnd = self.dom_bnd[:-1]

        self.FU = FlowUtils(ff_data)
        self.viscid = viscid
        self.stationary = stationary
        self.order = order
        self.compile_flag = True
        self.time_solver = time_solver

        self.force = force

        self.time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        # self.base_dir = os.path.join(os.path.abspath(os.getcwd()), "simulation_" + self.time)
        self.base_dir = None  # os.path.abspath(os.getcwd())
        self.solutions_dir = None  # os.path.join(self.base_dir, "solutions")
        self.forces_dir = None  # os.path.join(self.base_dir, "forces")

        #################################################################################

    def SetDirName(self, base_dir):
        os.path.join(os.path.abspath(os.getcwd()), base_dir)
        self.base_dir = base_dir
        self.solutions_dir = os.path.join(self.base_dir, "solutions")
        self.forces_dir = os.path.join(self.base_dir, "forces")

    def InitializeDir(self, base_dir=None):
        if base_dir is not None:
            self.SetDirName(base_dir)
        else:
            self.SetDirName("simulation_" + self.time)
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.solutions_dir, exist_ok=True)
        os.makedirs(self.forces_dir, exist_ok=True)

    def SaveConfig(self, comment=None, save_mesh=False):
        if self.base_dir is None:
            self.InitializeDir()

        file_name = os.path.join(self.base_dir, "config_info")
        file = open(file_name, "w")
        file.write("#" * 40 + "\n")
        file.write("Compressible HDG Solver plugin\n")
        file.write("Authors: Philip Lederer\n")
        file.write("Institute: (2022 - ) ASC TU Wien\n")
        file.write("Funding: FWF (Austrian Science Fund) - P35391N\n")
        file.write("https://github.com/plederer/dream_solver\n")
        file.write("#" * 40 + "\n")
        file.write("Simulation created on: " + self.time + "\n")
        file.write("order: {}".format(self.order) + "\n")
        file.write("Re: {}".format(self.FU.Re.Get()) + "\n")
        file.write("mu: {}".format(self.FU.mu.Get()) + "\n")
        file.write("Pr: {}".format(self.FU.Pr.Get()) + "\n")
        file.write("Minf: {}".format(self.FU.Minf) + "\n")
        file.write("Gamma: {}".format(self.FU.Minf) + "\n")
        file.write("time step: {}".format(self.FU.dt.Get()) + "\n")
        file.write("time solver: {}".format(self.time_solver) + "\n")
        file.write("#" * 40 + "\n")
        if comment:
            file.write(comment + "\n")
            file.write("#" * 40 + "\n")
        file.close()

        outfile = open(os.path.join(self.base_dir, "config_file"), 'wb')
        pickle.dump([self.bnd_data, self.FU.GetData(), self.order], outfile)
        outfile.close

        if save_mesh:
            self.SaveMesh()

    def SaveMesh(self):
        if self.base_dir is None:
            self.InitializeDir()
        outfile = open(os.path.join(self.base_dir, "mesh_file"), 'wb')
        pickle.dump([self.mesh.ngmesh, self.mesh.ngmesh.GetGeometry()], outfile)
        outfile.close

    def SaveSolution(self):
        if self.base_dir is None:
            self.InitializeDir()
        outfile = open(os.path.join(self.base_dir, "gfu_file"), 'wb')
        pickle.dump(self.gfu, outfile)
        outfile.close

    def SaveState(self, s):
        if self.base_dir is None:
            self.InitializeDir()
        state_name = os.path.join(self.solutions_dir, "state_step_" + str(s))
        self.gfu.Save(state_name)

    def LoadState(self, s):
        if self.base_dir is None:
            raise Exception("Please first set base directory with SetDirName()")
        state_name = os.path.join(self.solutions_dir, "state_step_" + str(s))
        self.gfu.Load(state_name)

    def SetUp(self, condense=True):
        self.bi_vol = 0
        self.bi_bnd = 0
        self.condense = condense
        self.V1 = L2(self.mesh, order=self.order)
        self.V2 = FacetFESpace(self.mesh, order=self.order)
        self.V3 = VectorL2(self.mesh, order=self.order)

        if self.viscid:
            if self.FU.Du:
                self.fes = self.V1**4 * self.V2**4 * self.V1**5
            else:
                self.fes = self.V1**4 * self.V2**4 * self.V3**4

        else:
            self.fes = self.V1**4 * self.V2**4

        self.gfu = GridFunction(self.fes)
        self.gfu_old = GridFunction(self.fes)
        self.gfu_old_2 = GridFunction(self.fes)

        self.InitLF()
        self.InitBLF()

    def InitLF(self, force=CF((0, 0, 0, 0))):
        if self.viscid:
            u, uhat, q = self.fes.TrialFunction()
            v, vhat, r = self.fes.TestFunction()

        else:
            u, uhat = self.fes.TrialFunction()
            v, vhat = self.fes.TestFunction()

        self.f = LinearForm(self.fes)
        if self.force is not None:
            self.f += InnerProduct(force, v) * dx(bonus_intorder=self.bi_vol)
        self.f.Assemble()

    def InitBLF(self):
        if self.viscid:
            u, uhat, q = self.fes.TrialFunction()
            v, vhat, r = self.fes.TestFunction()

        else:
            u, uhat = self.fes.TrialFunction()
            v, vhat = self.fes.TestFunction()
            q, r = None, None

        self.a = BilinearForm(self.fes, condense=self.condense)

        h = specialcf.mesh_size
        n = specialcf.normal(2)
        t = specialcf.tangential(2)

        # q = nabla u
        if self.viscid:
            if self.FU.Du:
                eps = CF((q[0], q[1], q[1], q[2]), dims=(2, 2))
                zeta = CF((r[0], r[1], r[1], r[2]), dims=(2, 2))
                Dr = CF(grad(r), dims=(5, 2))
                dev_zeta = 2 * zeta - 2/3 * (zeta[0, 0] + zeta[1, 1]) * Id(2)
                div_dev_zeta = 2 * CF((Dr[0, 0] + Dr[1, 1], Dr[1, 0] + Dr[2, 1])) \
                    - 2/3 * CF((Dr[0, 0] + Dr[2, 0], Dr[0, 1] + Dr[2, 1]))
                vel = self.FU.vel(u)
                vel_hat = self.FU.vel(uhat)

                self.a += (InnerProduct(eps, zeta) + InnerProduct(vel, div_dev_zeta)) \
                    * dx()
                self.a += -InnerProduct(vel_hat, dev_zeta*n) \
                    * dx(element_boundary=True)

                phi = CF((q[3], q[4]))
                xi = CF((r[3], r[4]))
                div_xi = Dr[3, 0] + Dr[4, 1]

                T = self.FU.T(u)
                T_hat = self.FU.T(uhat)

                self.a += (InnerProduct(phi, xi) + InnerProduct(T, div_xi)) \
                    * dx()
                self.a += -InnerProduct(T_hat, xi*n) \
                    * dx(element_boundary=True)

            else:
                self.a += (InnerProduct(q, r) + InnerProduct(u, div(r))) \
                    * dx()
                self.a += -InnerProduct(uhat, r*n) \
                    * dx(element_boundary=True)

        #  konv flux

        self.a += -InnerProduct(self.FU.convective_flux(u), grad(v)).Compile(self.compile_flag) * dx()
        self.a += InnerProduct(self.FU.numerical_convective_flux(uhat, u, uhat, n), v) \
            * dx(element_boundary=True)
        self.a += InnerProduct(self.FU.numerical_convective_flux(uhat, u, uhat, n), vhat) \
            * dx(element_boundary=True)

        # subtract integrals that were added in the line above
        # if "dirichlet" in self.bnd_data:  #self.bnd_data["dirichlet"][0] != "":

        self.a += -InnerProduct(self.FU.numerical_convective_flux(uhat, u, uhat, n), vhat) \
            * ds(skeleton=True, definedon=self.mesh.Boundaries(self.dom_bnd))

        if self.time_solver == "IE" or self.stationary:
            print("Using IE solver")
            self.a += 1/self.FU.dt * InnerProduct(u - self.gfu_old.components[0], v) * dx
        else:
            print("Using BDF2 solver")
            self.a += 3/2 * 1/self.FU.dt * InnerProduct(
                1 * u - 4/3 * self.gfu_old.components[0] + 1/3 * self.gfu_old_2.components[0], v) * dx

        #  diff flux
        if self.viscid:
            self.a += InnerProduct(self.FU.diffusive_flux(u, q), grad(v)).Compile(self.compile_flag) \
                * dx(bonus_intorder=self.bi_vol)
            self.a += -InnerProduct(self.FU.numerical_diffusive_flux(u, uhat, q, n), v).Compile(self.compile_flag) \
                * dx(element_boundary=True, bonus_intorder=self.bi_bnd)

            self.a += -InnerProduct(self.FU.numerical_diffusive_flux(u, uhat, q, n), vhat).Compile(self.compile_flag) \
                * dx(element_boundary=True, bonus_intorder=self.bi_bnd)
            # if "dirichlet" in self.bnd_data:
            self.a += InnerProduct(self.FU.numerical_diffusive_flux(u, uhat, q, n), vhat).Compile(self.compile_flag) \
                * ds(skeleton=True, definedon=self.mesh.Boundaries(self.dom_bnd))

        # bnd fluxes

        if "dirichlet" in self.bnd_data:
            Bhat = self.bnd_data["dirichlet"][1]
            self.a += ((Bhat-uhat) * vhat) \
                * ds(skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["dirichlet"][0]))

        if "NRBC" in self.bnd_data:
            pinf = self.bnd_data["NRBC"][1]
            L = self.FU.charachteristic_amplitudes(u, q, n, uhat)
            K = 0.25 * self.FU.c(uhat) * (1 - self.FU.M(uhat)) / h
            incoming = K * (self.FU.p(uhat) - pinf)
            # incoming = 0
            L = CF((L[0], L[1], L[2], incoming))
            D = self.FU.P_matrix(uhat, n) * L

            Ft = self.FU.tangential_flux_gradient(u, q, t)
            self.a += 3/2*1/self.FU.dt * InnerProduct(uhat - 4/3*self.gfu_old.components[1] + 1/3 * self.gfu_old_2.components[1], vhat) * ds(
                skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["NRBC"][0]), bonus_intorder=10)
            self.a += (D * vhat) * ds(skeleton=True,
                                      definedon=self.mesh.Boundaries(self.bnd_data["NRBC"][0]), bonus_intorder=10)
            self.a += ((Ft * t) * vhat) * ds(skeleton=True,
                                             definedon=self.mesh.Boundaries(self.bnd_data["NRBC"][0]), bonus_intorder=10)

        if "Test" in self.bnd_data:
            vn = InnerProduct(self.FU.vel(uhat), n)
            abs_vn = IfPos(vn, vn, -vn)
            s_p = IfPos(vn + self.FU.c(uhat), vn + self.FU.c(uhat), 0)
            s_m = IfPos(vn - self.FU.c(uhat), 0, vn - self.FU.c(uhat))

            coef = abs_vn/(abs_vn + self.FU.c(uhat))
            spd = CF((coef, 0, 0, 0,
                      0, coef, 0, 0,
                      0, 0, 1, 0,
                      0, 0, 0, 1), dims=(4, 4))
            spd = self.FU.P_matrix(uhat, n) * spd * self.FU.P_inverse_matrix(uhat, n)

            inflow = CF(tuple(self.bnd_data["Test"][1]))
            Bhat = s_p * spd * (u-uhat)
            Bhat += s_m * spd * (inflow - uhat)

            self.a += (InnerProduct(Bhat, vhat)).Compile(self.compile_flag) \
                * ds(skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["Test"][0]))

        if "ForceFree" in self.bnd_data:

            rho = self.FU.rho(uhat)
            c = self.FU.c(uhat)
            vel = self.FU.vel(uhat) * n
            velgrad = (self.FU.gradvel(u) * n) * n

            L = self.FU.charachteristic_amplitudes(u, q, n, uhat)
            incoming = L[2] - 2*rho*c*vel*velgrad
            L = CF((L[0], L[1], L[2], incoming))
            D = self.FU.P_matrix(uhat, n) * L

            Ft = self.FU.tangential_flux_gradient(u, q, t)
            self.a += 1/self.FU.dt * InnerProduct(uhat - self.gfu_old.components[1], vhat) * ds(
                skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["ForceFree"][0]), bonus_intorder=10)
            self.a += (D * vhat) * ds(skeleton=True,
                                      definedon=self.mesh.Boundaries(self.bnd_data["ForceFree"][0]), bonus_intorder=10)
            self.a += ((Ft * t) * vhat) * ds(skeleton=True,
                                             definedon=self.mesh.Boundaries(self.bnd_data["ForceFree"][0]), bonus_intorder=10)

        if "inflow" in self.bnd_data:
            Bhat = self.FU.Aplus(uhat, n) * (u-uhat)
            inflow_cf = CF(tuple(self.bnd_data["inflow"][1]))
            Bhat += self.FU.Aminus(uhat, n) * (inflow_cf-uhat)
            self.a += (Bhat * vhat).Compile(self.compile_flag) \
                * ds(skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["inflow"][0]))

        if "inv_wall" in self.bnd_data:
            # if self.viscid and "inv_wall" in self.bnd_data:
            #     raise Exception("inv_val boundary only for inviscid flows available")
            self.a += (InnerProduct(self.FU.reflect(u, n)-uhat, vhat)).Compile(self.compile_flag) \
                * ds(skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["inv_wall"]))

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

        if "outflow" in self.bnd_data:
            p_out = self.bnd_data["outflow"][1]
            rho_E = p_out/(self.FU.gamma - 1) + 1/(2*u[0]) * (u[1]**2 + u[2]**2)
            U = CF((u[0], u[1], u[2], rho_E))
            Bhat = U - uhat
            self.a += (InnerProduct(Bhat, vhat)).Compile(self.compile_flag) \
                * ds(skeleton=True, definedon=self.mesh.Boundaries(self.bnd_data["outflow"][0]))

        # self.a += (   (u-uhat) * vhat) * ds(skeleton = True, \
        # definedon = self.mesh.Boundaries(self.dirichlet))
        #######################################################################

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
