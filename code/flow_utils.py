from ngsolve import *

import math
from math import pi, atan2

class FlowUtils():
    def __init__(self, ff_data):

        # stationary solution without pseudo time stepping
        if "dt" not in ff_data.keys():
            ff_data["dt"] = 1e10

        if "Du" not in ff_data.keys():
            ff_data["Du"] = True

        self.Du = ff_data["Du"]

        for k in ["Re", "Pr", "mu", "Minf", "gamma"]:
            if k not in ff_data.keys():
                print("Setting standard value: {} = 1".format(k))
                ff_data[k] = 1

            setattr(self, k, ff_data[k])

        if "R" in ff_data.keys():
            print("Are you sure about that? Values might be incorrect")
            self.R = ff_data["R"]
        else:
            self.R = self.gamma - 1
        
        # time step for pseudo time stepping
        self.dt = Parameter(ff_data["dt"])

    def rho(self, u):
        return u[0]
    def vel(self, u):
        return CF((u[1]/u[0], u[2]/u[0]))
    def p(self, u):
        # p = (gamma-1) * rho*E - rho/2 * ||v||**2    
        # return (self.gamma-1) * (u[3] - u[0]/2 * ((u[1]/u[0])**2 + (u[2]/u[0])**2))
        return self.R * (u[3] - u[0]/2 * ((u[1]/u[0])**2 + (u[2]/u[0])**2))
    def T(self, u):
        return self.p(u)/(u[0] * self.R)
    def E(self, u):
        return u[3]/u[0]
    def c(self, u):
        return sqrt(self.gamma * self.R * self.T(u))
    def H(self, u):
        return self.p(u) / u[0] * self.gamma / (self.gamma - 1) + ((u[1]/u[0])**2 + (u[2]/u[0])**2)/2
    def M(self, u):
        return sqrt( ((u[1]/u[0])**2 + (u[2]/u[0])**2)) / self.c(u)
        
    def gradvel(self, u, q):
        if not self.Du:
            vel = self.vel(u)
            #u_x = 1/rho * [(rho*u)_x - rho_x * u)]
            u_x = 1/u[0] * (q[1,0] - q[0,0] * u[1]/u[0])
            #u_y = 1/rho * [(rho*u)_y - rho_y * u)]
            u_y = 1/u[0] * (q[1,1] - q[0,1] * u[1]/u[0])
            v_x = 1/u[0] * (q[2,0] - q[0,0] * u[2]/u[0])    
            v_y = 1/u[0] * (q[2,1] - q[0,1] * u[2]/u[0])
        else:
            u_x = q[0]
            u_y = q[1]
            v_x = q[1]
            v_y = q[2]
        return CoefficientFunction((u_x, u_y, v_x, v_y), dims=(2,2))

    def gradE(self, u, q):
        #E_x = 1/rho * [(rho*E)_x - rho_x*E]
        E_x = 1/u[0] * (q[3,0] - q[0,0] * u[3]/u[0])
        E_y = 1/u[0] * (q[3,1] - q[0,1] * u[3]/u[0])
        return CF((E_x, E_y))

    def gradT(self, u, q):
        # T = (self.gamma-1) * self.gamma * Minf**2(E - 1/2 (u**2 + v**2)
        # self.gamma * Minf**2 comes due to the non-dimensional NVS T = self.gamma Minf**2*p/rho
        # temp flux
        if not self.Du:
            vel = self.vel(u) 
            grad_vel = self.gradvel(u,q)
            u_x = grad_vel[0,0]
            v_x = grad_vel[1,0]
            u_y = grad_vel[0,1]
            v_y = grad_vel[1,1]

            E_x = self.gradE(u,q)[0]
            E_y = self.gradE(u,q)[1]
            
            # T_x = (self.gamma-1)*(self.gamma * self.Minf2) * (E_x - (u_x * vel[0] + v_x * vel[1]))
            # T_y = (self.gamma-1)*(self.gamma * self.Minf2) * (E_y - (u_y * vel[0] + v_y * vel[1]))
            T_x = (self.gamma-1)/self.R * (E_x - (u_x * vel[0] + v_x * vel[1]))
            T_y = (self.gamma-1)/self.R * (E_y - (u_y * vel[0] + v_y * vel[1]))
        else:
            T_x = q[3]
            T_y = q[4]
        return CF((T_x, T_y))

    
    def jacA(self, u):
        '''
        First Jacobian of the convective Euler Fluxes F_c = (f_c, g_c) for conservative variables U
        A = \partial f_c / \partial U
        input: u = (rho, rho * u, rho * E)
        See also Page 144 in C. Hirsch, Numerical Computation of Internal and External Flows: Vol.2 
        '''
        vel_1 = u[1]/u[0]
        vel_2 = u[2]/u[0]
        E = u[3]/u[0]
        return CoefficientFunction((0,1,0,0,
                                    (self.gamma-3)/2 * vel_1**2 + (self.gamma - 1)/2 * vel_2**2, (3-self.gamma) * vel_1, -(self.gamma-1)*vel_2, self.gamma - 1,
                                    -vel_1*vel_2, vel_2, vel_1, 0,
                                    -self.gamma*vel_1*E + (self.gamma-1)*vel_1 * (vel_1**2 + vel_2**2), self.gamma*E - (self.gamma-1)/2*(vel_2**2 + 3 * vel_1**2), -(self.gamma-1)*vel_1*vel_2, self.gamma * vel_1), dims= (4,4)) #.Compile()

    def jacB(self, u):
        '''
        Second Jacobian of the convective Euler Fluxes F_c = (f_c, g_c)for conservative variables U
        B = \partial g_c / \partial U
        input: u = (rho, rho * u, rho * E)
        See also Page 144 in C. Hirsch, Numerical Computation of Internal and External Flows: Vol.2 
        '''
        vel_1 = u[1]/u[0]
        vel_2 = u[2]/u[0]
        E = u[3]/u[0]
        return CoefficientFunction((0,0,1,0,                                
                                    -vel_1*vel_2, vel_2, vel_1, 0,
                                    (self.gamma-3)/2 * vel_2**2 + (self.gamma - 1)/2 * vel_1**2, -(self.gamma-1)*vel_1, (3-self.gamma) * vel_2, self.gamma - 1,
                                    -self.gamma*vel_2*E + (self.gamma-1)*vel_2 * (vel_1**2 + vel_2**2),  -(self.gamma-1)*vel_1*vel_2, self.gamma*E - (self.gamma-1)/2*(vel_1**2 + 3 * vel_2**2), self.gamma * vel_2), dims= (4,4)) #.Compile()

    # there holds for the Jacobian A = (jacA,jacB)
    # A \cdot n = P Lambda Pinv
    # P is the matrix where each column is a right eigenvector

    def P(self, u,k):
        rho = u[0]
        vel_1 = u[1]/u[0]
        vel_2 = u[2]/u[0]         
        p = self.p(u) 
        T = self.T(u) 
        E = u[3]/u[0]        
        c = self.c(u)
        H = self.H(u)

        # should be the same as:
        # H = (vel_1**2 + vel_2**2)/2 + c**2/(self.gamma-1) 
        
        # k is assumed to be normalized!
        kx = k[0] #/ sqrt(k*k)
        ky = k[1] #/ sqrt(k*k)

        return CoefficientFunction((1,0,rho/(2*c),rho/(2*c),                                
                                    vel_1,  rho * ky, rho/(2*c) * (vel_1 + c * kx), rho/(2*c) * (vel_1 - c * kx),
                                    vel_2, -rho * kx, rho/(2*c) * (vel_2 + c * ky), rho/(2*c) * (vel_2 - c * ky),
                                    (vel_1**2+vel_2**2)/2,  rho * (vel_1 * ky - vel_2 * kx), rho/(2*c) * (H + c * (vel_1 * kx + vel_2 * ky)), rho/(2*c) * (H - c * (vel_1 * kx + vel_2 * ky))), dims= (4,4)).Compile()

    def Pinv(self, u,k):
        rho = u[0]
        vel_1 = u[1]/u[0]
        vel_2 = u[2]/u[0]
        p = self.p(u) 
        T = self.T(u) 
        E = u[3]/u[0]        
        c = self.c(u)
        M = self.M(u)
    
        # k is assumed to be normalized!
        kx = k[0] #/  sqrt(k*k)
        ky = k[1] #/ sqrt(k*k)
        return CoefficientFunction((1 - (self.gamma-1)/2 * M**2,(self.gamma-1) * vel_1/(c**2),(self.gamma-1) * vel_2/(c**2),-(self.gamma-1)/(c**2),                                
                                1/u[0]*(vel_2 * kx - vel_1 * ky), ky/rho, -kx/rho,0,
                                c/rho*((self.gamma-1)/2*M**2 - (vel_1*kx + vel_2 * ky)/c), 1/rho * (kx - (self.gamma-1) *vel_1/c), 1/rho * (ky - (self.gamma-1) *vel_2/c), (self.gamma-1)/(rho * c),
                                c/rho*((self.gamma-1)/2*M**2 + (vel_1*kx + vel_2 * ky)/c), -1/rho * (kx + (self.gamma-1) *vel_1/c), -1/rho * (ky + (self.gamma-1) *vel_2/c), (self.gamma-1)/(rho * c)), dims= (4,4)).Compile()


    def Lam(self, u,k, ABS = False):
        vel_1 = u[1]/u[0]
        vel_2 = u[2]/u[0]
        p = self.p(u) 
        T = self.T(u) 
        c = self.c(u) 
        vk = vel_1 * k[0] + vel_2 * k[1]

        # k is assumed to be normalized!
        ck = c # * sqrt(k*k)
    
        vk_p_c = vk+ck
        vk_m_c = vk-ck
        
        # ABS = absolute value
        if (ABS==True):
            vk = sqrt(vk**2)
            vk_p_c = sqrt(vk_p_c**2)
            vk_m_c = sqrt(vk_m_c**2)
        
        return CoefficientFunction((vk,0,0,0,
                                    0,vk,0,0,
                                    0,0,vk_p_c,0,
                                    0,0,0,vk_m_c), dims = (4,4)).Compile()
                    
    def Aplus(self, u,k):
        return 0.5 * (self.P(u,k) * (self.Lam(u,k,False) + self.Lam(u,k,True)) * self.Pinv(u,k))  

    def Aminus(self, u,k):
        return 0.5 * (self.P(u,k) * (self.Lam(u,k,False) - self.Lam(u,k,True)) * self.Pinv(u,k))

    def Flux(self, u):
        m = CoefficientFunction((u[1],u[2]),dims=(2,1))
        p = self.p(u) # (self.gamma-1) * (u[3] - 0.5*InnerProduct(m,m)/u[0])
        return CoefficientFunction(tuple([m, 1/u[0] * m*m.trans + p*Id(2), 1/u[0] * (u[3]+p) * m]), 
                               dims = (4,2))

    def diffFlux(self, u, q):
        if not self.Du:
            grad_vel = self.gradvel(u,q)
            tau = self.mu/self.Re * (2 * (grad_vel+grad_vel.trans) - 2/3 * (grad_vel[0,0] + grad_vel[1,1]) * Id(2))
            grad_T = self.gradT(u,q)
        else:
            tau = self.mu/self.Re * CF((q[0], q[1], q[1], q[2]), dims = (2,2))
            grad_T = CF((q[3], q[4]))

        tau_vel = tau * self.vel(u) #CoefficientFunction((tau[0,0] * vel[0] + tau[0,1] * vel[1],tau[1,0] * vel[0] + tau[1,1] * vel[1]))

        #k = 1/((self.gamma-1) * self.Minf2 * self.Re*self.Pr)
        k = self.mu / (self.Re * self.Pr)

        return CoefficientFunction((0,0,
                                    tau[0,0],tau[0,1],
                                    tau[1,0],tau[1,1],
                                    tau_vel[0] + k*grad_T[0] , tau_vel[1] + k*grad_T[1]),dims = (4,2))


    def numFlux(self, uhatold, u,uhat,n):        
        #Lax-Friedrich flux
        return self.Flux(uhat)*n + self.c(uhat) * (u-uhat)  #self.Flux(uhat)*n + self.c(u) * (u-uhat)

    def numdiffFlux(self, u, uhat,q,n):
        #C = calcmaxspeed(uhat)
        # tau_d = 1 / ((self.gamma - 1) * self.Minf**2 * self.Pr) #self.mu/(self.Pr))
        C = CoefficientFunction(
                (0, 0, 0, 0,
                 0, 1/self.Re, 0, 0,
                 0, 0, 1/self.Re, 0,
                 0, 0, 0, 1/self.Re * self.tau_d), dims = (4, 4))

        if self.Du:
            return self.diffFlux(uhat,q)*n - C * (u-uhat)
        else:
            return self.diffFlux(uhat,q)*n

    def reflect(self,u,n):
        m = CoefficientFunction(tuple([u[i] for i in range(1,3)]),dims=(2,1))
        mn = InnerProduct(m,n)
        m -= mn*n
        # P = Id(2) - OuterProduct(n,n)
        return CoefficientFunction(tuple([u[0],m,u[3]]), dims = (4,1))

    @property
    def tau_d(self):
        # return 1 / ((self.gamma - 1) * self.Minf**2 * self.Pr)
        return 1 / (self.gamma * self.R * self.Minf**2 * self.Pr)