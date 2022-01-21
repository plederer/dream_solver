from ngsolve import *

import math
from math import pi, atan2

class FlowUtils():
    def __init__(self, ff_data):
        self.Re = ff_data["Re"]
        self.Pr = ff_data["Pr"]
        self.mu = ff_data["mu"]
        self.Minf = ff_data["Minf"]
        self.Minf2 = self.Minf**2
        self.gamma = ff_data["gamma"]
      

    def jacA(self, u):
        vel_1 = u[1]/u[0]
        vel_2 = u[2]/u[0]
        E = u[3]/u[0]
        return CoefficientFunction((0,1,0,0,
                                    (self.gamma-3)/2 * vel_1**2 + (self.gamma - 1)/2 * vel_2**2, (3-self.gamma) * vel_1, -(self.gamma-1)*vel_2, self.gamma - 1,
                                    -vel_1*vel_2, vel_2, vel_1, 0,
                                    -self.gamma*vel_1*E + (self.gamma-1)*vel_1 * (vel_1**2 + vel_2**2), self.gamma*E - (self.gamma-1)/2*(vel_2**2 + 3 * vel_1**2), -(self.gamma-1)*vel_1*vel_2, self.gamma * vel_1), dims= (4,4)) #.Compile()

    def jacB(self, u):
        vel_1 = u[1]/u[0]
        vel_2 = u[2]/u[0]
        E = u[3]/u[0]
        return CoefficientFunction((0,0,1,0,                                
                                    -vel_1*vel_2, vel_2, vel_1, 0,
                                    (self.gamma-3)/2 * vel_2**2 + (self.gamma - 1)/2 * vel_1**2, -(self.gamma-1)*vel_1, (3-self.gamma) * vel_2, self.gamma - 1,
                                    -self.gamma*vel_2*E + (self.gamma-1)*vel_2 * (vel_1**2 + vel_2**2),  -(self.gamma-1)*vel_1*vel_2, self.gamma*E - (self.gamma-1)/2*(vel_1**2 + 3 * vel_2**2), self.gamma * vel_2), dims= (4,4)) #.Compile()

    #there holds for the Jacobian A = (jacA,jacB)
    # A \cdot n = P Lambda Pinv

    def P(self, u,k):
        rho = u[0]
        vel_1 = u[1]/u[0]
        vel_2 = u[2]/u[0]         
        p = (self.gamma-1) * (u[3] - u[0]/2 * (vel_1**2 + vel_2**2)) # p = rho*E - rho/2 * ||v||**2    
        T = p / u[0] * self.gamma * self.Minf2
        E = u[3]/u[0]
    
    
        c = sqrt(self.gamma *T)
        H = (vel_1**2 + vel_2**2)/2 + c**2/(self.gamma-1)
        #M = sqrt( (vel_1**2 + vel_2**2)) / c
        
        kx = k[0] / sqrt(k*k)
        ky = k[1] / sqrt(k*k)
        return CoefficientFunction((1,0,rho/(2*c),rho/(2*c),                                
                                    vel_1,  rho * ky, rho/(2*c) * (vel_1 + c * kx), rho/(2*c) * (vel_1 - c * kx),
                                    vel_2, -rho * kx, rho/(2*c) * (vel_2 + c * ky), rho/(2*c) * (vel_2 - c * ky),
                                    (vel_1**2+vel_2**2)/2,  rho * (vel_1 * ky - vel_2 * kx),rho/(2*c) * (H + c * (vel_1 * kx + vel_2 * ky)),rho/(2*c) * (H - c * (vel_1 * kx + vel_2 * ky))), dims= (4,4)).Compile()

    def Pinv(self, u,k):
        rho = u[0]
        vel_1 = u[1]/u[0]
        vel_2 = u[2]/u[0]
        p = (self.gamma-1) * (u[3] - u[0]/2 * (vel_1**2 + vel_2**2)) # p = rho*E - rho/2 * ||v||**2    
        T = p / u[0] * self.gamma * self.Minf2
        E = u[3]/u[0]
        #H = E + p/u[0]
        c = sqrt(self.gamma *T)
        M = sqrt( (vel_1**2 + vel_2**2)) / c
    
        kx = k[0] / sqrt(k*k)
        ky = k[1] / sqrt(k*k)
        return CoefficientFunction((1 - (self.gamma-1)/2 * M**2,(self.gamma-1) * vel_1/(c**2),(self.gamma-1) * vel_2/(c**2),-(self.gamma-1)/(c**2),                                
                                1/u[0]*(vel_2 * kx - vel_1 * ky), ky/rho, -kx/rho,0,
                                c/rho*((self.gamma-1)/2*M**2 - (vel_1*kx + vel_2 * ky)/c), 1/rho * (kx - (self.gamma-1) *vel_1/c), 1/rho * (ky - (self.gamma-1) *vel_2/c), (self.gamma-1)/(rho * c),
                                c/rho*((self.gamma-1)/2*M**2 + (vel_1*kx + vel_2 * ky)/c), -1/rho * (kx + (self.gamma-1) *vel_1/c), -1/rho * (ky + (self.gamma-1) *vel_2/c), (self.gamma-1)/(rho * c)), dims= (4,4)).Compile()


    def Lam(self, u,k, ABS = False):
        vel_1 = u[1]/u[0]
        vel_2 = u[2]/u[0]
        p = (self.gamma-1) *  (u[3] - u[0]/2 * (vel_1**2 + vel_2**2)) # p = rho*E - rho/2 * ||v||**2    

        T = p / u[0] * self.gamma * self.Minf2
        #E = u[3]/u[0]
        #H = E + p/u[0]
        c = sqrt(self.gamma *T)

        vk = vel_1 * k[0] + vel_2 * k[1]

        ck = c * sqrt(k*k)
    
        vk_p_c = vk+ck
        vk_m_c = vk-ck

    
        if (ABS==True):
            vk = sqrt(vk**2)
            vk_p_c = sqrt(vk_p_c**2)
            vk_m_c = sqrt(vk_m_c**2)
        
        return CoefficientFunction((vk,0,0,0,
                                    0,vk,0,0,
                                    0,0,vk_p_c,0,
                                    0,0,0,vk_m_c), dims = (4,4)).Compile()
                    
    def Aplus(self, u,k):
        return 0.5 * (P(u,k) * (Lam(u,k,False) + Lam(u,k,True)) * Pinv(u,k))  

    def Aminus(self, u,k):
        return 0.5 * (P(u,k) * (Lam(u,k,False) - Lam(u,k,True)) * Pinv(u,k))

    def Flux(self, u):
        m = CoefficientFunction((u[1],u[2]),dims=(2,1))
        p = (self.gamma-1) * (u[3] - 0.5*InnerProduct(m,m)/u[0])
        return CoefficientFunction(tuple([m, 1/u[0] * m*m.trans + p*Id(2), 1/u[0] * (u[3]+p) * m]), 
                               dims = (4,2))

    def diffFlux(self, u,q):
        vel = CoefficientFunction((u[1]/u[0],u[2]/u[0]))

        #u_x = 1/rho * [(rho*u)_x - rho_x * u)]
        u_x = 1/u[0] * (q[1,0] - q[0,0] * u[1]/u[0])
        #u_y = 1/rho * [(rho*u)_y - rho_y * u)]
        u_y = 1/u[0] * (q[1,1] - q[0,1] * u[1]/u[0])

        v_x = 1/u[0] * (q[2,0] - q[0,0] * u[2]/u[0])    
        v_y = 1/u[0] * (q[2,1] - q[0,1] * u[2]/u[0])

        grad_vel = CoefficientFunction((u_x,u_y,v_x,v_y), dims = (2,2))

        #scaled viscosity = 1
        tau = 1/self.Re * ((grad_vel+grad_vel.trans) - 2/3 * (u_x + v_y) * Id(2))

        #E_x = 1/rho * [(rho*E)_x - rho_x*E]
        E_x = 1/u[0] * (q[3,0] - q[0,0] * u[3]/u[0])
        E_y = 1/u[0] * (q[3,1] - q[0,1] * u[3]/u[0])

        # T = (self.gamma-1) * self.gamma * Minf**2(E - 1/2 (u**2 + v**2)
        # self.gamma * Minf**2 comes due to the non-dimensional NVS T = self.gamma Minf**2*p/rho
        #temp flux
        T_x = (self.gamma-1)*(self.gamma * self.Minf2) * (E_x - (u_x * vel[0] + v_x * vel[1]))
        T_y = (self.gamma-1)*(self.gamma * self.Minf2) * (E_y - (u_y * vel[0] + v_y * vel[1]))
        k = 1/((self.gamma-1) * self.Minf2 * self.Re*self.Pr)

        tau_vel = CoefficientFunction((tau[0,0] * vel[0] + tau[0,1] * vel[1],tau[1,0] * vel[0] + tau[1,1] * vel[1]))

        #conducticity in non-dimensional equation

        return CoefficientFunction((0,0,
                                    -tau[0,0],-tau[0,1],
                                    -tau[1,0],-tau[1,1],
                                    -tau_vel[0] - k*T_x ,- tau_vel[1] - k*T_y),dims = (4,2))

    def calcmaxspeed(self,u):
        vel_1 = u[1]/u[0]
        vel_2 = u[2]/u[0]
        m = CoefficientFunction((u[1],u[2]),dims=(2,1))
       
        p = (self.gamma-1) * (u[3] - 0.5*InnerProduct(m,m)/u[0])
    
        #T = 2 * p / u[0]
        return sqrt(self.gamma*p/u[0] * self.gamma * self.Minf2)

    def numFlux(self, uhatold, u,uhat,n):
        C = self.calcmaxspeed(uhat)
        #Lax-Friedrich flux
        return self.Flux(uhat)*n + C * (u-uhat)

    def numdiffFlux(self, u,uhat,q,n):
        #C = calcmaxspeed(uhat)
        C = CoefficientFunction((0,0,0,0,
                                 0,1,0,0,
                                 0,0,1,0,
                                 0,0,0,1/((self.gamma-1)*self.Minf2 * self.Re * self.Pr)), dims = (4,4))
    
        #Lax-Friedrich flux
        return self.diffFlux(uhat,q)*n + C * (u-uhat)


    def reflect(self,u,n):
        m = CoefficientFunction(tuple([u[i] for i in range(1,3)]),dims=(2,1))
        mn = InnerProduct(m,n)
        m -= mn*n
        #P = Id(2) - OuterProduct(n,n)
        return CoefficientFunction(tuple([u[0],m,u[3]]), dims = (4,1))
