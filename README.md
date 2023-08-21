# Dream solver

This is a solver plugin for NGSolve (www.ngsolve.org) which can be used to solve the compressible Navier-Stokes equations (CNSE). The consider a "direct aeroacoustic simulation" using an Hybrid Discontinuous (HDG) Galerkin method. 

## Compressible Navier-Stokes equations
<!-- ### Notation and problem parameters -->
We consider the following quantities 

* $\rho$ ... density
* $p$ ... pressure
* $u = (u_x, u_y, u_z)$ ...velocity
* $E$ ...total energy
* $T$ ...temperature 
  
and aim to solve
$$
\frac{\partial \rho}{\partial t} + \operatorname{div}(\rho u) = 0, 
\\
\frac{\partial \rho u}{\partial t} + \operatorname{div}(\rho u  \otimes u) - \operatorname{div}(\tau) +\nabla{p} = \rho f, \\
\frac{\partial \rho E}{\partial t} + 
\operatorname{div}(\rho E u + p u) - 
\operatorname{div}(\tau u)  + \operatorname{div}( {q}) = 
 \rho f \cdot u.
$$
with the constitutive relations 
$$
p = \rho R T,
\quad
\tau = 2\mu(T) \varepsilon(u),
\quad
q = k(T) \nabla,
\quad 
\varepsilon(u) = \frac{1}{2}(\nabla u + (\nabla u)^T) - \frac{1}{3} \operatorname{div}(u)I,
$$
where $k(T)$ is a temperature dependent diffusivity and the viscosity is given by Sutherlands Law (with reference quantities $\mu_0, T_0$ and $S_0$),
$$
\mu(T) = \mu_0 \left(\frac{T}{T_0} \right)^\frac{3}{2} \frac{T_0 + S_0}{T + S_0}.
$$
Further we consider the thermodynamic relations $\gamma = c_p/c_v$ with the specific heats $c_v,c_p$ at constant volume and constant pressure, respectively and $\gamma R = c_p (\gamma - 1)$, where $R = c_p - c_v$ is the universal gas constant. The speed of sound is given by
$$
c = \sqrt{\gamma R T}.
$$

### Non-dimensionalization
We discuss the possible non-dimensionalizations that are (currently) available in the dream solver package.  In this section non-dimensional quantities are denoted with a superscript $^*$. For simplification we omit $^*$ in other sections. 
In the following let $\rho_\infty$, $u_\infty$, $T_\infty$ and $\text{Ma}_\infty$ be the free-stream (far-field) density, velocity, temperature and Mach number, respectively. Further we use a reference length $L$ and  a reference viscosity $\mu_\infty$.

#### Aerodynamic scaling
We consider a density-velocity-temperature dimensionalization, i.e. we choose the reference values $\rho_{ref} = \rho_\infty$, $u_{ref} = u_\infty$ and $T_{ref}=  T_\infty (\gamma - 1)\text{Ma}_\infty$ and the dimensionless variables
$$
{x}^* = \frac{x}{L}, \quad 
{t}^* = \frac{u_\infty t}{L}, \quad 
{{u}}^* = \frac{{u}}{u_\infty}, \\
{\mu}^*(T) = \frac{\mu(T)}{\mu_\infty}, \quad 
{\rho}^* = \frac{\rho}{\rho_\infty},  \quad 
{p}^* = \frac{p}{\rho_{\infty} u_{\infty}^2}, \\
{e}^* = \frac{e}{u_\infty^2}, \quad 
{T}^* = \frac{T}{T_\infty (\gamma - 1)\text{Ma}_\infty}, \quad 
{{q}}^* = \frac{{q} L}{\rho_{\infty} u_{\infty}^3}.
$$
With the dimensionless numbers 
$$
\text{Re}_\infty = \frac{\rho_\infty u_\infty L}{\mu_\infty}, \quad
\text{Ma}_\infty = \frac{u_\infty}{c_\infty} = \frac{u_\infty}{\sqrt{\gamma R T_\infty}}, \quad 
\text{Pr}:= \frac{c_p \mu}{k},
\quad   \text{Fr}_\infty := \sqrt{\frac{u_\infty^2}{gL}},
$$
i.e. the Reynolds, Mach, Prandtl and Frode number, 
this results in the following set of equations
$$
\frac{\partial \rho^*}{\partial t^*} + \operatorname{div^*}(\rho^* u^*) = 0, 
\\
\frac{\partial \rho^* u^*}{\partial t^*} + \operatorname{div^*}(\rho^* u^*  \otimes u^*) - \frac{1}{\text{Re}}\operatorname{div^*}(\tau^*) +\nabla{p^*} = \frac{1}{\text{Fr}^2} \rho^* f^*, \\
\frac{\partial \rho^* E^*}{\partial t^*} + 
\operatorname{div^*}(\rho^* E^* u^* + p^* u^*) - 
\frac{1}{\text{Re}}\operatorname{div^*}(\tau^* u^*)  + 
\frac{1}{\text{Re}\text{Pr}} \operatorname{div^*}( {q^*}) = 
 \rho^* f^* \cdot u^*.
$$
with
$$
\gamma p^* = (\gamma - 1) \rho^* T^*, \quad E^* = \frac{T^*}{\gamma} + \frac{| u^*|^2}{2}, \quad c = \sqrt{(\gamma-1)T}.
$$

#### Aeroacoustic scaling

# Hybrid discontinuous Galerkin approximation



# Funding  

We thank the Austrian Science Fund (FWF) for funding via the stand alone project P35391N.

Developed at:
* (2022 - ) TU Wien, Institute of Analysis and Scientific Computing
* (2023 - ) University of Twente, Institute of Applied Mathematics







