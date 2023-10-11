# Considered equations

## Compressible Navier-Stokes equations

We consider the following quantities 

* $\rho$ ... density
* $p$ ... pressure
* $\VEL = (u_x, u_y, u_z)$ ...velocity
* $E$ ...total energy
* $\HEAT$ ...heat flux

 
and aim to solve

$$
    \begin{align*}
    \pdt{\rho} + \div{(\rho \VEL)} &= 0, \\
    \pdt{(\rho \VEL)} + \div{(\rho \VEL \otimes \VEL)} &= \div{\TAU} -\grad{p} + \rho \vec{f}, \\
    \pdt{(\rho E)} + \div{(\rho E \VEL)} &= \div{(\TAU \VEL)} - \div{(p \VEL)} + \rho \vec{f} \cdot \VEL - \div{\HEAT}.
    \end{align*}
$$

We define the specific energy $E$ as the sum of the specific kinetic energy $E_k$ and the specific inner energy $E_i$, hence leading to the following expression

$$
    \begin{align*}
    E &= E_i + E_k =  E_i + \tfrac{1}{2} |\VEL|^2
    \end{align*}
$$

# Constitutive Relations


For the time being we assume the simplest equation of state, namely the ideal gas law

$$
    \begin{align*}
    p &= \rho R T,
    \end{align*}
$$

which holds for a **calorically perfect gas**. In this special setting additional equations hold

$$
    \begin{align*}
    E_i &= c_v T, & \frac{c_p}{c_v} &= \gamma, & R = c_p - c_v. 
    \end{align*}
$$

As for the deviatoric stress tensor $\TAU$ we assume a Newtonian relationship between the stresses $\TAU$ and the rate-of-strain $\EPS$ 

$$
    \begin{align*}
    \TAU &= 2 \mu(T) \EPS = 2 \mu(T) \left[ \frac{\grad{\VEL} + (\grad{\VEL})^{\T}}{2}  - \frac{1}{3} \div \VEL \I \right],
    \end{align*}
$$

in which the viscosity $\mu$ is either constant or temperature dependent, given by *Sutherland's law*, respectively

$$
    \begin{align*}
    \mu(T) &:= \mu_0, & & & \mu(T) &:= \mu_0 \left(\frac{T}{T_0} \right)^\frac{3}{2} \frac{T_0 + S_0}{T + S_0}.
    \end{align*}
$$

As for the heatflux *Fourier's law* is considered

$$
    \begin{align*}
    \HEAT &= -k(T)  \grad T
    \end{align*}
$$


## Dimensionless Compressible Equations

We solve the Navier-Stokes equations in a dimensionless form. By this
approach settings with equal dimensionless numbers, such as *Reynolds
number*, *Mach number*, etc. produce equal results. From dimensional
analysis (*Buckingham*-$\pi$-*theorem*) one has to prescribe four
primal quantities in order to derive a dimensionless version of the
compressible Navier-Stokes equations. We opt for a
density-velocity-temperature dimensionalisation and introduce three
different scaling options:
* **Aerodynamic**

\begin{align*}
\Lref &= L, & \rhoref &= \rhoinf,  & \uref &= \uinf, & \Tref &= \Tinf (\gamma - 1) \Ma_\infty^2
\end{align*}


* **Acoustic**

\begin{align*}
\Lref &= L, & \rhoref &= \rhoinf,  & \uref &= \cinf, & \Tref &= \Tinf (\gamma - 1)
\end{align*}


* **Aeroacoustic**

\begin{align*}
\Lref &= L, & \rhoref &= \rhoinf,  & \uref &= \cinf (1 + \Ma_\infty), & \Tref &= \Tinf (\gamma - 1) (1+\Ma_\infty)^2
\end{align*}

 
The infinity subscript $_\infty$ denotes the free stream values of the flow. Note, that following relationship holds for all three scalings

\begin{align*}
\frac{\Tref}{\uref^2} &= \frac{\gamma - 1}{\gamma R} = \frac{1}{c_p}
\end{align*}


At this point it is possible to define dimensionless variables

\begin{align*}
\dl{x} &= \frac{x}{\Lref}, &  
\dl{t} &= \frac{t \uref}{\Lref}, & 
\dl{\VEL} &= \frac{\VEL}{\uref}, \\
\dl{\rho} &= \frac{\rho}{\rhoref}, & 
\dl{p} &= \frac{p}{\rhoref \uref^2}, &
\dl{T} &= \frac{T}{\Tref}, \\
\end{align*}

and the more important dimensionless numbers
* **Reynolds number**
  \begin{align*} 
  \Re_\infty := \frac{\rhoinf \uinf \Lref}{\mu_\infty} 
  \end{align*}

  * **Mach number**
  \begin{align*}\Ma_\infty := \frac{\uinf}{\cinf} = \frac{\uinf}{\sqrt{\gamma R \Tinf}}  \end{align*}

  * **Prandtl number**
  \begin{align*}\Pr_\infty := \frac{c_p \muinf}{k_\infty} \end{align*}

  * **Froude number**
  \begin{align*} \Fr_\infty := \sqrt{\frac{\uinf^2}{gL}} \end{align*}

Note, that for air one usually sets $\Pr_\infty = 0.72$, as the Prandtl number is almost constant over a broad temperature range.

### Derivation
* **Continuity Equation**
  \begin{align*}
  \pdt{\rho} + \div{(\rho \VEL)} &= 0, \\
  \frac{\rhoref \uref}{\Lref} \pdtdl{\dl{\rho}} + \frac{\rhoref \uref}{\Lref} \dl{\div}(\dl{\rho} \dl{\VEL}) &= 0, \\
  \pdtdl{\dl{\rho}} + \dl{\div}(\dl{\rho} \dl{\VEL}) &= 0.
  \end{align*}
  

* **Momentum Equation**
  \begin{align*}
  \pdt{(\rho \VEL)} + \div{(\rho \VEL \otimes \VEL + p \I)} &= \div{\TAU} + \rho \vec{f}, \\
  \frac{\rhoref \uref^2}{\Lref}  \pdtdl{(\dl{\rho} \dl{\VEL})} + 
  \frac{\rhoref \uref^2}{\Lref} \dl{\div}(\dl{\rho} \dl{\VEL} \otimes \dl{\VEL} + \dl{p} \I) &= 
  \frac{\muref \uref}{\Lref^2} \dl{\div}\dl{\TAU} + 
  \rhoref g \dl{\rho} \dl{\vec{f}}, \\
  \pdtdl{(\dl{\rho} \dl{\VEL})} + 
  \dl{\div}(\dl{\rho} \dl{\VEL} \otimes \dl{\VEL} + \dl{p} \I) &= 
  \frac{\muref}{\Lref \rhoref \uref} \dl{\div}\dl{\TAU} + \frac{g \Lref}{\uref^2} \dl{\rho} \dl{\vec{f}}, \\
  \pdtdl{(\dl{\rho} \dl{\VEL})} + 
  \dl{\div}(\dl{\rho} \dl{\VEL} \otimes \dl{\VEL} + \dl{p} \I) &= 
  \frac{1}{\Re_{\text{ref}}} \dl{\div}\dl{\TAU} + \frac{1}{\Fr^2_{\text{ref}}} \dl{\rho} \dl{\vec{f}}.
  \end{align*}
  

* **Energy Equation**
  \begin{align*}
  \pdt{(\rho E)} + \div{([\rho E + p] \VEL)} &= \div{(\TAU \VEL)} - \div{\HEAT} + \rho \vec{f} \cdot \VEL , \\
  \frac{\rhoref \uref^3}{\Lref}\pdtdl{(\dl{\rho} \dl{E})} + 
  \frac{\rhoref \uref^3}{\Lref} \dl{\div}([\dl{\rho} \dl{E} + \dl{p}] \dl{\VEL}) &= 
  \frac{\muref \uref^2}{\Lref^2}\dl{\div}(\dl{\TAU}) \dl{\VEL} -  
  \frac{\Tref \kref}{\Lref^2} \frac{\muref}{\muref} \dl{\div}{\dl{\HEAT}} + 
  \rhoref g \uref \dl{\rho} \dl{\vec{f}} \cdot \dl{\VEL} , \\
  \pdtdl{(\dl{\rho} \dl{E})} + 
  \dl{\div}([\dl{\rho} \dl{E} + \dl{p}] \dl{\VEL}) &= 
  \frac{\muref}{\Lref \rhoref \uref} \dl{\div}(\dl{\TAU} \dl{\VEL}) - 
  \frac{\muref}{\rhoref \uref \Lref} \frac{\Tref \kref}{\uref^2 \muref} \dl{\div}{\dl{\HEAT}} + 
  \frac{g \Lref}{\uref^2} \dl{\rho} \dl{\vec{f}} \cdot \dl{\VEL}, \\
  \pdtdl{(\dl{\rho} \dl{E})} + 
  \dl{\div}([\dl{\rho} \dl{E} + \dl{p}] \dl{\VEL}) &= 
  \frac{1}{\Re_{\text{ref}}}\dl{\div}(\dl{\TAU} \dl{\VEL}) -
  \frac{1}{\Re_{\text{ref}} \Pr_{\text{ref}}} \dl{\div}\dl{\HEAT} + 
  \frac{1}{\Fr^2_{\text{ref}}} \dl{\rho} \dl{\vec{f}} \cdot \dl{\VEL}.
  \end{align*}
  

* **Ideal Gas Law**
  \begin{align*}
  p &= \rho R T, \\
  \rhoref \uref^2 \dl{p} &= \rhoref \Tref R \dl{\rho} \dl{T}, \\
  \dl{p} &= \frac{\Tref R}{\uref^2} \dl{\rho} \dl{T}, \\
  \gamma \dl{p} &= (\gamma -1) \dl{\rho} \dl{T}.
  \end{align*}
  

* **Newtonian Law** 
  \begin{align*}
  \TAU &= 2 \mu(T) \EPS, \\
  \TAU &= 2 \frac{\muref \uref}{\Lref} \dl{\mu}(\dl{T}) \dl{\EPS}, \\
  \dl{\TAU} = \frac{\Lref}{\muref \uref} \TAU &= 2 \dl{\mu}(\dl{T}) \dl{\EPS}.
  \end{align*}
  

* ***Fourier's Law**
  \begin{align*}
  \HEAT &= -k(T)  \grad T, \\
  \HEAT &= - \frac{\Tref \kref}{\Lref} \dl{k}(\dl{T}) \dl{\grad}\dl{T}, \\
  \dl{\HEAT } =  \frac{\Lref}{\Tref \kref} \frac{\muref}{\muref} \HEAT &= -\dl{k}(\dl{T}) \dl{\grad}\dl{T}.
  \end{align*}
  

* **Sutherland's Law**
  \begin{align*}
  \mu(T) &= \mu_0 \left(\frac{T}{T_0} \right)^\frac{3}{2} \frac{T_0 + S_0}{T + S_0}, \\
  \dl{\mu}(T)  &= \frac{\mu_0}{\muref} \left(\frac{T}{T_0} \right)^\frac{3}{2} \frac{T_0 + S_0}{T + S_0}, \\
  \dl{\mu}(\Tinf)  = 1 \rightarrow \frac{\muref}{\mu_0} &=  \left(\frac{\Tinf}{T_0} \right)^\frac{3}{2} \frac{T_0 + S_0}{\Tinf + S_0}, \\
  \dl{\mu}(T)  &= \left(\frac{T_0}{\Tinf} \right)^\frac{3}{2} \frac{\Tinf + S_0}{T_0 + S_0} \left(\frac{T}{T_0} \right)^\frac{3}{2} \frac{T_0 + S_0}{T + S_0}, \\  
  \dl{\mu}(T)  &= \left(\frac{T}{\Tinf} \right)^\frac{3}{2} \frac{\Tinf + S_0}{T + S_0}, \\
  \dl{\mu}(\dl{T})  &= \left(\frac{\dl{T}}{\dl{\Tinf}} \right)^\frac{3}{2}  \frac{\dl{\Tinf} + \dl{S_0}}{\dl{T} + \dl{S_0}}.
  \end{align*}
  
  With $\dl{\Tinf} = \frac{\Tinf}{\Tref}$ and $\dl{S_0} = \frac{S_0}{\Tref}$.

  For sake of simplicity from now on we will omit the asterisk and consider only the dimensionless compressible equations. Last, but not least the difference scalings introduce different dimensionless reference numbers. In terms of freestream variables they are given by:
* **Aerodynamic**
   \begin{align*}
    \Re_{\text{ref}} &= \Re_{\infty} & \Pr_{\text{ref}} &= \Pr_{\infty},
   \end{align*}
   

* **Acoustic**
   \begin{align*}
    \Re_{\text{ref}} &=  \frac{\Re_{\infty}}{\Ma_\infty} & \Pr_{\text{ref}} &= \Pr_{\infty},
   \end{align*}
   
  
* **Aeroacoustic**
   \begin{align*}
    \Re_{\text{ref}} &= \frac{\Re_{\infty}}{\Ma_\infty} (1 + \Ma_\infty) & \Pr_{\text{ref}} &= \Pr_{\infty}.
   \end{align*}
   


<!-- 
Non-dimensionalization
######################

We discuss the possible non-dimensionalizations that are (currently)
available in the dream solver package.  In this section
non-dimensional quantities are denoted with a superscript :math:`^*`. For
simplification we omit :math:`^*` in other sections. In the following let
:math:`\rho_\infty`, :math:`u_\infty`, :math:`T_\infty` and
:math:`\text{Ma}_\infty` be the free-stream (far-field) density,
velocity, temperature and Mach number, respectively. Further we use a
reference length :math:`L` and  a reference viscosity :math:`\mu_\infty`.

Aerodynamic scaling
*******************

We consider a density-velocity-temperature dimensionalization, i.e. we
choose the reference values :math:`\rho_{ref} = \rho_\infty`, :math:`u_{ref} =u_\infty` 
and :math:`T_{ref}=  T_\infty (\gamma - 1)\text{Ma}_\infty` and
the dimensionless variables

.. math::

    {x}^* &= \frac{x}{L}, \quad &
    {t}^* &= \frac{u_\infty t}{L}, \quad &
    {{u}}^* &= \frac{{u}}{u_\infty}, \\
    {\mu}^*(T) &= \frac{\mu(T)}{\mu_\infty}, \quad &
    {\rho}^* &= \frac{\rho}{\rho_\infty},  \quad &
    {p}^* &= \frac{p}{\rho_{\infty} u_{\infty}^2}, \\
    {e}^* &= \frac{e}{u_\infty^2}, \quad &
    {T}^* &= \frac{T}{T_\infty (\gamma - 1)\text{Ma}_\infty}, \quad &
    {{q}}^* &= \frac{{q} L}{\rho_{\infty} u_{\infty}^3}.

With the dimensionless numbers 

.. math::

    \text{Re}_\infty = \frac{\rho_\infty u_\infty L}{\mu_\infty}, \quad
    \text{Ma}_\infty = \frac{u_\infty}{c_\infty} = \frac{u_\infty}{\sqrt{\gamma R T_\infty}}, \quad 
    \text{Pr}:= \frac{c_p \mu}{k},
    \quad   \text{Fr}_\infty := \sqrt{\frac{u_\infty^2}{gL}},

i.e. the Reynolds, Mach, Prandtl and Frode number, 
this results in the following set of equations

.. math::

    \frac{\partial \rho^*}{\partial t^*} + {\text{div}}^*(\rho^* u^*) &= 0, 
    \\
    \frac{\partial \rho^* u^*}{\partial t^*} + {\text{div}}^*(\rho^* u^*  \otimes u^*) - \frac{1}{\text{Re}}{\text{div}}^*(\tau^*) +\nabla{p^*} &= \frac{1}{\text{Fr}^2} \rho^* f^*, \\
    \frac{\partial \rho^* E^*}{\partial t^*} + 
    {\text{div}}^*(\rho^* E^* u^* + p^* u^*) - 
    \frac{1}{\text{Re}}{\text{div}}^*(\tau^* u^*)  + 
    \frac{1}{\text{Re}\text{Pr}} {\text{div}}^*( {q^*}) &= 
    \rho^* f^* \cdot u^*.


with :math:`\tau^* =  2 \mu^*(T)\Big(\varepsilon^*(u^*) - 1/3 {\text{div}}^*(u^*)\Big)` and :math:`q^* = \mu^*(T) \nabla^* T^*` and 

.. math::

    \gamma p^* = (\gamma - 1) \rho^* T^*, \quad E^* = \frac{T^*}{\gamma} + \frac{| u^*|^2}{2}, \quad c = \sqrt{(\gamma-1)T}.


Aeroacoustic scaling
********************

.. math::

    \pdt{\rho} = 0 -->
