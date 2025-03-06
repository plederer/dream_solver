The fundamental equations describing the motion of an unsteady, viscous and compressible flow in a space-time cylinder
$\Omega \times (0, t_{end}] \in \mathbb{R}^{d+1}$ with non-empty bounded $d$-dimensional spatial domain $\Omega$, with
boundary $\partial \Omega$, and final time $t_{end}$, are specified by the Navier-Stokes equations. In terms of
conservative variables, $\vec{U} = \begin{pmatrix} \rho, & \rho \vec{u}, & \rho E \end{pmatrix}^\T$, with density $\rho$, velocity $\vec{u}$, and total specific energy $E$,
this system can be expressed in dimensionless form as

$$
\pdt{\vec{U}} + \div(\mat{F}(\vec{U}) - \mat{G}(\vec{U}, \grad \vec{U})) = \vec{0} \quad \text{in} \quad \Omega \times (0, t_{end}],
$$

by applying a proper scaling, see 
<!-- 
$$
        \rhoref & := \rho^*_\infty, \qquad & \uref & := |\VEL^*_\infty|, \qquad & \Tref & := \theta^*_\infty (\gamma - 1) \Ma^2_\infty,  \qquad &  & \text{(aerodynamic)} \\
        \label{eq::acoustic-scaling}
        \rhoref & := \rho^*_\infty, \qquad & \uref & := c^*_\infty, \qquad      & \Tref & := \theta^*_\infty (\gamma - 1), \qquad               &  & \text{(acoustic)}
$$

depending on the flow regime under consideration, where $\theta$ and $c$ denote the temperature and speed of sound, respectively.
With the Mach number $\Mainf$, the Reynolds number $\Reinf$ and the Prandtl number $\Prinf$ defined as
\begin{align*}
    \Mainf & := \frac{|\VEL^*_\infty|}{c^*_\infty}, & \Reinf & := \frac{\uref \rho^*_\infty L^*}{\mu^*_\infty}, & \Prinf & := \frac{\mu^*_\infty c_p}{\kappa^*_\infty},
\end{align*}
the convective and the viscous fluxes are given by
\begin{align*}
    \F(\CVEC)              & = \begin{pmatrix}
                                   \rho \VEL^\T                  \\[0.8ex]
                                   \rho \VEL \otimes \VEL + p \I \\[0.8ex]
                                   \rho H \VEL^\T
                               \end{pmatrix},      &
    \G(\CVEC, \grad \CVEC) & = \frac{1}{\Reinf}\begin{pmatrix}
                                                   \vec{0}^\T \\[0.8ex]
                                                   \TAU       \\[0.8ex]
                                                   (\TAU \VEL - \frac{\HEAT}{\Prinf})^\T
                                               \end{pmatrix},
\end{align*}
respectively, where $p$ denotes the pressure, $H = E + p/\rho$ the total specific enthalpy,
$\TAU$ the deviatoric stress tensor, $\HEAT$ the heat flux, $\mu$ the dynamic viscosity,
$c_p$ the specific heat capacity, $\kappa$ the thermal conductivity and $L$ a characteristic length.
Assuming ideal gas conditions, Newtonian fluid and Fourier's law, the necessary
constitutive relations to close the system of equations \eqref{eq::Navier-Stokes} are
\begin{subequations}
    \begin{align}
        \gamma p & = \rho (\gamma - 1) \theta,                                                         \\
        \TAU     & = 2 \left[ \frac{\grad \VEL + (\grad \VEL)^\T}{2} - \frac{1}{3} \div(\VEL) \right], \\
        \HEAT    & = -\grad \theta,
    \end{align}
\end{subequations}
with heat capacity ratio $\gamma=1.4$ for air. In this setting the isentropic speed of sound $c$
is defined as
$$
    c = \sqrt{\gamma \frac{p}{\rho}} = \sqrt{(\gamma - 1) \theta}.
$$ -->

<!-- We consider the following quantities 

* $\rho$ ... density
* $p$ ... pressure
* $\vec{u} = (u_x, u_y, u_z)$ ...velocity
* $E$ ...total energy
* $\HEAT$ ...heat flux

 
and aim to solve

$$
    \begin{align*}
    \pdt{\rho} + \div{(\rho \vec{u})} &= 0, \\
    \pdt{(\rho \vec{u})} + \div{(\rho \vec{u} \otimes \vec{u})} &= \div{\TAU} -\grad{p} + \rho \vec{f}, \\
    \pdt{(\rho E)} + \div{(\rho E \vec{u})} &= \div{(\TAU \vec{u})} - \div{(p \vec{u})} + \rho \vec{f} \cdot \vec{u} - \div{\HEAT}.
    \end{align*}
$$

We define the specific energy $E$ as the sum of the specific kinetic energy $E_k$ and the specific inner energy $E_i$, hence leading to the following expression

$$
    \begin{align*}
    E &= E_i + E_k =  E_i + \tfrac{1}{2} |\vec{u}|^2
    \end{align*}
$$

## Constitutive Relations


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
    \TAU &= 2 \mu(T) \EPS = 2 \mu(T) \left[ \frac{\grad{\vec{u}} + (\grad{\vec{u}})^{\T}}{2}  - \frac{1}{3} \div \vec{u} \I \right],
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
\dl{\vec{u}} &= \frac{\vec{u}}{\uref}, \\
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
  \pdt{\rho} + \div{(\rho \vec{u})} &= 0, \\
  \frac{\rhoref \uref}{\Lref} \pdtdl{\dl{\rho}} + \frac{\rhoref \uref}{\Lref} \dl{\div}(\dl{\rho} \dl{\vec{u}}) &= 0, \\
  \pdtdl{\dl{\rho}} + \dl{\div}(\dl{\rho} \dl{\vec{u}}) &= 0.
  \end{align*}
  

* **Momentum Equation**
  \begin{align*}
  \pdt{(\rho \vec{u})} + \div{(\rho \vec{u} \otimes \vec{u} + p \I)} &= \div{\TAU} + \rho \vec{f}, \\
  \frac{\rhoref \uref^2}{\Lref}  \pdtdl{(\dl{\rho} \dl{\vec{u}})} + 
  \frac{\rhoref \uref^2}{\Lref} \dl{\div}(\dl{\rho} \dl{\vec{u}} \otimes \dl{\vec{u}} + \dl{p} \I) &= 
  \frac{\muref \uref}{\Lref^2} \dl{\div}\dl{\TAU} + 
  \rhoref g \dl{\rho} \dl{\vec{f}}, \\
  \pdtdl{(\dl{\rho} \dl{\vec{u}})} + 
  \dl{\div}(\dl{\rho} \dl{\vec{u}} \otimes \dl{\vec{u}} + \dl{p} \I) &= 
  \frac{\muref}{\Lref \rhoref \uref} \dl{\div}\dl{\TAU} + \frac{g \Lref}{\uref^2} \dl{\rho} \dl{\vec{f}}, \\
  \pdtdl{(\dl{\rho} \dl{\vec{u}})} + 
  \dl{\div}(\dl{\rho} \dl{\vec{u}} \otimes \dl{\vec{u}} + \dl{p} \I) &= 
  \frac{1}{\Re_{\text{ref}}} \dl{\div}\dl{\TAU} + \frac{1}{\Fr^2_{\text{ref}}} \dl{\rho} \dl{\vec{f}}.
  \end{align*}
  

* **Energy Equation**
  \begin{align*}
  \pdt{(\rho E)} + \div{([\rho E + p] \vec{u})} &= \div{(\TAU \vec{u})} - \div{\HEAT} + \rho \vec{f} \cdot \vec{u} , \\
  \frac{\rhoref \uref^3}{\Lref}\pdtdl{(\dl{\rho} \dl{E})} + 
  \frac{\rhoref \uref^3}{\Lref} \dl{\div}([\dl{\rho} \dl{E} + \dl{p}] \dl{\vec{u}}) &= 
  \frac{\muref \uref^2}{\Lref^2}\dl{\div}(\dl{\TAU}) \dl{\vec{u}} -  
  \frac{\Tref \kref}{\Lref^2} \frac{\muref}{\muref} \dl{\div}{\dl{\HEAT}} + 
  \rhoref g \uref \dl{\rho} \dl{\vec{f}} \cdot \dl{\vec{u}} , \\
  \pdtdl{(\dl{\rho} \dl{E})} + 
  \dl{\div}([\dl{\rho} \dl{E} + \dl{p}] \dl{\vec{u}}) &= 
  \frac{\muref}{\Lref \rhoref \uref} \dl{\div}(\dl{\TAU} \dl{\vec{u}}) - 
  \frac{\muref}{\rhoref \uref \Lref} \frac{\Tref \kref}{\uref^2 \muref} \dl{\div}{\dl{\HEAT}} + 
  \frac{g \Lref}{\uref^2} \dl{\rho} \dl{\vec{f}} \cdot \dl{\vec{u}}, \\
  \pdtdl{(\dl{\rho} \dl{E})} + 
  \dl{\div}([\dl{\rho} \dl{E} + \dl{p}] \dl{\vec{u}}) &= 
  \frac{1}{\Re_{\text{ref}}}\dl{\div}(\dl{\TAU} \dl{\vec{u}}) -
  \frac{1}{\Re_{\text{ref}} \Pr_{\text{ref}}} \dl{\div}\dl{\HEAT} + 
  \frac{1}{\Fr^2_{\text{ref}}} \dl{\rho} \dl{\vec{f}} \cdot \dl{\vec{u}}.
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
    -->
