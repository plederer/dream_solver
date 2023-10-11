## Compressible Equations in Conservative Form

### Navier Stokes Equations
Neglecting external forces $\vec{f} = 0$, the above derived dimensionless compressible equations can be expressed in a simple matter in conservative form

\begin{align*}
\pdt{\CVAR} + \div \left[ \vec{F}(\CVAR) - \vec{G}(\CVAR, \grad \CVAR) \right] &= 0.
\end{align*}

All the fluxes are split into a convective $\vec{F}$ and a diffusive $\vec{G}$ part, respectively.

\begin{align*}
\vec{F} &= 
\begin{bmatrix}
\rho \VEL \\ \rho \VEL \otimes \VEL + p \I \\ (\rho E +p) \VEL
\end{bmatrix}, &
\vec{G} &= 
\frac{1}{\Re_{\text{ref}}} 
\begin{bmatrix}
\vec{0} \\ \TAU \\ \TAU \VEL - \frac{1}{\Pr_{\text{ref}}} \HEAT
\end{bmatrix}
\end{align*} 


The conserved variables  are given by 

\begin{align*}
\CVAR &=
\begin{bmatrix}
\rho \\ \rho \VEL \\ \rho E
\end{bmatrix},
\end{align*}

and consist of mass, momentum and energy, respectively.

### Euler Equations

By further neglecting every dissipative contribution, we recover *Euler's* equations

\begin{align*}
\pdt{\CVAR} + \div \left[ \vec{F}(\CVAR) \right] &= 0.
\end{align*}


For the analysis the quasi-linear form is very helpful

\begin{align*}
\pdt{\CVAR} + \frac{\partial \vec{F}}{\partial \CVAR}\frac{\partial \CVAR}{\partial \vec{x}} &= 0.
\end{align*}


#### 2D Setting
In two dimensions the convective flux vector is defined as

\begin{align*}
\vec{F} &= 
\begin{bmatrix}
\rho u & \rho v \\ \rho u^2 + p & \rho u v  \\ \rho u v & \rho v^2 + p  \\ (\rho E +p ) u & (\rho E +p) v
\end{bmatrix} = \begin{bmatrix}
\vec{f} & \vec{g}
\end{bmatrix}.
\end{align*}


Then the quasi-linear form is given by


$$
\pdt{\CVAR} +  \mat{A}_{\CVAR} \frac{\partial \CVAR}{\partial x} + \mat{B}_{\CVAR}  \frac{\partial \CVAR}{\partial y}  = 0,
$$

with

$$
\mat{A}_{\CVAR}  = \frac{\partial \vec{f}}{\partial \CVAR} 
\begin{bmatrix}
0 & 1 & 0 & 0 \\
\frac{\gamma - 3}{2} u^2 + \frac{\gamma - 1}{2} v^2  & (3 - \gamma)u & -(\gamma - 1)v & \gamma - 1 \\
-uv & v & u & 0 \\
-\gamma u E + (\gamma - 1) u |\VEL |^2 & \gamma E - \frac{\gamma - 1}{2}(v^2 + 3u^2) & -(\gamma - 1) u v & \gamma u
\end{bmatrix},\\
\mat{B}_{\CVAR}  = \frac{\partial \vec{g}}{\partial \CVAR} =
\begin{bmatrix}
0 & 0 & 1 & 0 \\
-uv & v & u & 0 \\
\frac{\gamma - 3}{2} v^2 + \frac{\gamma - 1}{2} u^2  & -(\gamma - 1)u & (3 - \gamma)v  & \gamma - 1 \\
-\gamma v E + (\gamma - 1) v |\VEL |^2 & -(\gamma - 1) u v & \gamma E - \frac{\gamma - 1}{2}(u^2 + 3v^2) &  \gamma v
\end{bmatrix}.
$$
