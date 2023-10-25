## Quasi-linear Euler Equations

\begin{align*}
\pdt{\CVAR} +  \mat{A}_{\CVAR} \frac{\partial \CVAR}{\partial x} + \mat{B}_{\CVAR}  \frac{\partial \CVAR}{\partial y}  &= 0,
\end{align*}

\begin{align*}
\mat{A}_{\CVAR}  &= \frac{\partial \vec{f}}{\partial \CVAR} = 
\begin{bmatrix}
0 & 1 & 0 & 0 \\
\frac{\gamma - 3}{2} u^2 + \frac{\gamma - 1}{2} v^2  & (3 - \gamma)u & -(\gamma - 1)v & \gamma - 1 \\
-uv & v & u & 0 \\
-\gamma u E + (\gamma - 1) u |\VEL |^2 & \gamma E - \frac{\gamma - 1}{2}(v^2 + 3u^2) & -(\gamma - 1) u v & \gamma u
\end{bmatrix}, \\
\mat{B}_{\CVAR}  &= \frac{\partial \vec{g}}{\partial \CVAR} =
\begin{bmatrix}
0 & 0 & 1 & 0 \\
-uv & v & u & 0 \\
\frac{\gamma - 3}{2} v^2 + \frac{\gamma - 1}{2} u^2  & -(\gamma - 1)u & (3 - \gamma)v  & \gamma - 1 \\
-\gamma v E + (\gamma - 1) v |\VEL |^2 & -(\gamma - 1) u v & \gamma E - \frac{\gamma - 1}{2}(u^2 + 3v^2) &  \gamma v
\end{bmatrix}.
\end{align*}

For sake of simplicity we recast the system in terms of primitive variables $\PVAR$

\begin{align*}
\PVAR &=
\begin{bmatrix}
\rho \\ \VEL \\ p
\end{bmatrix},
\end{align*}

\begin{align*}
\pdt{\PVAR} +  \mat{A}_{\PVAR}  \frac{\partial \PVAR}{\partial x} + \mat{B}_{\PVAR}  \frac{\partial \PVAR}{\partial y}  &= 0,
\end{align*}

where the transformation matrices to/from the variation of conservative variables from/to the variation of primitive variables are given as
\begin{align*}
\delta \CVAR &= \mat{M} \delta \PVAR & \delta \PVAR &= \mat{M}^{-1} \delta \CVAR
\end{align*}
\begin{align*}
\mat{M} := \frac{\partial \CVAR}{\partial \PVAR} &= 
\begin{bmatrix}
1 & 0 & 0 & 0 \\
u & \rho & 0 & 0 \\
v & 0 & \rho & 0 \\
\frac{|\VEL|^2}{2} & \rho u  & \rho v & \frac{1}{\gamma - 1}
\end{bmatrix}, \\
\mat{M}^{-1} := \frac{\partial \PVAR}{\partial \CVAR} &= 
\begin{bmatrix}
1 & 0 & 0 & 0 \\
-\frac{u}{\rho} & \frac{1}{\rho} & 0 & 0 \\
-\frac{v}{\rho} & 0 & \frac{1}{\rho} & 0 \\
(\gamma - 1) \frac{|\VEL|^2}{2} & -(\gamma - 1) u  & -(\gamma - 1) v & (\gamma - 1)
\end{bmatrix},
\end{align*}


and the quasi-linear convective matrices


\begin{align*}
\mat{A}_{\PVAR} := \mat{M}^{-1} \mat{A}_{\CVAR} \mat{M} &= 
\begin{bmatrix}
u & \rho & 0 & 0 \\
0 & u & 0 & \frac{1}{\rho} \\
0 & 0 & u & 0 \\
0 & \rho c^2 & 0 & u
\end{bmatrix}, &
\mat{B}_{\PVAR} := \mat{M}^{-1} \mat{B}_{\CVAR}  \mat{M} &=
\begin{bmatrix}
v & 0 & \rho & 0 \\
0 & v & 0 & 0 \\
0 & 0 & v & \frac{1}{\rho} \\
0 & 0 &\rho c^2 & v
\end{bmatrix}.
\end{align*}

Since the *Euler* equations are rotationally invariant, it is possible to introduce a local orthonormal frame of reference $\vec{\xi}$. Then it holds

\begin{align*}
\pdt{\PVAR} +  \mat{A}^{\xi}_{\PVAR}  \frac{\partial \PVAR}{\partial \xi} + \mat{B}^{\eta}_{\PVAR}  \frac{\partial \PVAR}{\partial \eta}  &= 0,
\end{align*}


\begin{align*}
\mat{A}^{\xi}_{\PVAR} := \mat{A}_{\PVAR} \xi_x +  \mat{B}_{\PVAR} \xi_y &= 
\begin{bmatrix}
u_{\vec{\xi}} & \rho \xi_x & \rho \xi_y & 0 \\
0 & u_{\vec{\xi}} & 0 & \frac{\xi_x}{\rho} \\
0 & 0 & u_{\vec{\xi}} & \frac{\xi_y}{\rho}  \\
0 & \rho c^2 \xi_x &  \rho c^2 \xi_y & u_{\vec{\xi}}
\end{bmatrix}, \\
\mat{B}^{\eta}_{\PVAR} := \mat{A}_{\PVAR} \eta_x +  \mat{B}_{\PVAR} \eta_y &= 
\begin{bmatrix}
u_{\vec{\eta}} & \rho \eta_x & \rho \eta_y & 0 \\
0 & u_{\vec{\eta}} & 0 & \frac{\eta_x}{\rho} \\
0 & 0 & u_{\vec{\eta}} & \frac{\eta_y}{\rho}  \\
0 & \rho c^2 \eta_x &  \rho c^2 \eta_y & u_{\vec{\eta}}
\end{bmatrix}.
\end{align*}

We will go on by performing an eigenvalue decomposition of the quasi-linear convective matrix $\mat{A}^{\xi}_V$. Unfortunately, as we will see, the right and left eigenvectors do not orthogonalise $\mat{B}^{\eta}_V$. This naturally arises from the physical properties of multidimensional waves
- If the spatial dimension is N, waves may have any dimension between $1$ and $N$ in the $\vec{x}$-$t$ plane
- Waves may travel in an infinite number of directions.
- Characteristic variables may not be constant along characteristics
- The characteristic form is not unique. 


\begin{align*}
\Lambda := \mat{L}^{-1} \mat{A}^{\xi}_{\PVAR}  \mat{L} &= 
\begin{bmatrix}
u_{\vec{\xi}} - c & 0 & 0 & 0 \\
0 & u_{\vec{\xi}} & 0 & 0 \\
0 & 0 & u_{\vec{\xi}} & 0 \\
0 & 0 & 0 & u_{\vec{\xi}} + c
\end{bmatrix}, \\
\mat{L}  &:= 
\begin{bmatrix}
\frac{1}{2c^2} & \frac{1}{c^2} & 0 & \frac{1}{2c^2} \\
-\frac{\xi_x}{2 c \rho} & 0 & \xi_y &  \frac{\xi_x}{2 c \rho}\\
-\frac{\xi_y}{2 c \rho} & 0 & -\xi_x & \frac{\xi_y}{2 c \rho} \\
\frac{1}{2} & 0 & 0 & \frac{1}{2}
\end{bmatrix}, \\
\mat{L}^{-1} &:= 
\begin{bmatrix}
0 & -\rho c \xi_x & -\rho c \xi_y & 1 \\
c^2 & 0 & 0 & -1 \\
0 & \xi_y & -\xi_x & 0 \\
0 & \rho c \xi_x & \rho c \xi_y & 1
\end{bmatrix}.
\end{align*}

At this point we can rewrite the system consisting of primitive variables $\PVAR$ as the variation of the characteristic variables $\delta \CHVAR$

\begin{align*}
\delta \CHVAR &=
\begin{bmatrix}
\delta p - \rho c (\delta u \xi_x + \delta v \xi_y) \\
c^2 \delta \rho - \delta p \\
\xi_y \delta u - \xi_x \delta v \\
\delta p + \rho c (\delta u \xi_x + \delta v \xi_y) \\
\end{bmatrix}
\end{align*}

\begin{align*}
\pdt{\CHVAR} +  \Lambda  \frac{\partial \CHVAR}{\partial \xi} + \mat{B}^{\eta}_{\CHVAR}\frac{\partial \CHVAR}{\partial \eta}  &= 0,
\end{align*}


As mentioned, note that $\mat{B}^{\eta}_{\CHVAR} = \mat{L}^{-1} \mat{B}^{\eta}_{\PVAR} \mat{L}$ is non-diagonal

\begin{align*}
\mat{B}^{\eta}_{\CHVAR} &= 
\begin{bmatrix}
u_{\vec{\eta}}& 0 & -c^2 \rho & 0 \\
0 & u_{\vec{\eta}} & 0 & 0 \\
-\frac{1}{2\rho} & 0 & u_{\vec{\eta}} & -\frac{1}{2\rho} \\
0 & 0 & -c^2 \rho & u_{\vec{\eta}}
\end{bmatrix}.
\end{align*}

From above relation, we can define the direct transformation from/to the variation of conservative variables to/from the variation of characteristic variables
\begin{align*}
\delta \CHVAR &= \mat{L}^{-1} \delta \PVAR & \delta \PVAR &= \mat{L} \delta \CHVAR \\
\delta \CHVAR &= \underbrace{\mat{L}^{-1} \mat{M}^{-1}}_{\mat{P}^{-1}} \delta \CVAR & \delta \CVAR &= \underbrace{\mat{M} \mat{L}}_{\mat{P}} \delta \CHVAR
\end{align*}

\begin{align*}
\mat{P} &:= 
\begin{bmatrix}
\frac{1}{2 c^{2}} & \frac{1}{c^{2}} & 0 & \frac{1}{2 c^{2}} \\ 
- \frac{\xi_{x}}{2 c} + \frac{u}{2 c^{2}} & \frac{u}{c^{2}} & \rho \xi_{y} & \frac{\xi_{x}}{2 c} + \frac{u}{2 c^{2}}\\
- \frac{\xi_{y}}{2 c} + \frac{v}{2 c^{2}} & \frac{v}{c^{2}} & - \rho \xi_{x} & \frac{\xi_{y}}{2 c} + \frac{v}{2 c^{2}}\\
\frac{0.5}{\gamma - 1} - \frac{u_{\xi}}{2 c} + \frac{\frac{u^{2}}{2} + \frac{v^{2}}{2}}{2 c^{2}} & \frac{\frac{u^{2}}{2} + \frac{v^{2}}{2}}{c^{2}} & \rho u \xi_{y} - \rho v \xi_{x} & \frac{0.5}{\gamma - 1} + \frac{u_{\xi}}{2 c} + \frac{\frac{u^{2}}{2} + \frac{v^{2}}{2}}{2 c^{2}}
\end{bmatrix} \\
\mat{P}^{-1} &:= 
\begin{bmatrix}
c u_{\xi} + \frac{\left(\gamma - 1\right) \left(u^{2} + v^{2}\right)}{2} & - c \xi_{x} + u \left(1 - \gamma\right) & - c \xi_{y} + v \left(1 - \gamma\right) & \gamma - 1\\
c^{2} - \frac{\left(\gamma - 1\right) \left(u^{2} + v^{2}\right)}{2} & - u \left(1 - \gamma\right) & - v \left(1 - \gamma\right) & 1 - \gamma\\
- \frac{u \xi_{y}}{\rho} + \frac{v \xi_{x}}{\rho} & \frac{\xi_{y}}{\rho} & - \frac{\xi_{x}}{\rho} & 0\\
- c u_{\xi} + \frac{\left(\gamma - 1\right) \left(u^{2} + v^{2}\right)}{2} & c \xi_{x} + u \left(1 - \gamma\right) & c \xi_{y} + v \left(1 - \gamma\right) & \gamma - 1
\end{bmatrix}
\end{align*}

