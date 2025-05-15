## Compressible flow equations
The fundamental equations describing the motion of an unsteady, viscous and compressible flow in a space-time cylinder
$\Omega \times (0, t_{end}] \in \mathbb{R}^{d+1}$ with non-empty bounded $d$-dimensional spatial domain $\Omega$, with
boundary $\partial \Omega$, and final time $t_{end}$, are specified by the Navier-Stokes equations. In terms of
conservative variables, $\vec{U} = \begin{pmatrix} \rho, & \rho \vec{u}, & \rho E \end{pmatrix}^\T$, with density $\rho$, velocity $\vec{u}$, and total specific energy $E$,
this system can be expressed in dimensionless form as

\begin{align*}
    \frac{\partial \vec{U}}{\partial t} + \div(\vec{F}(\vec{U}) - \vec{G}(\vec{U}, \nabla \vec{U} )) = \vec{0}.
\end{align*}

In a general form the convective $\vec{F}$ and the viscous fluxes $\vec{G}$ are given by
\begin{align*}
    \vec{F}(\vec{U}) & = \begin{pmatrix}
                             \rho \vec{u}^\T                     \\
                             \rho \vec{u} \otimes \vec{u} + p \I \\
                             \rho H \vec{u}^\T
                         \end{pmatrix}, &
    \vec{G}(\vec{U}, \nabla \vec{U}) = \begin{pmatrix}\vec{0}^\T \\\mat{\tau}    \\ (\mat{\tau} \vec{u} - \vec{q})^\T \end{pmatrix},
\end{align*}
where $p$ denotes the pressure, $H = E + p/\rho$ the specific enthalpy, $\mat{\tau}$ the deviatoric stress tensor, and $\vec{q}$ the heat flux vector. 

To close the system of equations we need to specify the equation of state ({py:class}`dream.compressible.eos`) and the constitutive relations for the deviatoric stress tensor $\mat{\tau}$ ({py:class}`dream.compressible.viscosity`)  and the heat flux vector.

## Quasi-linear Euler equations
The hyperbolic nature of the Navier-Stokes equations with respect to time lies in the Euler
equations
\begin{align*}
    \frac{\partial \vec{U}}{\partial t} + \div(\vec{F}(\vec{U}))  = \vec{0},
\end{align*}
which are derived by neglecting the viscous contributions. From a characteristic point of view,
it is essential to express these equations in quasi-linear form
\begin{align*}
    \frac{\partial \vec{U}}{\partial t} + \sum_{i=1}^d \mat{A}_i \frac{\partial \vec{U}}{\partial x_i}  &= \vec{0}, \\
\end{align*}
where the $\mat{A}_i$ are the directional convective Jacobians.

## References
[^1]: C. Hirsch, Numerical Computation of Internal and External Flows. 2: Computational Methods for
Inviscid and Viscous Flows, repr Edition, A Wiley Interscience Publication, Wiley, Chichester, 2002.
[^2]: Jan Ellmenreich, Matteo Giacomini, Antonio Huerta, and Philip L. Lederer. Characteristic
boundary conditions for Hybridizable Discontinuous Galerkin methods, March 2025.
