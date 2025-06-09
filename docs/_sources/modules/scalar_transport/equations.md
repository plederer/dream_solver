## Scalar transport equation
A transport equation expresses a conservation principle by describing how a physical quantity $\phi$ evolves in space and time due to the combined effects of convection and diffusion. A general form of this process is modeled by the (linear and scalar) convection–diffusion equation:
\begin{align*}
    \frac{\partial \phi}{\partial t} + \nabla \cdot (\vec{b}\phi) - \nabla \cdot (\kappa \nabla \phi) = 0,
\end{align*}
where $\vec{b}$ is the advecting velocity field (possibly space-dependent) and $\kappa$ is the diffusivity coefficient. Note, the dimension of $\vec{b}$ is deduced from the spatial dimension of the input grid.

:note: A pure convection equation can be solved, by setting {py:func}`~dream.scalar_transport.solver.ScalarTransportSolver.is_inviscid` to `False`.
