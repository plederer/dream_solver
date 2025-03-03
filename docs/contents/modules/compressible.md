The fundamental equations describing the motion of an unsteady, viscous and compressible flow in a space-time cylinder
$\Omega \times (0, t_{end}] \in \mathbb{R}^{d+1}$ with non-empty bounded $d$-dimensional spatial domain $\Omega$, with
boundary $\partial \Omega$, and final time $t_{end}$, are specified by the Navier-Stokes equations. In terms of
conservative variables, $\vec{U} = \begin{pmatrix} \rho, & \rho \vec{u}, & \rho E \end{pmatrix}^\T$, with density $\rho$, velocity $\vec{u}$, and total specific energy $E$,
this system can be expressed in dimensionless form as

$$
\pdt{\vec{U}} + \div(\mat{F}(\vec{U}) - \mat{G}(\vec{U}, \grad \vec{U})) = \vec{0} \quad \text{in} \quad \Omega \times (0, t_{end}],
$$

by applying a proper scaling, see 

### Solver
```{eval-rst}
    .. autoclass:: dream.compressible.solver.CompressibleFlowSolver
        :members:
``` 


### Finite Element Methods


### Equation of State
```{eval-rst}
    .. automodule:: dream.compressible.eos
        :members:
        :exclude-members: EquationOfState
``` 

### Riemann Solvers
```{eval-rst}
    .. autoclass:: dream.compressible.riemann_solver.LaxFriedrich
        :members:
    .. autoclass:: dream.compressible.riemann_solver.Roe
        :members:
    .. autoclass:: dream.compressible.riemann_solver.HLL
        :members:
    .. autoclass:: dream.compressible.riemann_solver.HLLEM
        :members:
    .. autoclass:: dream.compressible.riemann_solver.Upwind
        :members:
``` 

### Scaling
```{eval-rst}
    .. autoclass:: dream.compressible.scaling.Aerodynamic
        :members:
    .. autoclass:: dream.compressible.scaling.Acoustic
        :members:
    .. autoclass:: dream.compressible.scaling.Aeroacoustic
        :members:
``` 

### Dynamic Viscosity
```{eval-rst}
    .. autoclass:: dream.compressible.viscosity.Inviscid
        :members:
    .. autoclass:: dream.compressible.viscosity.Constant
        :members:
    .. autoclass:: dream.compressible.viscosity.Sutherland
        :members:
``` 

### Boundary Conditions

### Domain Conditions

### Utils
```{eval-rst}
    .. autoclass:: dream.compressible.config.flowstate
        :members:
``` 