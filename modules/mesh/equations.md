The {py:mod}`dream.mesh` module provides mesh-generation utilities, buffer/sponge layer
infrastructure, and the condition containers shared by all `dream` solvers.  It wraps Netgen's
meshing API so that common aeroacoustic mesh layouts can be constructed with a few function calls.

### Predefined mesh generators

Several helper functions produce ready-to-use {py:class}`~ngsolve.Mesh` objects for standard
geometries:

* {py:func}`~dream.mesh.get_cylinder_mesh` — unstructured mesh around a circular cylinder,
  optionally with a body-fitted boundary layer, a graded transition region and an outer sponge
  annulus.
* {py:func}`~dream.mesh.get_cylinder_omesh` — fully structured O-grid ring mesh with a fixed
  number of elements in the polar and radial directions.
* {py:func}`~dream.mesh.get_2d_naca_occ_profile` and
  {py:func}`~dream.mesh.get_3d_naca_occ_profile` — NACA 4-digit airfoil profiles as OCC
  shapes that can be embedded in any surrounding geometry before meshing.

### Structured mesh generators

{py:func}`~dream.mesh.get_rectangular_mesh` and
{py:func}`~dream.mesh.get_structured_cylinder_mesh` build fully structured meshes from explicit
nodal coordinate arrays.  The caller supplies one array per spatial direction for each domain
region; the function takes their union as the global set of mesh nodes and connects them into
quad or triangle elements:

```python
import numpy as np
from dream.mesh import get_rectangular_mesh, get_nodal_points

nx = np.linspace(0, 4, 41)
ny = get_nodal_points(21, distribution='tanh', beta=3) - 0.5   # cluster near walls

domains    = [("channel", (nx, ny))]
boundaries = [("inflow",  (np.array([0.0]), ny)),
              ("outflow", (np.array([4.0]), ny)),
              ("wall",    (nx, np.array([-0.5]))),
              ("wall",    (nx, np.array([0.5])))]

mesh = get_rectangular_mesh(domains, boundaries)
```

The polar variant {py:func}`~dream.mesh.get_structured_cylinder_mesh` works the same way but
accepts radial and angular coordinates $(r, \varphi)$ instead of Cartesian ones.

{py:func}`~dream.mesh.get_nodal_points` returns 1-D node distributions in $[0,1]$ with several
clustering options (`'uniform'`, `'cosine'`, `'polynomial'`, `'tanh'`, `'exponential'`) that
can be used to concentrate grid points near boundaries.

### Buffer and sponge layer infrastructure

`dream` uses e.g. buffer layers to implement non-reflecting far-field conditions.  Two mechanisms
are available:

**Grid deformation** ({py:class}`~dream.mesh.GridDeformation`): the computational mesh is
stretched inside the buffer region so that outgoing waves encounter progressively coarser
resolution and are damped numerically.  The deformation is described by a
{py:class}`~dream.mesh.GridMapping`, which maps a {py:class}`~dream.mesh.BufferCoord` in the
computational domain to a new position in the physical domain:

```python
from dream.mesh import BufferCoord, GridMapping, GridDeformation

x   = BufferCoord.x(x0=3.0, xn=5.0)          # buffer extends from x=3 to x=5
map = GridMapping.exponential(scale=5, coordinate=x)
deformation = GridDeformation(x=map, order=3)
```

Four mapping types are available: {py:meth}`~dream.mesh.GridMapping.none` (identity),
{py:meth}`~dream.mesh.GridMapping.linear`, {py:meth}`~dream.mesh.GridMapping.exponential`, and
{py:meth}`~dream.mesh.GridMapping.tangential`.  Polar and spherical buffer coordinates are
supported via {py:meth}`~dream.mesh.BufferCoord.polar` and
{py:meth}`~dream.mesh.BufferCoord.spherical`.

**Sponge layer** ({py:class}`~dream.mesh.SpongeLayer`): a volumetric penalty term
$\sigma(\vec{x})\,(\vec{U} - \vec{U}_\infty)$ is added to the right-hand side inside the
sponge region, where $\sigma$ is the sponge weight function provided by
{py:class}`~dream.mesh.SpongeFunction`:

```python
from dream.mesh import SpongeFunction

sigma = SpongeFunction.polynomial(weight=2.0, x=x, order=3)
sponge = SpongeLayer(function=sigma, target_state={"rho": 1.0, "u": (1.0, 0.0)})
```

The p-type variant {py:class}`~dream.mesh.PSpongeLayer` additionally reduces the local
polynomial order from a high value at the inner edge of the layer down to a low value at the
outer edge, introducing extra numerical dissipation alongside the explicit damping term.

### Condition containers

{py:class}`~dream.mesh.BoundaryConditions` and {py:class}`~dream.mesh.DomainConditions` map
mesh regions to {py:class}`~dream.mesh.Condition` instances using NGSolve's pipe-separated
region-name syntax:

```python
from dream.mesh import BoundaryConditions, DomainConditions, Periodic, Initial

bcs = BoundaryConditions(mesh, options=[Periodic])
bcs["inflow|outflow"] = Periodic()

dcs = DomainConditions(mesh, options=[SpongeLayer, GridDeformation])
dcs["sponge"] = SpongeLayer(function=sigma, target_state={"rho": 1.0})
```

Both containers warn when a region pattern does not match any mesh region, and they flag
multiple conditions set on the same region.
