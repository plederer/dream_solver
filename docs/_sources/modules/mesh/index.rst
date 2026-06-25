mesh
========================================
.. currentmodule:: dream.mesh

.. include:: equations.md
    :parser: myst_parser.sphinx_

.. autosummary::

    get_nodal_points
    get_cylinder_mesh
    get_cylinder_omesh
    get_rectangular_mesh
    get_structured_cylinder_mesh
    get_2d_naca_occ_profile
    get_3d_naca_occ_profile
    get_chord_naca_4digit_series_coordinates
    BufferCoord
    SpongeFunction
    GridMapping
    Condition
    Periodic
    Initial
    Perturbation
    Buffer
    GridDeformation
    SpongeLayer
    PSpongeLayer
    Conditions
    BoundaryConditions
    DomainConditions

.. autofunction:: get_nodal_points
.. autofunction:: get_cylinder_mesh
.. autofunction:: get_cylinder_omesh
.. autofunction:: get_rectangular_mesh
.. autofunction:: get_structured_cylinder_mesh
.. autofunction:: get_2d_naca_occ_profile
.. autofunction:: get_3d_naca_occ_profile
.. autofunction:: get_chord_naca_4digit_series_coordinates

.. autoclass:: BufferCoord
   :members:
   :member-order: bysource
.. autoclass:: SpongeFunction
   :members:
   :member-order: bysource
.. autoclass:: GridMapping
   :members:
   :member-order: bysource

.. autoclass:: Condition
   :members:
   :member-order: bysource
.. autoclass:: Periodic
   :members:
   :member-order: bysource
.. autoclass:: Initial
   :members:
   :member-order: bysource
.. autoclass:: Perturbation
   :members:
   :member-order: bysource
.. autoclass:: Buffer
   :members:
   :member-order: bysource
.. autoclass:: GridDeformation
   :members:
   :member-order: bysource
.. autoclass:: SpongeLayer
   :members:
   :member-order: bysource
.. autoclass:: PSpongeLayer
   :members:
   :member-order: bysource

.. autoclass:: Conditions
   :members:
   :member-order: bysource
.. autoclass:: BoundaryConditions
   :members:
   :member-order: bysource
.. autoclass:: DomainConditions
   :members:
   :member-order: bysource

Examples
--------
.. toctree::
    :maxdepth: 1

    ../../examples/cylinder_mesh.ipynb
    ../../examples/naca_airfoil_mesh.ipynb
    ../../examples/rectangular_mesh.ipynb
    ../../examples/structured_cylinder_mesh.ipynb
