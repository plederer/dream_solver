from __future__ import annotations
from ngsolve import *
from .conditions import BoundaryConditions, DomainConditions
from ..utils import DreAmLogger

logger = DreAmLogger.get_logger("Mesh")


class DreamMesh:

    def __init__(self, mesh: Mesh) -> None:

        self._mesh = mesh
        self._bcs = BoundaryConditions(self.mesh.GetBoundaries())
        self._dcs = DomainConditions(self.mesh.GetMaterials())
        self._is_periodic = bool(mesh.GetPeriodicNodePairs(VERTEX))

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @property
    def dim(self) -> int:
        return self.mesh.dim

    @property
    def bcs(self) -> BoundaryConditions:
        return self._bcs

    @property
    def dcs(self) -> DomainConditions:
        return self._dcs

    @property
    def boundary_names(self) -> tuple[str]:
        return tuple(self.bcs)

    @property
    def domain_names(self) -> tuple[str]:
        return tuple(self.dcs)

    @property
    def is_grid_deformation(self) -> bool:
        return bool(self.dcs.grid_deformation)

    @property
    def is_periodic(self) -> bool:
        return self._is_periodic

    @property
    def highest_order_psponge(self) -> int:
        return max([sponge.order.high for sponge in self.dcs.psponge_layers.values()], default=0)

    def boundary(self, region: str) -> Region:
        return self.mesh.Boundaries(region)

    def domain(self, region: str) -> Region:
        return self.mesh.Materials(region)

    def pattern(self, sequence: list) -> str:
        return "|".join(sequence)

    def set_grid_deformation(self):
        grid = self.get_grid_deformation_function()
        self.mesh.SetDeformation(grid)

    def get_grid_deformation_function(self) -> GridFunction:
        return self._get_buffer_grid_function(self.dcs.GridDeformation)

    def get_sponge_weight_function(self) -> GridFunction:
        return self._get_buffer_grid_function(self.dcs.SpongeLayer)

    def get_psponge_weight_function(self) -> GridFunction:
        return self._get_buffer_grid_function(self.dcs.PSpongeLayer)

    def _get_buffer_grid_function(self, type) -> GridFunction:

        fes = type.fes(self.mesh, order=type.fes_order)
        if type is self.dcs.GridDeformation and self.is_periodic:
            fes = Periodic(fes)

        u, v = fes.TnT()
        buffer = GridFunction(fes)

        domains = self.dcs._get_condition(type)

        if domains:

            blf = BilinearForm(fes)
            blf += InnerProduct(u, v) * dx

            lf = LinearForm(fes)
            for domain, bc in domains.items():

                domain = self.domain(domain)

                if isinstance(bc, self.dcs.GridDeformation):
                    lf += InnerProduct(bc.deformation_function(self.dim),
                                       v) * dx(definedon=domain, bonus_intorder=bc.bonus_int_order)
                else:
                    lf += bc.weight_function * v * dx(definedon=domain, bonus_intorder=bc.bonus_int_order)

            blf.Assemble()
            lf.Assemble()

            buffer.vec.data = blf.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * lf.vec

        else:
            buffer.vec[:] = 0
            logger.warning(f"{type.__name__} has not been set in domain conditions! Returning zero GridFunction.")

        return buffer