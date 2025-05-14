from .implicit import (ImplicitEuler)
__all__ = ['ImplicitEuler']



from dream.time import TimeSchemes, TransientConfig, PseudoTimeSteppingConfig
from dream.config import dream_configuration


class ScalarTransportTransient(TransientConfig):

    name: str = "transient"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "scheme": ImplicitEuler(mesh, root)
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def scheme(self) -> ImplicitEuler:
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: ImplicitEuler):
        OPTIONS = [ImplicitEuler]
        self._scheme = self._get_configuration_option(scheme, OPTIONS, TimeSchemes)


