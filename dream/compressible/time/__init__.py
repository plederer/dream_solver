from .implicit import (ImplicitEuler,
                       BDF2,
                       SDIRK22,
                       SDIRK33,
                       SDIRK54,
                       DIRK43_WSO2,
                       DIRK34_LDD)
from .explicit import (ExplicitEuler,
                       SSPRK3,
                       CRK4)
from .imex     import (IMEXRK_ARS443)

__all__ = ['ImplicitEuler',
           'BDF2',
           'SDIRK22',
           'SDIRK33',
           'SDIRK54',
           'DIRK43_WSO2',
           'DIRK34_LDD',
           'ExplicitEuler',
           'SSPRK3',
           'CRK4',
           'IMEXRK_ARS443']



from dream.time import TimeSchemes, TransientConfig, PseudoTimeSteppingConfig
from dream.config import dream_configuration


class CompressibleTransient(TransientConfig):

    name: str = "transient"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "scheme": ImplicitEuler(mesh, root)
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def scheme(self) -> ExplicitEuler | SSPRK3 | CRK4 | ImplicitEuler | BDF2 | SDIRK22 | SDIRK33 | SDIRK54 | DIRK43_WSO2 | DIRK34_LDD | IMEXRK_ARS443:
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: ExplicitEuler | SSPRK3 | CRK4 | ImplicitEuler | BDF2 | SDIRK22 | SDIRK33 | SDIRK54 | DIRK43_WSO2 | DIRK34_LDD | IMEXRK_ARS443):
        OPTIONS = [ExplicitEuler, SSPRK3, CRK4, ImplicitEuler, BDF2, SDIRK22, SDIRK33, SDIRK54, DIRK43_WSO2, DIRK34_LDD, IMEXRK_ARS443]
        self._scheme = self._get_configuration_option(scheme, OPTIONS, TimeSchemes)


class CompressiblePseudoTimeStepping(PseudoTimeSteppingConfig):

    name: str = "pseudo_time_stepping"

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "scheme": ImplicitEuler(mesh, root)
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def scheme(self) -> ExplicitEuler | SSPRK3 | CRK4 | ImplicitEuler | BDF2 | SDIRK22 | SDIRK33 | SDIRK54 | DIRK43_WSO2 | DIRK34_LDD:
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: ImplicitEuler | BDF2):
        OPTIONS = [ImplicitEuler, BDF2]
        self._scheme = self._get_configuration_option(scheme, OPTIONS, TimeSchemes)
