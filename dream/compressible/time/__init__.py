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



from dream.config import interface
from dream.time import TransientConfig

# Proxy class for defining local transient time schemes.
class CompressibleTransient(TransientConfig):

    name = "CompressibleTransient"

    @interface(default=ImplicitEuler)
    def scheme(self, scheme):
        return scheme




