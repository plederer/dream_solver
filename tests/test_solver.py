import pytest
import ngsolve as ngs
from dream.config import dream_configuration
from dream.solver import SolverConfiguration, FiniteElementMethod
from dream.mesh import BoundaryConditions, DomainConditions
from dream.time import StationaryRoutine, TransientRoutine, TimeRoutine, TimeSchemes

mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))


class DummyFiniteElementMethod(FiniteElementMethod):

    name: str = 'dummy'
    root: SolverConfiguration

    def __init__(self, mesh, root=None, **default):

        DEFAULT = {
            "scheme": DummyTimeScheme(mesh, root)
        }
        DEFAULT.update(default)

        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def scheme(self):
        return self._scheme

    @scheme.setter
    def scheme(self, scheme: TimeSchemes):
        OPTIONS = [DummyTimeScheme]
        self._scheme = self._get_configuration_option(scheme, OPTIONS, TimeSchemes)

    def initialize_time_scheme_gridfunctions(self):
        return super().initialize_time_scheme_gridfunctions('U', 'Uhat')

    def add_finite_element_spaces(self, spaces: dict[str, ngs.FESpace]):
        spaces['U'] = ngs.L2(self.mesh, order=0)
        spaces['Uhat'] = ngs.L2(self.mesh, order=0)

    def get_temporal_integrators(self):
        return {'U': ngs.dx, 'Uhat': ngs.dx}

    def add_symbolic_spatial_forms(self, blf, lf):
        u, v = self.TnT['U']

        blf['U']['test'] = u * v * ngs.dx
        lf['U']['test'] = v * ngs.dx

    def add_symbolic_temporal_forms(self, blf, lf):
        pass

    def get_solution_fields(self):
        pass

    def set_initial_conditions(self) -> None:
        pass

    def set_boundary_conditions(self) -> None:
        pass


class DummyTimeScheme(TimeSchemes):
    time_levels = ("n", "n+1")

    def add_symbolic_temporal_forms(self, blf, lf):
        ...


class DummySolverConfiguration(SolverConfiguration):

    name: str = 'dummy'

    def __init__(self, mesh, **default):
        bcs = BoundaryConditions(mesh, [])
        dcs = DomainConditions(mesh, [])

        DEFAULT = {
            "fem": DummyFiniteElementMethod(mesh, self),
            "time": TransientRoutine(mesh, self),
        }
        DEFAULT.update(default)
        super().__init__(mesh, bcs, dcs, **DEFAULT)

    @dream_configuration
    def fem(self):
        return self._fem

    @fem.setter
    def fem(self, fem: FiniteElementMethod):
        self._fem = fem

    @dream_configuration
    def time(self) -> TransientRoutine:
        return self._time

    @time.setter
    def time(self, time: FiniteElementMethod):
        OPTIONS = [TransientRoutine, StationaryRoutine]
        self._time = self._get_configuration_option(time, OPTIONS, TimeRoutine)


@pytest.fixture
def cfg():
    return DummySolverConfiguration(mesh=mesh)


def test_initialize_finite_element_spaces_dictionary(cfg: DummySolverConfiguration):
    cfg.fem.initialize_finite_element_spaces()

    assert tuple(cfg.fem.spaces) == ('U', 'Uhat')

    for space in cfg.fem.spaces.values():
        assert isinstance(space, ngs.L2)


def test_initialize_fininte_element_space(cfg: DummySolverConfiguration):
    cfg.fem.initialize_finite_element_spaces()

    assert isinstance(cfg.fem.fes, ngs.ProductSpace)


def test_trial_and_test_functions(cfg: DummySolverConfiguration):
    cfg.fem.initialize_finite_element_spaces()
    cfg.fem.initialize_trial_and_test_functions()

    assert tuple(cfg.fem.TnT) == ('U', 'Uhat')

    for trial, test in cfg.fem.TnT.values():
        assert isinstance(trial, ngs.comp.ProxyFunction)
        assert isinstance(test, ngs.comp.ProxyFunction)


def test_initialize_gridfunction_components(cfg: DummySolverConfiguration):
    cfg.fem.initialize_finite_element_spaces()
    cfg.fem.initialize_trial_and_test_functions()
    cfg.fem.initialize_gridfunctions()

    assert tuple(cfg.fem.gfus) == ('U', 'Uhat')

    for gfu in cfg.fem.gfus.values():
        assert isinstance(gfu, ngs.GridFunction)


def test_initialize_gridfunction(cfg: DummySolverConfiguration):
    cfg.fem.initialize_finite_element_spaces()
    cfg.fem.initialize_trial_and_test_functions()
    cfg.fem.initialize_gridfunctions()

    assert isinstance(cfg.fem.gfu, ngs.GridFunction)


def test_initialze_transient_gridfunctions(cfg: DummySolverConfiguration):
    cfg.fem.initialize_finite_element_spaces()
    cfg.fem.initialize_trial_and_test_functions()
    cfg.fem.initialize_gridfunctions()
    cfg.fem.initialize_time_scheme_gridfunctions()

    assert 'U' in cfg.fem.scheme.gfus
    assert 'Uhat' in cfg.fem.scheme.gfus
    assert isinstance(cfg.fem.scheme.gfus['U']['n+1'], ngs.GridFunction)
    assert isinstance(cfg.fem.scheme.gfus['U']['n+1'].space, ngs.L2)
    assert isinstance(cfg.fem.scheme.gfus['Uhat']['n+1'], ngs.GridFunction)
    assert isinstance(cfg.fem.scheme.gfus['Uhat']['n+1'].space, ngs.L2)


def test_initialize_symbolic_forms(cfg: DummySolverConfiguration):
    cfg.fem.initialize_finite_element_spaces()
    cfg.fem.initialize_trial_and_test_functions()
    cfg.fem.initialize_symbolic_forms()

    assert 'test' in cfg.fem.blf['U']
    assert 'test' in cfg.fem.lf['U']
