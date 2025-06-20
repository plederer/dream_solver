import pytest
import ngsolve as ngs
from dream.config import dream_configuration
from dream.solver import SolverConfiguration, FiniteElementMethod, DirectSolver
from dream.mesh import BoundaryConditions, DomainConditions
from dream.time import StationaryRoutine, TransientRoutine, TimeRoutine, TimeSchemes
import numpy as np
import numpy.testing as npt
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


def test_fem_bonus_int_order_default_setter():

    fem = FiniteElementMethod(None)

    assert fem.bonus_int_order == {}


def test_fem_bonus_int_order_sequence_setter():

    fem = FiniteElementMethod(None, bonus_int_order=('convection', 'diffusion'))

    assert fem.bonus_int_order == {'convection': {'vol': 0, 'bnd': 0}, 'diffusion': {'vol': 0, 'bnd': 0}, }


def test_fem_bonus_int_order_set_all():

    fem = FiniteElementMethod(None, bonus_int_order=('convection', 'diffusion'))
    fem.bonus_int_order = 1

    assert fem.bonus_int_order == {'convection': {'vol': 1, 'bnd': 1}, 'diffusion': {'vol': 1, 'bnd': 1}, }


def test_fem_bonus_int_order_set_single():

    fem = FiniteElementMethod(None, bonus_int_order=('convection', 'diffusion'))
    fem.bonus_int_order['convection']['vol'] = 1

    assert fem.bonus_int_order == {'convection': {'vol': 1, 'bnd': 0}, 'diffusion': {'vol': 0, 'bnd': 0}, }


def test_direct_solver_inverse():
    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))

    solver = DirectSolver(mesh)

    fes = ngs.H1(mesh, order=1)
    u, v = fes.TnT()

    blf = ngs.BilinearForm(fes)
    blf += u * v * ngs.dx

    blf.Assemble()

    inv = np.linalg.inv(np.array(blf.mat.ToDense()))

    npt.assert_array_almost_equal(inv, np.array(solver.get_inverse(blf, fes).ToDense()))
    npt.assert_array_almost_equal(inv, np.array(solver.get_inverse(blf, fes, inverse="sparsecholesky").ToDense()))

    inv[:, :] = 0.0
    inv[:2, :2] = np.linalg.inv(np.array(blf.mat.ToDense())[:2, :2])
    dofs = ngs.BitArray(fes.ndof)
    dofs.Clear()
    dofs[:2] = True

    npt.assert_array_almost_equal(inv, np.array(solver.get_inverse(blf, fes, freedofs=dofs).ToDense()))


def test_direct_solver_solve_linear_system():

    ue = ngs.sin(ngs.x * ngs.pi) * ngs.sin(ngs.y * ngs.pi) + 1
    f = 2*ngs.pi**2 * ngs.sin(ngs.x * ngs.pi) * ngs.sin(ngs.y * ngs.pi)

    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2))

    solver = DirectSolver(mesh)

    fes = ngs.H1(mesh, order=4, dirichlet='bottom|right|top|left')
    u, v = fes.TnT()

    lf = ngs.LinearForm(fes)
    lf += f * v * ngs.dx
    lf.Assemble()

    gfu = ngs.GridFunction(fes)
    gfu.Set(1)

    blf = ngs.BilinearForm(fes)
    blf += ngs.grad(u) * ngs.grad(v) * ngs.dx
    blf.Assemble()

    # Solve the system with the gridfunction set to 1. With the operator set to '+='.
    solver.solve_linear_system(blf, gfu, lf.vec, operator='+=')
    npt.assert_almost_equal(ngs.Integrate(gfu - ue, mesh), 0)

    # Solve the system with the gridfunction set to 1. With the operator set to '='.
    # This should raise an AssertionError, since the gridfunction is going to be reset to zero on the boundary.
    solver.solve_linear_system(blf, gfu, lf.vec, operator='=')
    with pytest.raises(AssertionError):
        npt.assert_almost_equal(ngs.Integrate(gfu - ue, mesh), 0)


def test_direct_solver_solve_linear_system_static_condensation():

    ue = ngs.sin(ngs.x * ngs.pi) * ngs.sin(ngs.y * ngs.pi) + 1
    f = 2*ngs.pi**2 * ngs.sin(ngs.x * ngs.pi) * ngs.sin(ngs.y * ngs.pi)

    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.2))

    solver = DirectSolver(mesh)

    fes = ngs.H1(mesh, order=4, dirichlet='bottom|right|top|left')
    u, v = fes.TnT()

    lf = ngs.LinearForm(fes)
    lf += f * v * ngs.dx
    lf.Assemble()

    gfu = ngs.GridFunction(fes)
    gfu.Set(1)

    blf = ngs.BilinearForm(fes, condense=True)
    blf += ngs.grad(u) * ngs.grad(v) * ngs.dx
    blf.Assemble()

    # Solve the system with the gridfunction set to 1. With the operator set to '+='.
    solver.solve_linear_system(blf, gfu, lf.vec, operator='+=')
    npt.assert_almost_equal(ngs.Integrate(gfu - ue, mesh), 0)

    # Solve the system with the gridfunction set to 1. With the operator set to '='.
    # This should raise an AssertionError, since the gridfunction is going to be reset to zero on the boundary.
    solver.solve_linear_system(blf, gfu, lf.vec, operator='=')
    with pytest.raises(AssertionError):
        npt.assert_almost_equal(ngs.Integrate(gfu - ue, mesh), 0)

def test_direct_solver_solve_nonlinear_system():

    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))

    solver = DirectSolver(mesh)

    fes = ngs.L2(mesh, order=0)
    u, v = fes.TnT()

    blf = ngs.BilinearForm(fes)
    blf += (u**3 - 8) * v * ngs.dx

    gfu = ngs.GridFunction(fes)
    gfu.Set(1)

    solver.initialize_nonlinear_routine(blf, gfu)
    for log in solver.solve_nonlinear_system():
        ...

    npt.assert_almost_equal(ngs.Integrate(gfu - 2, mesh), 0)

def test_direct_solver_solve_nonlinear_system_with_rhs():

    mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))

    solver = DirectSolver(mesh)

    fes = ngs.L2(mesh, order=0)
    u, v = fes.TnT()

    blf = ngs.BilinearForm(fes)
    blf += u**3 * v * ngs.dx

    lf = ngs.LinearForm(fes)
    lf += 8 * v * ngs.dx
    lf.Assemble()

    gfu = ngs.GridFunction(fes)
    gfu.Set(1)

    solver.initialize_nonlinear_routine(blf, gfu, lf.vec)
    for log in solver.solve_nonlinear_system():
        ...

    npt.assert_almost_equal(ngs.Integrate(gfu - 2, mesh), 0)


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
