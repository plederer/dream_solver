import pytest
import ngsolve as ngs
import numpy.testing as nptest

from dream.compressible import CompressibleFlowSolver, flowfields, Initial

mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))


@pytest.fixture()
def cfg():
    cfg = CompressibleFlowSolver(mesh)
    cfg.time = "transient"

    cfg.fem = "conservative_hdg"
    cfg.fem.scheme = "implicit_euler"
    cfg.fem.order = 1

    yield cfg


def test_set_initial_conditions(cfg: CompressibleFlowSolver):

    cfg.dynamic_viscosity = "constant"
    cfg.fem.mixed_method = "strain_heat"

    # cfg.dcs['default'] = Initial(flowfields(rho=1, momentum=(ngs.x, -ngs.y), p=ngs.x))
    cfg.dcs['default'] = Initial(flowfields(rho=1, u=(ngs.x, -ngs.y), p=ngs.x))

    cfg.fem.initialize_finite_element_spaces()
    cfg.fem.initialize_trial_and_test_functions()
    cfg.fem.initialize_gridfunctions()
    cfg.fem.initialize_time_scheme_gridfunctions()
    cfg.fem.set_initial_conditions()

    U = cfg.get_solution_fields('rho', 'momentum', 'rho_E', 'strain_rate_tensor', 'grad_T')

    nptest.assert_almost_equal(ngs.Integrate(U.rho, mesh), 1)
    nptest.assert_almost_equal(ngs.Integrate(U.rho_u, mesh), [0.5, -0.5])
    nptest.assert_almost_equal(ngs.Integrate(U.p, mesh), 0.5)
    nptest.assert_almost_equal(ngs.Integrate(U.eps, mesh), [1, 0, 0, -1])
    nptest.assert_almost_equal(ngs.Integrate(U.grad_T, mesh), [1.4/(1.4 - 1), 0])
