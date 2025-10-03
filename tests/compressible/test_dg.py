import pytest
import ngsolve as ngs
import numpy as np
import numpy.testing as nptest
import dream.bla as bla

from dream.compressible import CompressibleFlowSolver, flowfields, Initial

mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))


@pytest.fixture()
def cfg():
    cfg = CompressibleFlowSolver(mesh)
    cfg.time = "transient"

    cfg.fem = "conservative_dg"
    cfg.fem.order = 1

    cfg.fem.scheme = "explicit_euler"

    yield cfg


def test_set_initial_conditions(cfg: CompressibleFlowSolver):

    cfg.dynamic_viscosity = "constant"
    cfg.fem.viscous_treatment = "interior_penalty_method_sdg"

    cfg.dcs['default'] = Initial(flowfields(rho=1, u=(ngs.x, -ngs.y), p=ngs.x))

    cfg.fem.initialize_finite_element_spaces()
    cfg.fem.initialize_trial_and_test_functions()
    cfg.fem.initialize_gridfunctions()
    cfg.fem.initialize_time_scheme_gridfunctions()
    cfg.fem.set_initial_conditions()

    U = cfg.get_solution_fields('rho', 'momentum', 'rho_E')
    U.grad_u = ngs.CF((1, 0, 0, -1), dims=(2, 2))
    U.grad_rho = (0, 0)
    U.grad_p = (1, 0)

    U.eps = cfg.strain_rate_tensor(U)
    U.grad_T = cfg.temperature_gradient(U, U)

    nptest.assert_almost_equal(ngs.Integrate(U.rho, mesh), 1)
    nptest.assert_almost_equal(ngs.Integrate(U.rho_u, mesh), [0.5, -0.5])
    nptest.assert_almost_equal(ngs.Integrate(U.p, mesh), 0.5)
    nptest.assert_almost_equal(ngs.Integrate(U.eps, mesh), [1, 0, 0, -1])
    nptest.assert_almost_equal(ngs.Integrate(U.grad_T, mesh), [1.4/(1.4 - 1), 0])


def test_viscous_diffusion_matrices(cfg: CompressibleFlowSolver):

    cfg.dynamic_viscosity = "constant"
    cfg.fem.viscous_treatment = "interior_penalty_method_sdg"

    cfg.reynolds_number = 10
    cfg.mach_number = 1
    cfg.prandtl_number = 0.72
    cfg.equation_of_state.heat_capacity_ratio = 1.4

    U = flowfields(rho=1+ngs.y, u=(ngs.x, -ngs.y), p=1+ngs.x)
    U.grad_u = ngs.CF((1, 0, 0, -1), dims=(2, 2))
    U.grad_rho = (0, 1)
    U.grad_p = (1, 0)

    U.grad_T = cfg.temperature_gradient(U, U)
    U.rho_Ei = cfg.inner_energy(U)
    U.rho_Ek = cfg.kinetic_energy(U)
    U.rho_E = cfg.energy(U)

    U.Ei = cfg.specific_inner_energy(U)
    U.grad_Ei = cfg.specific_inner_energy_gradient(U, U)

    U.grad_rho_Ek = cfg.kinetic_energy_gradient(U, U)
    U.grad_rho_Ei = cfg.inner_energy_gradient(U, U)

    U.grad_rho_u = cfg.momentum_gradient(U, U)
    U.grad_rho_E = cfg.energy_gradient(U, U)

    Fv = cfg.get_diffusive_flux(U, U)

    K11, K12, K21, K22 = cfg.fem.viscous_treatment.get_frozen_diffusion_matrices_conservative(U)
    grad_U = ngs.CF((U.grad_rho, U.grad_rho_u, U.grad_rho_E), dims=(4, 2))
    Fv_mat = ngs.CF((K11*grad_U[:, 0] + K12*grad_U[:, 1], K21 * grad_U[:, 0] + K22 * grad_U[:, 1]), dims=(2, 4)).trans

    print(Fv(mesh()), ngs.Integrate(Fv, mesh))
    print(Fv_mat(mesh()), ngs.Integrate(Fv_mat, mesh))
    nptest.assert_almost_equal(np.array(ngs.Integrate(Fv, mesh)), np.array(ngs.Integrate(Fv_mat, mesh)))


def test_diffusive_flux_from_conservative_jump(cfg: CompressibleFlowSolver):

    cfg.dynamic_viscosity = "constant"
    cfg.fem.viscous_treatment = "interior_penalty_method_sdg"

    cfg.reynolds_number = 10
    cfg.mach_number = 1
    cfg.prandtl_number = 0.72
    cfg.equation_of_state.heat_capacity_ratio = 1.4

    # Simulated normal vector
    for n in [ngs.CF((1, 1)), ngs.CF((-1, -1))]:

        U = flowfields(rho=1+ngs.y, u=(ngs.x, -ngs.y), p=1+ngs.x)

        # Viscous flux from jump in U only
        Ujump = ngs.CF((1, n, 1))
        Fv = cfg.fem.viscous_treatment.get_diffusive_flux_from_conservative_jump(U, Ujump, n)

        # Viscous flux viscous jacobians and gradient of U
        K11, K12, K21, K22 = cfg.fem.viscous_treatment.get_frozen_diffusion_matrices_conservative(U)
        grad_U = ngs.CF((n, bla.outer(n, n), n), dims=(4, 2))
        Fv_mat = ngs.CF(
            (K11 * grad_U[:, 0] + K12 * grad_U[:, 1],
             K21 * grad_U[:, 0] + K22 * grad_U[:, 1]),
            dims=(2, 4)).trans

        nptest.assert_almost_equal(np.array(ngs.Integrate(Fv, mesh)), np.array(ngs.Integrate(Fv_mat, mesh)))
