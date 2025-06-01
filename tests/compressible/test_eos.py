import pytest
import ngsolve as ngs
import numpy.testing as nptest

from dream.compressible import CompressibleFlowSolver, flowfields

mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))
mip = mesh(0.5, 0.5)


@pytest.fixture()
def eos():
    cfg = CompressibleFlowSolver(mesh)
    cfg.time = "transient"

    cfg.equation_of_state = "ideal"
    cfg.equation_of_state.heat_capacity_ratio = 1.4

    yield cfg.equation_of_state


def test_density(eos):
    U = flowfields()
    assert eos.density(U) is None

    U = flowfields(p=1, T=1)
    assert eos.density(U)(mip) == pytest.approx(3.5)

    U = flowfields(p=1, speed_of_sound=1)
    assert eos.density(U)(mip) == pytest.approx(1.4)

    U = flowfields(inner_energy=1, T=1)
    assert eos.density(U)(mip) == pytest.approx(1.4)


def test_pressure(eos):
    U = flowfields()
    assert eos.pressure(U) is None

    U = flowfields(rho=1, T=1)
    assert eos.pressure(U)(mip) == pytest.approx(2/7)

    U = flowfields(rho_Ei=1)
    assert eos.pressure(U)(mip) == pytest.approx(0.4)

    U = flowfields(rho=1, c=1)
    assert eos.pressure(U)(mip) == pytest.approx(5/7)


def test_temperature(eos):
    U = flowfields()
    assert eos.temperature(U) is None

    U = flowfields(rho=1, p=1)
    assert eos.temperature(U)(mip) == pytest.approx(7/2)

    U = flowfields(Ei=1)
    assert eos.temperature(U)(mip) == pytest.approx(1.4)

    U = flowfields(c=1)
    assert eos.temperature(U)(mip) == pytest.approx(5/2)


def test_inner_energy(eos):
    U = flowfields()
    assert eos.inner_energy(U) is None

    U = flowfields(p=1)
    assert eos.inner_energy(U)(mip) == pytest.approx(5/2)

    U = flowfields(rho=1, T=1)
    assert eos.inner_energy(U)(mip) == pytest.approx(5/7)


def test_specific_inner_energy(eos):
    U = flowfields()
    assert eos.specific_inner_energy(U) is None

    U = flowfields(T=1)
    assert eos.specific_inner_energy(U)(mip) == pytest.approx(5/7)

    U = flowfields(rho=1, p=1)
    assert eos.specific_inner_energy(U)(mip) == pytest.approx(5/2)


def test_speed_of_sound(eos):
    U = flowfields()
    assert eos.speed_of_sound(U) is None

    U = flowfields(rho=1, p=1)
    assert eos.speed_of_sound(U)(mip) == pytest.approx(ngs.sqrt(1.4))

    U = flowfields(T=1)
    assert eos.speed_of_sound(U)(mip) == pytest.approx(ngs.sqrt(0.4))

    U = flowfields(specific_inner_energy=1)
    assert eos.speed_of_sound(U)(mip) == pytest.approx(ngs.sqrt(2/7))


def test_specific_entropy(eos):
    U = flowfields()
    assert eos.specific_entropy(U) is None

    U = flowfields(rho=2, p=1)
    assert eos.specific_entropy(U)(mip) == pytest.approx(1/2**1.4)


def test_density_gradient(eos):
    U = flowfields()
    dU = flowfields()
    assert eos.density_gradient(U, dU) is None

    U = flowfields(T=1, p=1)
    dU = flowfields(grad_p=(1, 1), grad_T=(1, 0))
    nptest.assert_almost_equal(eos.density_gradient(U, dU)(mip), (0, 7/2))

    U = flowfields(T=1, rho_Ei=1)
    dU = flowfields(grad_T=(1, 1), grad_rho_Ei=(1, 0))
    nptest.assert_almost_equal(eos.density_gradient(U, dU)(mip), (0, -7/5))


def test_pressure_gradient(eos):
    U = flowfields()
    dU = flowfields()
    assert eos.pressure_gradient(U, dU) is None

    U = flowfields(T=1, rho=1)
    dU = flowfields(grad_rho=(1, 1), grad_T=(1, 0))
    nptest.assert_almost_equal(eos.pressure_gradient(U, dU)(mip), (4/7, 2/7))

    dU = flowfields(grad_rho_Ei=(1, 0))
    nptest.assert_almost_equal(eos.pressure_gradient(U, dU)(mip), (0.4, 0))


def test_temperature_gradient(eos):
    U = flowfields()
    dU = flowfields()
    assert eos.temperature_gradient(U, dU) is None

    U = flowfields(p=1, rho=1)
    dU = flowfields(grad_rho=(1, 0), grad_p=(1, 1))
    nptest.assert_almost_equal(eos.temperature_gradient(U, dU)(mip), (0, 7/2))

    dU = flowfields(grad_Ei=(1, 0))
    nptest.assert_almost_equal(eos.temperature_gradient(U, dU)(mip), (1.4, 0))


def test_characteristic_velocities(eos):
    U = flowfields()
    assert eos.characteristic_velocities(U, (1, 0)) is None

    U = flowfields(velocity=(1, 0), speed_of_sound=1.4)
    nptest.assert_almost_equal(eos.characteristic_velocities(U, (1, 0))(mip), (-0.4, 1, 1, 2.4))
    nptest.assert_almost_equal(eos.characteristic_velocities(U, (1, 0), "absolute")(mip), (0.4, 1, 1, 2.4))
    nptest.assert_almost_equal(eos.characteristic_velocities(U, (1, 0), "incoming")(mip), (-0.4, 0, 0, 0))
    nptest.assert_almost_equal(eos.characteristic_velocities(U, (1, 0), "outgoing")(mip), (0, 1, 1, 2.4))


def test_characteristic_variables(eos):
    U = flowfields()
    dU = flowfields()
    assert eos.characteristic_variables(U, dU, (1, 0)) is None

    U = flowfields(rho=2, c=1)
    dU = flowfields(grad_rho=(1, 0), grad_p=(0, 1), grad_u=ngs.CF((1, 0, 0, 1), dims=(2, 2)))
    nptest.assert_almost_equal(eos.characteristic_variables(U, dU, (1, 0))(mip), (-2.0, 1.0, 0.0, 2.0))


def test_characteristic_amplitudes(eos):
    U = flowfields()
    dU = flowfields()
    assert eos.characteristic_amplitudes(U, dU, (1, 0)) is None

    U = flowfields(rho=2, c=1, u=(2, 0))
    dU = flowfields(grad_rho=(1, 0), grad_p=(0, 1), grad_u=ngs.CF((1, 0, 0, 1), dims=(2, 2)))
    nptest.assert_almost_equal(eos.characteristic_amplitudes(U, dU, (1, 0))(mip), (-2.0, 2.0, 0.0, 6.0))


def test_primitive_from_conservative(eos):
    U = flowfields()
    assert eos.primitive_from_conservative(U) is None

    U = flowfields(rho=2, velocity=(2, 0))
    Minv = eos.primitive_from_conservative(U)(mip)
    nptest.assert_almost_equal(Minv, (1, 0, 0, 0, -1, 0.5, 0, 0, 0, 0, 0.5, 0, 0.8, -0.8, 0, 0.4))


def test_primitive_from_characteristic(eos):
    U = flowfields()
    assert eos.primitive_from_characteristic(U, (1, 0)) is None

    U = flowfields(rho=2, speed_of_sound=2)
    L = eos.primitive_from_characteristic(U, (1, 0))(mip)
    nptest.assert_almost_equal(L, (0.125, 0.25, 0, 0.125, -0.125, 0, 0, 0.125, 0, 0, -1, 0, 0.5, 0, 0, 0.5))


def test_primitive_convective_jacobian(eos):
    U = flowfields(rho=2, speed_of_sound=2, velocity=(1, 1))
    A = eos.primitive_convective_jacobian_x(U)(mip)
    nptest.assert_almost_equal(A, (1, 2, 0, 0, 0, 1, 0, 0.5, 0, 0, 1, 0, 0, 8, 0, 1))
    B = eos.primitive_convective_jacobian_y(U)(mip)
    nptest.assert_almost_equal(B, (1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0.5, 0, 0, 8, 1))


def test_conservative_from_primitive(eos):
    U = flowfields()
    assert eos.conservative_from_primitive(U) is None

    U = flowfields(rho=2, velocity=(2, 0))
    M = eos.conservative_from_primitive(U)(mip)
    nptest.assert_almost_equal(M, (1, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 2, 4, 0, 2.5))


def test_conservative_from_characteristic(eos):
    U = flowfields()
    assert eos.conservative_from_characteristic(U, (1, 0)) is None

    U = flowfields(rho=2, speed_of_sound=2, velocity=(2, 0))
    P = eos.conservative_from_characteristic(U, (1, 0))(mip)
    nptest.assert_almost_equal(P, (0.125, 0.25, 0,
                                   0.125, 0, 0.5, 0, 0.5,
                                   0, 0, -2, 0,
                                   1, 0.5, 0, 2))


def test_conservative_convective_jacobian(eos):
    U = flowfields(velocity=(2, 0), specific_energy=5)
    A = eos.conservative_convective_jacobian_x(U)(mip)
    nptest.assert_almost_equal(A, (0, 1, 0, 0, -3.2, 3.2, 0, 0.4, 0, 0, 2, 0, -10.8, 4.6, 0, 2.8))
    B = eos.conservative_convective_jacobian_y(U)(mip)
    nptest.assert_almost_equal(B, (0, 0, 1, 0, 0, 0, 2, 0, 0.8, -0.8, 0, 0.4, 0, 0, 6.2, 0))


def test_characteristic_from_primitive(eos):
    U = flowfields()
    assert eos.characteristic_from_primitive(U, (1, 0)) is None

    U = flowfields(rho=2, speed_of_sound=2)
    Linv = eos.characteristic_from_primitive(U, (1, 0))(mip)
    nptest.assert_almost_equal(Linv, (0, -4, 0, 1, 4, 0, 0, -1, 0, 0, -1, 0, 0, 4, 0, 1))


def test_characteristic_from_conservative(eos):
    U = flowfields()
    assert eos.characteristic_from_conservative(U, (1, 0)) is None

    U = flowfields(rho=2, speed_of_sound=2, velocity=(2, 0))
    Pinv = eos.characteristic_from_conservative(U, (1, 0))(mip)
    nptest.assert_almost_equal(Pinv, (4.8, -2.8, 0, 0.4,
                                      3.2, 0.8, 0, -0.4,
                                      0, 0, -0.5, 0,
                                      -3.2, 1.2, 0, 0.4))


def test_identity(eos):
    U = flowfields(rho=2, velocity=(2, 2))
    unit_vector = ngs.CF((1, 1))/ngs.sqrt(2)
    M = eos.conservative_from_primitive(U)
    Minv = eos.primitive_from_conservative(U)
    nptest.assert_almost_equal((M * Minv)(mip), ngs.Id(4)(mip))

    U = flowfields(rho=2, speed_of_sound=2)
    L = eos.primitive_from_characteristic(U, unit_vector)
    Linv = eos.characteristic_from_primitive(U, unit_vector)
    nptest.assert_almost_equal((L * Linv)(mip), ngs.Id(4)(mip))

    U = flowfields(rho=2, speed_of_sound=2, velocity=(2, 2))
    P = eos.conservative_from_characteristic(U, unit_vector)
    Pinv = eos.characteristic_from_conservative(U, unit_vector)
    nptest.assert_almost_equal((P * Pinv)(mip), ngs.Id(4)(mip))


def test_transformation(eos):
    U = flowfields(rho=2, speed_of_sound=2, velocity=(2, 2))
    unit_vector = ngs.CF((1, 1))/ngs.sqrt(2)
    M = eos.conservative_from_primitive(U)
    L = eos.primitive_from_characteristic(U, unit_vector)
    P = eos.conservative_from_characteristic(U, unit_vector)
    nptest.assert_almost_equal((M * L)(mip), P(mip))

    Minv = eos.primitive_from_conservative(U)
    Linv = eos.characteristic_from_primitive(U, unit_vector)
    Pinv = eos.characteristic_from_conservative(U, unit_vector)
    nptest.assert_almost_equal((Linv * Minv)(mip), Pinv(mip))
