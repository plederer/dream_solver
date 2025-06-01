import pytest
import numpy.testing as nptest
import ngsolve as ngs

from dream.compressible import CompressibleFlowSolver, flowfields

mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))
mip = mesh(0.5, 0.5)


def run_equation_test(cfg, name, throws=False, is_vector=False, extra=None):
    if throws:
        U = flowfields()
        with pytest.raises(ValueError):
            getattr(cfg, name)(U)
    value = 1
    if is_vector:
        value = (1, 0)
    U = flowfields()
    U.update({name: value})
    result = getattr(cfg, name)(U)(mip)
    if is_vector:
        nptest.assert_almost_equal(result, value)
    else:
        assert result == pytest.approx(value)
    if extra:
        extra(cfg, name)


@pytest.fixture()
def cfg():
    cfg = CompressibleFlowSolver(mesh)
    cfg.time = "transient"

    cfg.fem = "conservative"
    cfg.fem.method = "hdg"
    cfg.fem.mixed_method = "inactive"

    cfg.mach_number = 0.3
    cfg.reynolds_number = 2
    cfg.prandtl_number = 1
    cfg.equation_of_state = "ideal"
    cfg.equation_of_state.heat_capacity_ratio = 1.4
    cfg.dynamic_viscosity = "inviscid"
    cfg.scaling = "aerodynamic"
    cfg.scaling.dimensionful_values = {'length': 1.0, 'density': 1.293, 'velocity': 1.0,
                                       'speed_of_sound': 343.0, 'temperature': 293.15, 'pressure': 101325.0}
    cfg.riemann_solver = "lax_friedrich"

    yield cfg


def test_density(cfg):
    run_equation_test(cfg, "density", throws=True)


def test_velocity(cfg):
    def extra(cfg, name):
        U = flowfields(density=2, momentum=(2, 2))
        nptest.assert_almost_equal(cfg.velocity(U)(mip), (1, 1))
    run_equation_test(cfg, "velocity", throws=True, is_vector=True, extra=extra)


def test_momentum(cfg):
    def extra(cfg, name):
        U = flowfields(density=0.5, velocity=(1, 1))
        nptest.assert_almost_equal(cfg.momentum(U)(mip), (0.5, 0.5))
    run_equation_test(cfg, "momentum", throws=True, is_vector=True, extra=extra)


def test_pressure(cfg):
    run_equation_test(cfg, "pressure", throws=True)


def test_temperature(cfg):
    run_equation_test(cfg, "temperature", throws=True)


def test_inner_energy(cfg):
    def extra(cfg, name):
        U = flowfields(energy=2, kinetic_energy=1)
        assert cfg.inner_energy(U)(mip) == pytest.approx(1)
    run_equation_test(cfg, "inner_energy", throws=True, extra=extra)


def test_specific_inner_energy(cfg):
    def extra(cfg, name):
        U = flowfields(specific_energy=2, specific_kinetic_energy=1)
        assert cfg.specific_inner_energy(U)(mip) == pytest.approx(1)
        U = flowfields(inner_energy=2, density=1)
        assert cfg.specific_inner_energy(U)(mip) == pytest.approx(2)
    run_equation_test(cfg, "specific_inner_energy", throws=True, extra=extra)


def test_kinetic_energy(cfg):
    def extra(cfg, name):
        U = flowfields(density=2, velocity=(2, 2))
        assert cfg.kinetic_energy(U)(mip) == pytest.approx(8)
        U = flowfields(density=2, momentum=(2, 2))
        assert cfg.kinetic_energy(U)(mip) == pytest.approx(2)
        U = flowfields(energy=2, inner_energy=1)
        assert cfg.kinetic_energy(U)(mip) == pytest.approx(1)
        U = flowfields(specific_kinetic_energy=2, density=2)
        assert cfg.kinetic_energy(U)(mip) == pytest.approx(4)
    run_equation_test(cfg, "kinetic_energy", throws=True, extra=extra)


def test_specific_kinetic_energy(cfg):
    def extra(cfg, name):
        U = flowfields(velocity=(2, 2))
        assert cfg.specific_kinetic_energy(U)(mip) == pytest.approx(4)
        U = flowfields(density=2, momentum=(2, 2))
        assert cfg.specific_kinetic_energy(U)(mip) == pytest.approx(1)
        U = flowfields(specific_energy=2, specific_inner_energy=1)
        assert cfg.specific_kinetic_energy(U)(mip) == pytest.approx(1)
        U = flowfields(kinetic_energy=2, density=2)
        assert cfg.specific_kinetic_energy(U)(mip) == pytest.approx(1)
    run_equation_test(cfg, "specific_kinetic_energy", throws=True, extra=extra)


def test_energy(cfg):
    def extra(cfg, name):
        U = flowfields(specific_energy=2, density=2)
        assert cfg.energy(U)(mip) == pytest.approx(4)
        U = flowfields(kinetic_energy=2, inner_energy=2)
        assert cfg.energy(U)(mip) == pytest.approx(4)
    run_equation_test(cfg, "energy", throws=True, extra=extra)


def test_specific_energy(cfg):
    def extra(cfg, name):
        U = flowfields(energy=2, density=2)
        assert cfg.specific_energy(U)(mip) == pytest.approx(1)
        U = flowfields(specific_kinetic_energy=2, specific_inner_energy=2)
        assert cfg.specific_energy(U)(mip) == pytest.approx(4)
    run_equation_test(cfg, "specific_energy", throws=True, extra=extra)


def test_enthalpy(cfg):
    def extra(cfg, name):
        U = flowfields(pressure=2, energy=2)
        assert cfg.enthalpy(U)(mip) == pytest.approx(4)
        U = flowfields(specific_enthalpy=2, density=2)
        assert cfg.enthalpy(U)(mip) == pytest.approx(4)
    run_equation_test(cfg, "enthalpy", throws=True, extra=extra)


def test_specific_enthalpy(cfg):
    def extra(cfg, name):
        U = flowfields(enthalpy=2, density=2)
        assert cfg.specific_enthalpy(U)(mip) == pytest.approx(1)
    run_equation_test(cfg, "specific_enthalpy", throws=True, extra=extra)


def test_convective_flux(cfg):
    U = flowfields()
    with pytest.raises(Exception):
        cfg.get_convective_flux(U)
    U = flowfields(density=1, momentum=(1, 0), enthalpy=1, velocity=(1, 0), pressure=2)
    nptest.assert_almost_equal(cfg.get_convective_flux(U)(mip), (1, 0, 3, 0, 0, 2, 1, 0))


def test_diffusive_flux(cfg):
    cfg.dynamic_viscosity = "constant"
    U = flowfields()
    dU = flowfields()
    with pytest.raises(Exception):
        cfg.get_diffusive_flux(U, dU)
    U = flowfields(velocity=(1, 0))
    dU = flowfields(strain_rate_tensor=ngs.CF((0, 0.5, 0.5, 0), dims=(2, 2)), grad_T=(1, 1))
    nptest.assert_almost_equal(cfg.get_diffusive_flux(U, dU)(mip), (0, 0, 0, 0.5, 0.5, 0, 0.5, 1.0))


def test_transformation(cfg):
    U = flowfields(density=2, speed_of_sound=2, velocity=(2, 2), specific_energy=11.14285714)
    unit_vector = ngs.CF((1, 1))/ngs.sqrt(2)
    A = cfg.primitive_convective_jacobian_x(U)
    B = cfg.primitive_convective_jacobian_y(U)
    An_1 = A * unit_vector[0] + B * unit_vector[1]
    An_2 = cfg.get_primitive_convective_jacobian(U, unit_vector)
    nptest.assert_almost_equal(An_1(mip), An_2(mip))
    A = cfg.conservative_convective_jacobian_x(U)
    B = cfg.conservative_convective_jacobian_y(U)
    An_1 = A * unit_vector[0] + B * unit_vector[1]
    An_2 = cfg.get_conservative_convective_jacobian(U, unit_vector)
    nptest.assert_almost_equal(An_1(mip), An_2(mip))


def test_characteristic_identity(cfg):
    U = flowfields(velocity=(1, 0), speed_of_sound=1.4)
    unit_vector = (1, 0)
    with pytest.raises(ValueError):
        cfg.get_characteristic_identity(U, unit_vector, "unknown")(mip)
    nptest.assert_almost_equal(cfg.get_characteristic_identity(U, unit_vector, "incoming")(mip),
                               (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    nptest.assert_almost_equal(cfg.get_characteristic_identity(U, unit_vector, "outgoing")(mip),
                               (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))
    nptest.assert_almost_equal(cfg.get_characteristic_identity(U, unit_vector, None)(mip),
                               (-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))


def test_farfield_state(cfg):
    cfg.scaling = "aerodynamic"
    INF = cfg.get_farfield_fields((1, 0)).to_py(cfg.mesh)
    results = {"density": 1.0, "speed_of_sound": 10/3, "temperature": 1/(0.4 * 0.3**2),
               "pressure": 1/(1.4 * 0.3**2), "velocity": (1.0, 0.0), "inner_energy": 1/(1.4 * 0.3**2 * 0.4),
               "kinetic_energy": 0.5, "energy": 1/(1.4 * 0.3**2 * 0.4) + 0.5}
    for is_, exp_ in zip(INF.items(), results.items()):
        assert is_[0] == exp_[0]
        nptest.assert_almost_equal(is_[1], exp_[1])

    cfg.scaling = "aeroacoustic"
    INF = cfg.get_farfield_fields((1, 0)).to_py(cfg.mesh)
    results = {
        "density": 1.0, "speed_of_sound": 10 / 13, "temperature": 1 / (0.4 * (1 + 0.3) ** 2),
        "pressure": 1 / (1.4 * (1 + 0.3) ** 2),
        "velocity": (3 / 13, 0.0),
        "inner_energy": 1 / (1.4 * (1 + 0.3) ** 2 * 0.4),
        "kinetic_energy": 9 / 338, "energy": 1 / (1.4 * (1 + 0.3) ** 2 * 0.4) + 9 / 338}
    for is_, exp_ in zip(INF.items(), results.items()):
        assert is_[0] == exp_[0]
        nptest.assert_almost_equal(is_[1], exp_[1])

    cfg.scaling = "acoustic"
    INF = cfg.get_farfield_fields((1, 0)).to_py(cfg.mesh)
    results = {"density": 1.0, "speed_of_sound": 1, "temperature": 10/4,
               "pressure": 1/1.4, "velocity": (0.3, 0.0), "inner_energy": 10/5.6,
               "kinetic_energy": 0.045, "energy": 10/5.6 + 0.045}
    for is_, exp_ in zip(INF.items(), results.items()):
        assert is_[0] == exp_[0]
        nptest.assert_almost_equal(is_[1], exp_[1])
