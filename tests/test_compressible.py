from __future__ import annotations
import unittest
import numpy.testing as nptest
from tests import simplex
import ngsolve as ngs

from dream.formulations import compressible as cp


def test_equation(throws: bool = False, is_vector: bool = False):

    def wrapper(func):

        def test(self):
            name = func.__name__.split("test_")[1]

            if throws:
                state = cp.CompressibleState()
                with self.assertRaises(ValueError):
                    getattr(self.eq, name)(state)

            value = 1
            if is_vector:
                value = (1, 0)

            setattr(state, name, value)
            self.assertAlmostEqual(getattr(self.eq, name)(state)(self.mip), value)

            func(self)

        return test

    return wrapper


class Configuration(unittest.TestCase):
    ...


class IdealGas(unittest.TestCase):

    def setUp(self):
        self.eos = cp.IdealGas()
        self.eos.heat_capacity_ratio = 1.4

        self.mesh = simplex()
        self.mip = self.mesh(0.25, 0.25)

    def test_density(self):

        state = cp.CompressibleState()
        self.assertIs(self.eos.density(state), None)

        state = cp.CompressibleState(pressure=1, temperature=1)
        self.assertAlmostEqual(self.eos.density(state)(self.mip), 3.5)

        state = cp.CompressibleState(pressure=1, speed_of_sound=1)
        self.assertAlmostEqual(self.eos.density(state)(self.mip), 1.4)

        state = cp.CompressibleState(inner_energy=1, temperature=1)
        self.assertAlmostEqual(self.eos.density(state)(self.mip), 1.4)

    def test_pressure(self):

        state = cp.CompressibleState()
        self.assertIs(self.eos.pressure(state), None)

        state = cp.CompressibleState(density=1, temperature=1)
        self.assertAlmostEqual(self.eos.pressure(state)(self.mip), 2/7)

        state = cp.CompressibleState(inner_energy=1)
        self.assertAlmostEqual(self.eos.pressure(state)(self.mip), 0.4)

        state = cp.CompressibleState(density=1, speed_of_sound=1)
        self.assertAlmostEqual(self.eos.pressure(state)(self.mip), 5/7)

    def test_temperature(self):

        state = cp.CompressibleState()
        self.assertIs(self.eos.temperature(state), None)

        state = cp.CompressibleState(density=1, pressure=1)
        self.assertAlmostEqual(self.eos.temperature(state)(self.mip), 7/2)

        state = cp.CompressibleState(specific_inner_energy=1)
        self.assertAlmostEqual(self.eos.temperature(state)(self.mip), 1.4)

        state = cp.CompressibleState(speed_of_sound=1)
        self.assertAlmostEqual(self.eos.temperature(state)(self.mip), 5/2)

    def test_inner_energy(self):

        state = cp.CompressibleState()
        self.assertIs(self.eos.inner_energy(state), None)

        state = cp.CompressibleState(pressure=1)
        self.assertAlmostEqual(self.eos.inner_energy(state)(self.mip), 5/2)

        state = cp.CompressibleState(density=1, temperature=1)
        self.assertAlmostEqual(self.eos.inner_energy(state)(self.mip), 5/7)

    def test_specific_inner_energy(self):

        state = cp.CompressibleState()
        self.assertIs(self.eos.specific_inner_energy(state), None)

        state = cp.CompressibleState(temperature=1)
        self.assertAlmostEqual(self.eos.specific_inner_energy(state)(self.mip), 5/7)

        state = cp.CompressibleState(density=1, pressure=1)
        self.assertAlmostEqual(self.eos.specific_inner_energy(state)(self.mip), 5/2)

    def test_speed_of_sound(self):

        state = cp.CompressibleState()
        self.assertIs(self.eos.speed_of_sound(state), None)

        state = cp.CompressibleState(density=1, pressure=1)
        self.assertAlmostEqual(self.eos.speed_of_sound(state)(self.mip), ngs.sqrt(1.4))

        state = cp.CompressibleState(temperature=1)
        self.assertAlmostEqual(self.eos.speed_of_sound(state)(self.mip), ngs.sqrt(0.4))

        state = cp.CompressibleState(specific_inner_energy=1)
        self.assertAlmostEqual(self.eos.speed_of_sound(state)(self.mip), ngs.sqrt(2/7))

    def test_density_gradient(self):

        state = cp.CompressibleState()
        self.assertIs(self.eos.density_gradient(state), None)

        state = cp.CompressibleState(temperature=1, pressure=1, pressure_gradient=(1, 1), temperature_gradient=(1, 0))
        nptest.assert_almost_equal(self.eos.density_gradient(state)(self.mip), (0, 7/2))

        state = cp.CompressibleState(
            temperature=1, inner_energy=1, temperature_gradient=(1, 1),
            inner_energy_gradient=(1, 0))
        nptest.assert_almost_equal(self.eos.density_gradient(state)(self.mip), (0, -7/5))

    def test_pressure_gradient(self):

        state = cp.CompressibleState()
        self.assertIs(self.eos.pressure_gradient(state), None)

        state = cp.CompressibleState(temperature=1, density=1, density_gradient=(1, 1), temperature_gradient=(1, 0))
        nptest.assert_almost_equal(self.eos.pressure_gradient(state)(self.mip), (4/7, 2/7))

        state = cp.CompressibleState(inner_energy_gradient=(1, 0))
        nptest.assert_almost_equal(self.eos.pressure_gradient(state)(self.mip), (0.4, 0))

    def test_temperature_gradient(self):

        state = cp.CompressibleState()
        self.assertIs(self.eos.temperature_gradient(state), None)

        state = cp.CompressibleState(pressure=1, density=1, density_gradient=(1, 0), pressure_gradient=(1, 1))
        nptest.assert_almost_equal(self.eos.temperature_gradient(state)(self.mip), (0, 7/2))

        state = cp.CompressibleState(specific_inner_energy_gradient=(1, 0))
        nptest.assert_almost_equal(self.eos.temperature_gradient(state)(self.mip), (1.4, 0))

    def test_characteristic_velocities(self):

        state = cp.CompressibleState()
        self.assertIs(self.eos.characteristic_velocities(state, (1, 0)), None)

        state = cp.CompressibleState(velocity=(1, 0), speed_of_sound=1.4)
        nptest.assert_almost_equal(self.eos.characteristic_velocities(state, (1, 0))(self.mip), (-0.4, 1, 1, 2.4))
        nptest.assert_almost_equal(self.eos.characteristic_velocities(
            state, (1, 0), "absolute")(self.mip), (0.4, 1, 1, 2.4))
        nptest.assert_almost_equal(self.eos.characteristic_velocities(
            state, (1, 0), "incoming")(self.mip), (-0.4, 0, 0, 0))
        nptest.assert_almost_equal(self.eos.characteristic_velocities(
            state, (1, 0), "outgoing")(self.mip), (0, 1, 1, 2.4))

    def test_characteristic_variables(self):

        state = cp.CompressibleState()
        self.assertIs(self.eos.characteristic_variables(state, (1, 0)), None)

        state = cp.CompressibleState(density=2,
                                     speed_of_sound=1,
                                     density_gradient=(1, 0),
                                     pressure_gradient=(0, 1),
                                     velocity_gradient=(1, 0, 0, 1))
        nptest.assert_almost_equal(self.eos.characteristic_variables(state, (1, 0))(self.mip), (-2.0, 1.0, 0.0, 2.0))

    def test_characteristic_amplitudes(self):

        state = cp.CompressibleState()
        self.assertIs(self.eos.characteristic_amplitudes(state, (1, 0)), None)

        state = cp.CompressibleState(density=2,
                                     speed_of_sound=1,
                                     velocity=(2, 0),
                                     density_gradient=(1, 0),
                                     pressure_gradient=(0, 1),
                                     velocity_gradient=(1, 0, 0, 1))
        nptest.assert_almost_equal(self.eos.characteristic_amplitudes(state, (1, 0))(self.mip), (-2.0, 2.0, 0.0, 6.0))

    def test_primitive_from_conservative(self):
        state = cp.CompressibleState()
        self.assertIs(self.eos.primitive_from_conservative(state), None)

        state = cp.CompressibleState(density=2, velocity=(2, 0))
        Minv = self.eos.primitive_from_conservative(state)(self.mip)
        nptest.assert_almost_equal(Minv, (1, 0, 0, 0, -1, 0.5, 0, 0, 0, 0, 0.5, 0, 0.8, -0.8, 0, 0.4))

    def test_primitive_from_characteristic(self):
        state = cp.CompressibleState()
        self.assertIs(self.eos.primitive_from_characteristic(state, (1, 0)), None)

        state = cp.CompressibleState(density=2, speed_of_sound=2)
        L = self.eos.primitive_from_characteristic(state, (1, 0))(self.mip)
        nptest.assert_almost_equal(L, (0.125, 0.25, 0, 0.125, -0.125, 0, 0, 0.125, 0, 0, -1, 0, 0.5, 0, 0, 0.5))

    def test_primitive_convective_jacobian(self):
        state = cp.CompressibleState(density=2, speed_of_sound=2, velocity=(1, 1))

        A = self.eos.primitive_convective_jacobian_x(state)(self.mip)
        nptest.assert_almost_equal(A, (1, 2, 0, 0, 0, 1, 0, 0.5, 0, 0, 1, 0, 0, 8, 0, 1))

        B = self.eos.primitive_convective_jacobian_y(state)(self.mip)
        nptest.assert_almost_equal(B, (1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0.5, 0, 0, 8, 1))

    def test_conservative_from_primitive(self):
        state = cp.CompressibleState()
        self.assertIs(self.eos.conservative_from_primitive(state), None)

        state = cp.CompressibleState(density=2, velocity=(2, 0))
        M = self.eos.conservative_from_primitive(state)(self.mip)
        nptest.assert_almost_equal(M, (1, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 2, 4, 0, 2.5))

    def test_conservative_from_characteristic(self):
        state = cp.CompressibleState()
        self.assertIs(self.eos.conservative_from_characteristic(state, (1, 0)), None)

        state = cp.CompressibleState(density=2, speed_of_sound=2, velocity=(2, 0))
        P = self.eos.conservative_from_characteristic(state, (1, 0))(self.mip)
        nptest.assert_almost_equal(P, (0.125, 0.25, 0,
                                       0.125, 0, 0.5, 0, 0.5,
                                       0, 0, -2, 0,
                                       1, 0.5, 0, 2))

    def test_conservative_convective_jacobian(self):
        state = cp.CompressibleState(velocity=(2, 0), specific_energy=5)

        A = self.eos.conservative_convective_jacobian_x(state)(self.mip)
        nptest.assert_almost_equal(A, (0, 1, 0, 0, -3.2, 3.2, 0, 0.4, 0, 0, 2, 0, -10.8, 4.6, 0, 2.8))

        B = self.eos.conservative_convective_jacobian_y(state)(self.mip)
        nptest.assert_almost_equal(B, (0, 0, 1, 0, 0, 0, 2, 0, 0.8, -0.8, 0, 0.4, 0, 0, 6.2, 0))

    def test_characteristic_from_primitive(self):
        state = cp.CompressibleState()
        self.assertIs(self.eos.characteristic_from_primitive(state, (1, 0)), None)

        state = cp.CompressibleState(density=2, speed_of_sound=2)
        Linv = self.eos.characteristic_from_primitive(state, (1, 0))(self.mip)
        nptest.assert_almost_equal(Linv, (0, -4, 0, 1, 4, 0, 0, -1, 0, 0, -1, 0, 0, 4, 0, 1))

    def test_characteristic_from_conservative(self):
        state = cp.CompressibleState()
        self.assertIs(self.eos.characteristic_from_conservative(state, (1, 0)), None)

        state = cp.CompressibleState(density=2, speed_of_sound=2, velocity=(2, 0))
        Pinv = self.eos.characteristic_from_conservative(state, (1, 0))(self.mip)
        nptest.assert_almost_equal(Pinv, (4.8, -2.8, 0, 0.4,
                                          3.2, 0.8, 0, -0.4,
                                          0, 0, -0.5, 0,
                                          -3.2, 1.2, 0, 0.4))

    def test_identity(self):
        state = cp.CompressibleState(density=2, velocity=(2, 2))
        unit_vector = ngs.CF((1, 1))/ngs.sqrt(2)

        M = self.eos.conservative_from_primitive(state)
        Minv = self.eos.primitive_from_conservative(state)
        nptest.assert_almost_equal((M * Minv)(self.mip), ngs.Id(4)(self.mip))

        state = cp.CompressibleState(density=2, speed_of_sound=2)
        L = self.eos.primitive_from_characteristic(state, unit_vector)
        Linv = self.eos.characteristic_from_primitive(state, unit_vector)
        nptest.assert_almost_equal((L * Linv)(self.mip), ngs.Id(4)(self.mip))

        state = cp.CompressibleState(density=2, speed_of_sound=2, velocity=(2, 2))
        P = self.eos.conservative_from_characteristic(state, unit_vector)
        Pinv = self.eos.characteristic_from_conservative(state, unit_vector)
        nptest.assert_almost_equal((P * Pinv)(self.mip), ngs.Id(4)(self.mip))

    def test_transformation(self):
        state = cp.CompressibleState(density=2, speed_of_sound=2, velocity=(2, 2))
        unit_vector = ngs.CF((1, 1))/ngs.sqrt(2)

        M = self.eos.conservative_from_primitive(state)
        L = self.eos.primitive_from_characteristic(state, unit_vector)
        P = self.eos.conservative_from_characteristic(state, unit_vector)
        nptest.assert_almost_equal((M * L)(self.mip), P(self.mip))

        Minv = self.eos.primitive_from_conservative(state)
        Linv = self.eos.characteristic_from_primitive(state, unit_vector)
        Pinv = self.eos.characteristic_from_conservative(state, unit_vector)
        nptest.assert_almost_equal((Linv * Minv)(self.mip), Pinv(self.mip))


class Inviscid(unittest.TestCase):

    def setUp(self) -> None:
        self.mu = cp.Inviscid()

    def test_is_inviscid(self):
        self.assertTrue(self.mu.is_inviscid)

    def test_viscosity(self):
        with self.assertRaises(TypeError):
            self.mu.viscosity(cp.CompressibleState())


class Constant(unittest.TestCase):

    def setUp(self) -> None:
        self.mu = cp.Constant()

    def test_is_inviscid(self):
        self.assertFalse(self.mu.is_inviscid)

    def test_viscosity(self):
        state = cp.CompressibleState()
        self.assertAlmostEqual(self.mu.viscosity(state), 1)


class Sutherland(unittest.TestCase):

    def setUp(self) -> None:
        self.mu = cp.Sutherland()
        self.mu.measurement_temperature = 1

        self.cfg = cp.CompressibleFlowConfiguration()
        self.cfg.Mach_number = 1
        self.cfg.equation_of_state.heat_capacity_ratio = 1.4

        self.mesh = simplex()
        self.mip = self.mesh(0.25, 0.25)

    def test_is_inviscid(self):
        self.assertFalse(self.mu.is_inviscid)

    def test_viscosity(self):

        state = cp.CompressibleState()
        self.assertIs(self.mu.viscosity(state, self.cfg.equations), None)

        state = cp.CompressibleState(temperature=1)

        self.cfg.scaling = "aerodynamic"
        self.cfg.scaling.dimensional_infinity_values.temperature = 1
        self.assertAlmostEqual(self.mu.viscosity(state, self.cfg.equations)(self.mip), (0.4)**(3/2) * (2/0.4)/(1+1/0.4))

        self.cfg.scaling = "acoustic"
        self.cfg.scaling.dimensional_infinity_values.temperature = 1
        self.assertAlmostEqual(self.mu.viscosity(state, self.cfg.equations)(self.mip), (0.4)**(3/2) * (2/0.4)/(1+1/0.4))

        self.cfg.scaling = "aeroacoustic"
        self.cfg.scaling.dimensional_infinity_values.temperature = 1
        self.assertAlmostEqual(self.mu.viscosity(state, self.cfg.equations)(self.mip), (1.6)**(3/2) * (2/1.6)/(1+1/1.6))


class Aerodynamic(unittest.TestCase):

    def setUp(self) -> None:
        self.scaling = cp.Aerodynamic()

    def test_velocity_magnitude(self):
        self.assertAlmostEqual(self.scaling.velocity_magnitude(0.1), 1)

    def test_speed_of_sound(self):
        self.assertAlmostEqual(self.scaling.speed_of_sound(0.2), 5)

        with self.assertRaises(ValueError):
            self.scaling.speed_of_sound(0)


class Acoustic(unittest.TestCase):

    def setUp(self) -> None:
        self.scaling = cp.Acoustic()

    def test_velocity_magnitude(self):
        self.assertAlmostEqual(self.scaling.velocity_magnitude(0.1), 0.1)

    def test_speed_of_sound(self):
        self.assertAlmostEqual(self.scaling.speed_of_sound(0.2), 1)


class Aeroacoustic(unittest.TestCase):

    def setUp(self) -> None:
        self.scaling = cp.Aeroacoustic()

    def test_velocity_magnitude(self):
        self.assertAlmostEqual(self.scaling.velocity_magnitude(0.1), 0.1/(1 + 0.1))

    def test_speed_of_sound(self):
        self.assertAlmostEqual(self.scaling.speed_of_sound(0.2), 1/(1 + 0.2))


class CompressibleEquations(unittest.TestCase):

    def setUp(self) -> None:
        self.eq = cp.CompressibleEquations()

        self.mesh = simplex()
        self.mip = self.mesh(0.25, 0.25)

        self.eq.cfg.Reynolds_number = 2
        self.eq.cfg.Prandtl_number = 1
        self.eq.cfg.dynamic_viscosity = "constant"

    @test_equation(throws=True)
    def test_density(self):
        ...

    @test_equation(throws=True, is_vector=True)
    def test_velocity(self):
        state = cp.CompressibleState(density=2, momentum=(2, 2))
        nptest.assert_almost_equal(self.eq.velocity(state)(self.mip), (1, 1))

    @test_equation(throws=True, is_vector=True)
    def test_momentum(self):
        state = cp.CompressibleState(density=0.5, velocity=(1, 1))
        nptest.assert_almost_equal(self.eq.momentum(state)(self.mip), (0.5, 0.5))

    @test_equation(throws=True)
    def test_pressure(self):
        ...

    @test_equation(throws=True)
    def test_temperature(self):
        ...

    @test_equation(throws=True)
    def test_inner_energy(self):
        state = cp.CompressibleState(energy=2, kinetic_energy=1)
        self.assertAlmostEqual(self.eq.inner_energy(state)(self.mip), 1)

    @test_equation(throws=True)
    def test_specific_inner_energy(self):
        state = cp.CompressibleState(specific_energy=2, specific_kinetic_energy=1)
        self.assertAlmostEqual(self.eq.specific_inner_energy(state)(self.mip), 1)

        state = cp.CompressibleState(inner_energy=2, density=1)
        self.assertAlmostEqual(self.eq.specific_inner_energy(state)(self.mip), 2)

    @test_equation(throws=True)
    def test_kinetic_energy(self):
        state = cp.CompressibleState(density=2, velocity=(2, 2))
        self.assertAlmostEqual(self.eq.kinetic_energy(state)(self.mip), 8)

        state = cp.CompressibleState(density=2, momentum=(2, 2))
        self.assertAlmostEqual(self.eq.kinetic_energy(state)(self.mip), 2)

        state = cp.CompressibleState(energy=2, inner_energy=1)
        self.assertAlmostEqual(self.eq.kinetic_energy(state)(self.mip), 1)

        state = cp.CompressibleState(specific_kinetic_energy=2, density=2)
        self.assertAlmostEqual(self.eq.kinetic_energy(state)(self.mip), 4)

    @test_equation(throws=True)
    def test_specific_kinetic_energy(self):
        state = cp.CompressibleState(velocity=(2, 2))
        self.assertAlmostEqual(self.eq.specific_kinetic_energy(state)(self.mip), 4)

        state = cp.CompressibleState(density=2, momentum=(2, 2))
        self.assertAlmostEqual(self.eq.specific_kinetic_energy(state)(self.mip), 1)

        state = cp.CompressibleState(specific_energy=2, specific_inner_energy=1)
        self.assertAlmostEqual(self.eq.specific_kinetic_energy(state)(self.mip), 1)

        state = cp.CompressibleState(kinetic_energy=2, density=2)
        self.assertAlmostEqual(self.eq.specific_kinetic_energy(state)(self.mip), 1)

    @test_equation(throws=True)
    def test_energy(self):
        state = cp.CompressibleState(specific_energy=2, density=2)
        self.assertAlmostEqual(self.eq.energy(state)(self.mip), 4)

        state = cp.CompressibleState(kinetic_energy=2, inner_energy=2)
        self.assertAlmostEqual(self.eq.energy(state)(self.mip), 4)

    @test_equation(throws=True)
    def test_specific_energy(self):
        state = cp.CompressibleState(energy=2, density=2)
        self.assertAlmostEqual(self.eq.specific_energy(state)(self.mip), 1)

        state = cp.CompressibleState(specific_kinetic_energy=2, specific_inner_energy=2)
        self.assertAlmostEqual(self.eq.specific_energy(state)(self.mip), 4)

    @test_equation(throws=True)
    def test_enthalpy(self):
        state = cp.CompressibleState(pressure=2, energy=2)
        self.assertAlmostEqual(self.eq.enthalpy(state)(self.mip), 4)

        state = cp.CompressibleState(specific_enthalpy=2, density=2)
        self.assertAlmostEqual(self.eq.enthalpy(state)(self.mip), 4)

    @test_equation(throws=True)
    def test_specific_enthalpy(self):
        state = cp.CompressibleState(enthalpy=2, density=2)
        self.assertAlmostEqual(self.eq.specific_enthalpy(state)(self.mip), 1)

    def test_convective_flux(self):

        state = cp.CompressibleState()
        with self.assertRaises(Exception):
            self.eq.convective_flux(state)

        state = cp.CompressibleState(density=1, momentum=(1, 0), enthalpy=1, velocity=(1, 0), pressure=2)
        nptest.assert_almost_equal(self.eq.convective_flux(state)(self.mip), (1, 0, 3, 0, 0, 2, 1, 0))

    def test_diffusive_flux(self):

        state = cp.CompressibleState()
        with self.assertRaises(Exception):
            self.eq.convective_flux(state)

        state = cp.CompressibleState(velocity=(1, 0), deviatoric_stress_tensor=(0, 1, 1, 0), heat_flux=(1, 1))
        nptest.assert_almost_equal(self.eq.diffusive_flux(state)(self.mip), (0, 0, 0, 0.5, 0.5, 0, -0.5, 0))

    def test_transformation(self):
        state = cp.CompressibleState(density=2, speed_of_sound=2, velocity=(2, 2), specific_energy=11.14285714)
        unit_vector = ngs.CF((1, 1))/ngs.sqrt(2)

        A = self.eq.primitive_convective_jacobian_x(state)
        B = self.eq.primitive_convective_jacobian_y(state)
        An_1 = A * unit_vector[0] + B * unit_vector[1]
        An_2 = self.eq.get_primitive_convective_jacobian(state, unit_vector)

        nptest.assert_almost_equal(An_1(self.mip), An_2(self.mip))

        A = self.eq.conservative_convective_jacobian_x(state)
        B = self.eq.conservative_convective_jacobian_y(state)
        An_1 = A * unit_vector[0] + B * unit_vector[1]
        An_2 = self.eq.get_conservative_convective_jacobian(state, unit_vector)

        nptest.assert_almost_equal(An_1(self.mip), An_2(self.mip))

    def test_characteristic_identity(self):
        state = cp.CompressibleState(velocity=(1, 0), speed_of_sound=1.4)
        unit_vector = (1, 0)

        with self.assertRaises(ValueError):
            self.eq.get_characteristic_identity(state, unit_vector)(self.mip)

        nptest.assert_almost_equal(self.eq.get_characteristic_identity(state, unit_vector, "incoming")(
            self.mip), (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        nptest.assert_almost_equal(self.eq.get_characteristic_identity(state, unit_vector, "outgoing")(
            self.mip), (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))

    def test_farfield_state(self):

        self.eq.cfg.scaling = "aerodynamic"
        INF = self.eq.get_farfield_state((1, 0)).to_python()
        results = {"density": 1.0, "speed_of_sound": 10/3, "temperature": 1/(0.4 * 0.3**2),
                   "pressure": 1/(1.4 * 0.3**2), "velocity": (1.0, 0.0), "inner_energy": 1/(1.4 * 0.3**2 * 0.4),
                   "kinetic_energy": 0.5, "energy": 1/(1.4 * 0.3**2 * 0.4) + 0.5}

        for is_, exp_ in zip(INF.items(), results.items()):
            self.assertEqual(is_[0], exp_[0])
            nptest.assert_almost_equal(is_[1], exp_[1])

        self.eq.cfg.scaling = "aeroacoustic"
        INF = self.eq.get_farfield_state((1, 0)).to_python()
        results = {
            "density": 1.0, "speed_of_sound": 10 / 13, "temperature": 1 / (0.4 * (1 + 0.3) ** 2),
            "pressure": 1 / (1.4 * (1 + 0.3) ** 2),
            "velocity": (3 / 13, 0.0),
            "inner_energy": 1 / (1.4 * (1 + 0.3) ** 2 * 0.4),
            "kinetic_energy": 9 / 338, "energy": 1 / (1.4 * (1 + 0.3) ** 2 * 0.4) + 9 / 338}

        for is_, exp_ in zip(INF.items(), results.items()):
            self.assertEqual(is_[0], exp_[0])
            nptest.assert_almost_equal(is_[1], exp_[1])

        self.eq.cfg.scaling = "acoustic"
        INF = self.eq.get_farfield_state((1, 0)).to_python()
        results = {"density": 1.0, "speed_of_sound": 1, "temperature": 10/4,
                   "pressure": 1/1.4, "velocity": (0.3, 0.0), "inner_energy": 10/5.6,
                   "kinetic_energy": 0.045, "energy": 10/5.6 + 0.045}

        for is_, exp_ in zip(INF.items(), results.items()):
            self.assertEqual(is_[0], exp_[0])
            nptest.assert_almost_equal(is_[1], exp_[1])
