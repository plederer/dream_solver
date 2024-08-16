from __future__ import annotations
import unittest
import ngsolve as ngs
import numpy.testing as nptest

from tests import simplex
from dream.compressible.eos import IdealGas
from dream.compressible.state import CompressibleState, CompressibleStateGradient


class TestIdealGas(unittest.TestCase):

    def setUp(self):
        self.eos = IdealGas()
        self.eos.heat_capacity_ratio = 1.4

        self.mesh = simplex()
        self.mip = self.mesh(0.25, 0.25)

    def test_density(self):

        U = CompressibleState()
        self.assertIs(self.eos.density(U), None)

        U = CompressibleState(p=1, T=1)
        self.assertAlmostEqual(self.eos.density(U)(self.mip), 3.5)

        U = CompressibleState(p=1, speed_of_sound=1)
        self.assertAlmostEqual(self.eos.density(U)(self.mip), 1.4)

        U = CompressibleState(inner_energy=1, T=1)
        self.assertAlmostEqual(self.eos.density(U)(self.mip), 1.4)

    def test_pressure(self):

        U = CompressibleState()
        self.assertIs(self.eos.pressure(U), None)

        U = CompressibleState(rho=1, T=1)
        self.assertAlmostEqual(self.eos.pressure(U)(self.mip), 2/7)

        U = CompressibleState(rho_Ei=1)
        self.assertAlmostEqual(self.eos.pressure(U)(self.mip), 0.4)

        U = CompressibleState(rho=1, c=1)
        self.assertAlmostEqual(self.eos.pressure(U)(self.mip), 5/7)

    def test_temperature(self):

        U = CompressibleState()
        self.assertIs(self.eos.temperature(U), None)

        U = CompressibleState(rho=1, p=1)
        self.assertAlmostEqual(self.eos.temperature(U)(self.mip), 7/2)

        U = CompressibleState(Ei=1)
        self.assertAlmostEqual(self.eos.temperature(U)(self.mip), 1.4)

        U = CompressibleState(c=1)
        self.assertAlmostEqual(self.eos.temperature(U)(self.mip), 5/2)

    def test_inner_energy(self):

        U = CompressibleState()
        self.assertIs(self.eos.inner_energy(U), None)

        U = CompressibleState(p=1)
        self.assertAlmostEqual(self.eos.inner_energy(U)(self.mip), 5/2)

        U = CompressibleState(rho=1, T=1)
        self.assertAlmostEqual(self.eos.inner_energy(U)(self.mip), 5/7)

    def test_specific_inner_energy(self):

        U = CompressibleState()
        self.assertIs(self.eos.specific_inner_energy(U), None)

        U = CompressibleState(T=1)
        self.assertAlmostEqual(self.eos.specific_inner_energy(U)(self.mip), 5/7)

        U = CompressibleState(rho=1, p=1)
        self.assertAlmostEqual(self.eos.specific_inner_energy(U)(self.mip), 5/2)

    def test_speed_of_sound(self):

        U = CompressibleState()
        self.assertIs(self.eos.speed_of_sound(U), None)

        U = CompressibleState(rho=1, p=1)
        self.assertAlmostEqual(self.eos.speed_of_sound(U)(self.mip), ngs.sqrt(1.4))

        U = CompressibleState(T=1)
        self.assertAlmostEqual(self.eos.speed_of_sound(U)(self.mip), ngs.sqrt(0.4))

        U = CompressibleState(specific_inner_energy=1)
        self.assertAlmostEqual(self.eos.speed_of_sound(U)(self.mip), ngs.sqrt(2/7))

    def test_density_gradient(self):

        U = CompressibleState()
        dU = CompressibleStateGradient()
        self.assertIs(self.eos.density_gradient(U, dU), None)

        U = CompressibleState(T=1, p=1)
        dU = CompressibleStateGradient(p=(1, 1), T=(1, 0))
        nptest.assert_almost_equal(self.eos.density_gradient(U, dU)(self.mip), (0, 7/2))

        U = CompressibleState(T=1, rho_Ei=1)
        dU = CompressibleStateGradient(T=(1, 1), rho_Ei=(1, 0))
        nptest.assert_almost_equal(self.eos.density_gradient(U, dU)(self.mip), (0, -7/5))

    def test_pressure_gradient(self):

        U = CompressibleState()
        dU = CompressibleStateGradient()
        self.assertIs(self.eos.pressure_gradient(U, dU), None)

        U = CompressibleState(T=1, rho=1)
        dU = CompressibleStateGradient(rho=(1, 1), T=(1, 0))
        nptest.assert_almost_equal(self.eos.pressure_gradient(U, dU)(self.mip), (4/7, 2/7))

        dU = CompressibleStateGradient(rho_Ei=(1, 0))
        nptest.assert_almost_equal(self.eos.pressure_gradient(U, dU)(self.mip), (0.4, 0))

    def test_temperature_gradient(self):

        U = CompressibleState()
        dU = CompressibleStateGradient()
        self.assertIs(self.eos.temperature_gradient(U, dU), None)

        U = CompressibleState(p=1, rho=1)
        dU = CompressibleStateGradient(rho=(1, 0), p=(1, 1))
        nptest.assert_almost_equal(self.eos.temperature_gradient(U, dU)(self.mip), (0, 7/2))

        dU = CompressibleStateGradient(Ei=(1, 0))
        nptest.assert_almost_equal(self.eos.temperature_gradient(U, dU)(self.mip), (1.4, 0))

    def test_characteristic_velocities(self):

        U = CompressibleState()
        self.assertIs(self.eos.characteristic_velocities(U, (1, 0)), None)

        U = CompressibleState(velocity=(1, 0), speed_of_sound=1.4)
        nptest.assert_almost_equal(self.eos.characteristic_velocities(U, (1, 0))(self.mip), (-0.4, 1, 1, 2.4))
        nptest.assert_almost_equal(self.eos.characteristic_velocities(
            U, (1, 0), "absolute")(self.mip), (0.4, 1, 1, 2.4))
        nptest.assert_almost_equal(self.eos.characteristic_velocities(
            U, (1, 0), "incoming")(self.mip), (-0.4, 0, 0, 0))
        nptest.assert_almost_equal(self.eos.characteristic_velocities(
            U, (1, 0), "outgoing")(self.mip), (0, 1, 1, 2.4))

    def test_characteristic_variables(self):

        U = CompressibleState()
        dU = CompressibleStateGradient()
        self.assertIs(self.eos.characteristic_variables(U, dU, (1, 0)), None)

        U = CompressibleState(rho=2, c=1)
        dU = CompressibleStateGradient(rho=(1, 0), p=(0, 1), u=(1, 0, 0, 1))
        nptest.assert_almost_equal(self.eos.characteristic_variables(U, dU, (1, 0))(self.mip), (-2.0, 1.0, 0.0, 2.0))

    def test_characteristic_amplitudes(self):

        U = CompressibleState()
        dU = CompressibleStateGradient()
        self.assertIs(self.eos.characteristic_amplitudes(U, dU, (1, 0)), None)

        U = CompressibleState(rho=2, c=1, u=(2, 0))
        dU = CompressibleStateGradient(rho=(1, 0), p=(0, 1), u=(1, 0, 0, 1))
        nptest.assert_almost_equal(self.eos.characteristic_amplitudes(U, dU, (1, 0))(self.mip), (-2.0, 2.0, 0.0, 6.0))

    def test_primitive_from_conservative(self):
        U = CompressibleState()
        self.assertIs(self.eos.primitive_from_conservative(U), None)

        U = CompressibleState(rho=2, velocity=(2, 0))
        Minv = self.eos.primitive_from_conservative(U)(self.mip)
        nptest.assert_almost_equal(Minv, (1, 0, 0, 0, -1, 0.5, 0, 0, 0, 0, 0.5, 0, 0.8, -0.8, 0, 0.4))

    def test_primitive_from_characteristic(self):
        U = CompressibleState()
        self.assertIs(self.eos.primitive_from_characteristic(U, (1, 0)), None)

        U = CompressibleState(rho=2, speed_of_sound=2)
        L = self.eos.primitive_from_characteristic(U, (1, 0))(self.mip)
        nptest.assert_almost_equal(L, (0.125, 0.25, 0, 0.125, -0.125, 0, 0, 0.125, 0, 0, -1, 0, 0.5, 0, 0, 0.5))

    def test_primitive_convective_jacobian(self):
        U = CompressibleState(rho=2, speed_of_sound=2, velocity=(1, 1))

        A = self.eos.primitive_convective_jacobian_x(U)(self.mip)
        nptest.assert_almost_equal(A, (1, 2, 0, 0, 0, 1, 0, 0.5, 0, 0, 1, 0, 0, 8, 0, 1))

        B = self.eos.primitive_convective_jacobian_y(U)(self.mip)
        nptest.assert_almost_equal(B, (1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0.5, 0, 0, 8, 1))

    def test_conservative_from_primitive(self):
        U = CompressibleState()
        self.assertIs(self.eos.conservative_from_primitive(U), None)

        U = CompressibleState(rho=2, velocity=(2, 0))
        M = self.eos.conservative_from_primitive(U)(self.mip)
        nptest.assert_almost_equal(M, (1, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 2, 4, 0, 2.5))

    def test_conservative_from_characteristic(self):
        U = CompressibleState()
        self.assertIs(self.eos.conservative_from_characteristic(U, (1, 0)), None)

        U = CompressibleState(rho=2, speed_of_sound=2, velocity=(2, 0))
        P = self.eos.conservative_from_characteristic(U, (1, 0))(self.mip)
        nptest.assert_almost_equal(P, (0.125, 0.25, 0,
                                       0.125, 0, 0.5, 0, 0.5,
                                       0, 0, -2, 0,
                                       1, 0.5, 0, 2))

    def test_conservative_convective_jacobian(self):
        U = CompressibleState(velocity=(2, 0), specific_energy=5)

        A = self.eos.conservative_convective_jacobian_x(U)(self.mip)
        nptest.assert_almost_equal(A, (0, 1, 0, 0, -3.2, 3.2, 0, 0.4, 0, 0, 2, 0, -10.8, 4.6, 0, 2.8))

        B = self.eos.conservative_convective_jacobian_y(U)(self.mip)
        nptest.assert_almost_equal(B, (0, 0, 1, 0, 0, 0, 2, 0, 0.8, -0.8, 0, 0.4, 0, 0, 6.2, 0))

    def test_characteristic_from_primitive(self):
        U = CompressibleState()
        self.assertIs(self.eos.characteristic_from_primitive(U, (1, 0)), None)

        U = CompressibleState(rho=2, speed_of_sound=2)
        Linv = self.eos.characteristic_from_primitive(U, (1, 0))(self.mip)
        nptest.assert_almost_equal(Linv, (0, -4, 0, 1, 4, 0, 0, -1, 0, 0, -1, 0, 0, 4, 0, 1))

    def test_characteristic_from_conservative(self):
        U = CompressibleState()
        self.assertIs(self.eos.characteristic_from_conservative(U, (1, 0)), None)

        U = CompressibleState(rho=2, speed_of_sound=2, velocity=(2, 0))
        Pinv = self.eos.characteristic_from_conservative(U, (1, 0))(self.mip)
        nptest.assert_almost_equal(Pinv, (4.8, -2.8, 0, 0.4,
                                          3.2, 0.8, 0, -0.4,
                                          0, 0, -0.5, 0,
                                          -3.2, 1.2, 0, 0.4))

    def test_identity(self):
        U = CompressibleState(rho=2, velocity=(2, 2))
        unit_vector = ngs.CF((1, 1))/ngs.sqrt(2)

        M = self.eos.conservative_from_primitive(U)
        Minv = self.eos.primitive_from_conservative(U)
        nptest.assert_almost_equal((M * Minv)(self.mip), ngs.Id(4)(self.mip))

        U = CompressibleState(rho=2, speed_of_sound=2)
        L = self.eos.primitive_from_characteristic(U, unit_vector)
        Linv = self.eos.characteristic_from_primitive(U, unit_vector)
        nptest.assert_almost_equal((L * Linv)(self.mip), ngs.Id(4)(self.mip))

        U = CompressibleState(rho=2, speed_of_sound=2, velocity=(2, 2))
        P = self.eos.conservative_from_characteristic(U, unit_vector)
        Pinv = self.eos.characteristic_from_conservative(U, unit_vector)
        nptest.assert_almost_equal((P * Pinv)(self.mip), ngs.Id(4)(self.mip))

    def test_transformation(self):
        U = CompressibleState(rho=2, speed_of_sound=2, velocity=(2, 2))
        unit_vector = ngs.CF((1, 1))/ngs.sqrt(2)

        M = self.eos.conservative_from_primitive(U)
        L = self.eos.primitive_from_characteristic(U, unit_vector)
        P = self.eos.conservative_from_characteristic(U, unit_vector)
        nptest.assert_almost_equal((M * L)(self.mip), P(self.mip))

        Minv = self.eos.primitive_from_conservative(U)
        Linv = self.eos.characteristic_from_primitive(U, unit_vector)
        Pinv = self.eos.characteristic_from_conservative(U, unit_vector)
        nptest.assert_almost_equal((Linv * Minv)(self.mip), Pinv(self.mip))
