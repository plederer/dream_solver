from __future__ import annotations
import unittest
import ngsolve as ngs
import numpy.testing as nptest

from dream.compressible.config import flowstate
from tests.compressible.setup import cfg, mip


class TestIdealGas(unittest.TestCase):

    def setUp(self):
        cfg.pde.equation_of_state = "ideal"
        cfg.pde.equation_of_state.heat_capacity_ratio = 1.4

        self.eos = cfg.pde.equation_of_state

    def test_density(self):

        U = flowstate()
        self.assertIs(self.eos.density(U), None)

        U = flowstate(p=1, T=1)
        self.assertAlmostEqual(self.eos.density(U)(mip), 3.5)

        U = flowstate(p=1, speed_of_sound=1)
        self.assertAlmostEqual(self.eos.density(U)(mip), 1.4)

        U = flowstate(inner_energy=1, T=1)
        self.assertAlmostEqual(self.eos.density(U)(mip), 1.4)

    def test_pressure(self):

        U = flowstate()
        self.assertIs(self.eos.pressure(U), None)

        U = flowstate(rho=1, T=1)
        self.assertAlmostEqual(self.eos.pressure(U)(mip), 2/7)

        U = flowstate(rho_Ei=1)
        self.assertAlmostEqual(self.eos.pressure(U)(mip), 0.4)

        U = flowstate(rho=1, c=1)
        self.assertAlmostEqual(self.eos.pressure(U)(mip), 5/7)

    def test_temperature(self):

        U = flowstate()
        self.assertIs(self.eos.temperature(U), None)

        U = flowstate(rho=1, p=1)
        self.assertAlmostEqual(self.eos.temperature(U)(mip), 7/2)

        U = flowstate(Ei=1)
        self.assertAlmostEqual(self.eos.temperature(U)(mip), 1.4)

        U = flowstate(c=1)
        self.assertAlmostEqual(self.eos.temperature(U)(mip), 5/2)

    def test_inner_energy(self):

        U = flowstate()
        self.assertIs(self.eos.inner_energy(U), None)

        U = flowstate(p=1)
        self.assertAlmostEqual(self.eos.inner_energy(U)(mip), 5/2)

        U = flowstate(rho=1, T=1)
        self.assertAlmostEqual(self.eos.inner_energy(U)(mip), 5/7)

    def test_specific_inner_energy(self):

        U = flowstate()
        self.assertIs(self.eos.specific_inner_energy(U), None)

        U = flowstate(T=1)
        self.assertAlmostEqual(self.eos.specific_inner_energy(U)(mip), 5/7)

        U = flowstate(rho=1, p=1)
        self.assertAlmostEqual(self.eos.specific_inner_energy(U)(mip), 5/2)

    def test_speed_of_sound(self):

        U = flowstate()
        self.assertIs(self.eos.speed_of_sound(U), None)

        U = flowstate(rho=1, p=1)
        self.assertAlmostEqual(self.eos.speed_of_sound(U)(mip), ngs.sqrt(1.4))

        U = flowstate(T=1)
        self.assertAlmostEqual(self.eos.speed_of_sound(U)(mip), ngs.sqrt(0.4))

        U = flowstate(specific_inner_energy=1)
        self.assertAlmostEqual(self.eos.speed_of_sound(U)(mip), ngs.sqrt(2/7))

    def test_density_gradient(self):

        U = flowstate()
        dU = flowstate()
        self.assertIs(self.eos.density_gradient(U, dU), None)

        U = flowstate(T=1, p=1)
        dU = flowstate(grad_p=(1, 1), grad_T=(1, 0))
        nptest.assert_almost_equal(self.eos.density_gradient(U, dU)(mip), (0, 7/2))

        U = flowstate(T=1, rho_Ei=1)
        dU = flowstate(grad_T=(1, 1), grad_rho_Ei=(1, 0))
        nptest.assert_almost_equal(self.eos.density_gradient(U, dU)(mip), (0, -7/5))

    def test_pressure_gradient(self):

        U = flowstate()
        dU = flowstate()
        self.assertIs(self.eos.pressure_gradient(U, dU), None)

        U = flowstate(T=1, rho=1)
        dU = flowstate(grad_rho=(1, 1), grad_T=(1, 0))
        nptest.assert_almost_equal(self.eos.pressure_gradient(U, dU)(mip), (4/7, 2/7))

        dU = flowstate(grad_rho_Ei=(1, 0))
        nptest.assert_almost_equal(self.eos.pressure_gradient(U, dU)(mip), (0.4, 0))

    def test_temperature_gradient(self):

        U = flowstate()
        dU = flowstate()
        self.assertIs(self.eos.temperature_gradient(U, dU), None)

        U = flowstate(p=1, rho=1)
        dU = flowstate(grad_rho=(1, 0), grad_p=(1, 1))
        nptest.assert_almost_equal(self.eos.temperature_gradient(U, dU)(mip), (0, 7/2))

        dU = flowstate(grad_Ei=(1, 0))
        nptest.assert_almost_equal(self.eos.temperature_gradient(U, dU)(mip), (1.4, 0))

    def test_characteristic_velocities(self):

        U = flowstate()
        self.assertIs(self.eos.characteristic_velocities(U, (1, 0)), None)

        U = flowstate(velocity=(1, 0), speed_of_sound=1.4)
        nptest.assert_almost_equal(self.eos.characteristic_velocities(U, (1, 0))(mip), (-0.4, 1, 1, 2.4))
        nptest.assert_almost_equal(self.eos.characteristic_velocities(
            U, (1, 0), "absolute")(mip), (0.4, 1, 1, 2.4))
        nptest.assert_almost_equal(self.eos.characteristic_velocities(
            U, (1, 0), "incoming")(mip), (-0.4, 0, 0, 0))
        nptest.assert_almost_equal(self.eos.characteristic_velocities(
            U, (1, 0), "outgoing")(mip), (0, 1, 1, 2.4))

    def test_characteristic_variables(self):

        U = flowstate()
        dU = flowstate()
        self.assertIs(self.eos.characteristic_variables(U, dU, (1, 0)), None)

        U = flowstate(rho=2, c=1)
        dU = flowstate(grad_rho=(1, 0), grad_p=(0, 1),  grad_u=ngs.CF((1, 0, 0, 1), dims=(2, 2)))
        nptest.assert_almost_equal(self.eos.characteristic_variables(U, dU, (1, 0))(mip), (-2.0, 1.0, 0.0, 2.0))

    def test_characteristic_amplitudes(self):

        U = flowstate()
        dU = flowstate()
        self.assertIs(self.eos.characteristic_amplitudes(U, dU, (1, 0)), None)

        U = flowstate(rho=2, c=1, u=(2, 0))
        dU = flowstate(grad_rho=(1, 0), grad_p=(0, 1), grad_u=ngs.CF((1, 0, 0, 1), dims=(2, 2)))
        nptest.assert_almost_equal(self.eos.characteristic_amplitudes(U, dU, (1, 0))(mip), (-2.0, 2.0, 0.0, 6.0))

    def test_primitive_from_conservative(self):
        U = flowstate()
        self.assertIs(self.eos.primitive_from_conservative(U), None)

        U = flowstate(rho=2, velocity=(2, 0))
        Minv = self.eos.primitive_from_conservative(U)(mip)
        nptest.assert_almost_equal(Minv, (1, 0, 0, 0, -1, 0.5, 0, 0, 0, 0, 0.5, 0, 0.8, -0.8, 0, 0.4))

    def test_primitive_from_characteristic(self):
        U = flowstate()
        self.assertIs(self.eos.primitive_from_characteristic(U, (1, 0)), None)

        U = flowstate(rho=2, speed_of_sound=2)
        L = self.eos.primitive_from_characteristic(U, (1, 0))(mip)
        nptest.assert_almost_equal(L, (0.125, 0.25, 0, 0.125, -0.125, 0, 0, 0.125, 0, 0, -1, 0, 0.5, 0, 0, 0.5))

    def test_primitive_convective_jacobian(self):
        U = flowstate(rho=2, speed_of_sound=2, velocity=(1, 1))

        A = self.eos.primitive_convective_jacobian_x(U)(mip)
        nptest.assert_almost_equal(A, (1, 2, 0, 0, 0, 1, 0, 0.5, 0, 0, 1, 0, 0, 8, 0, 1))

        B = self.eos.primitive_convective_jacobian_y(U)(mip)
        nptest.assert_almost_equal(B, (1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0.5, 0, 0, 8, 1))

    def test_conservative_from_primitive(self):
        U = flowstate()
        self.assertIs(self.eos.conservative_from_primitive(U), None)

        U = flowstate(rho=2, velocity=(2, 0))
        M = self.eos.conservative_from_primitive(U)(mip)
        nptest.assert_almost_equal(M, (1, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 2, 4, 0, 2.5))

    def test_conservative_from_characteristic(self):
        U = flowstate()
        self.assertIs(self.eos.conservative_from_characteristic(U, (1, 0)), None)

        U = flowstate(rho=2, speed_of_sound=2, velocity=(2, 0))
        P = self.eos.conservative_from_characteristic(U, (1, 0))(mip)
        nptest.assert_almost_equal(P, (0.125, 0.25, 0,
                                       0.125, 0, 0.5, 0, 0.5,
                                       0, 0, -2, 0,
                                       1, 0.5, 0, 2))

    def test_conservative_convective_jacobian(self):
        U = flowstate(velocity=(2, 0), specific_energy=5)

        A = self.eos.conservative_convective_jacobian_x(U)(mip)
        nptest.assert_almost_equal(A, (0, 1, 0, 0, -3.2, 3.2, 0, 0.4, 0, 0, 2, 0, -10.8, 4.6, 0, 2.8))

        B = self.eos.conservative_convective_jacobian_y(U)(mip)
        nptest.assert_almost_equal(B, (0, 0, 1, 0, 0, 0, 2, 0, 0.8, -0.8, 0, 0.4, 0, 0, 6.2, 0))

    def test_characteristic_from_primitive(self):
        U = flowstate()
        self.assertIs(self.eos.characteristic_from_primitive(U, (1, 0)), None)

        U = flowstate(rho=2, speed_of_sound=2)
        Linv = self.eos.characteristic_from_primitive(U, (1, 0))(mip)
        nptest.assert_almost_equal(Linv, (0, -4, 0, 1, 4, 0, 0, -1, 0, 0, -1, 0, 0, 4, 0, 1))

    def test_characteristic_from_conservative(self):
        U = flowstate()
        self.assertIs(self.eos.characteristic_from_conservative(U, (1, 0)), None)

        U = flowstate(rho=2, speed_of_sound=2, velocity=(2, 0))
        Pinv = self.eos.characteristic_from_conservative(U, (1, 0))(mip)
        nptest.assert_almost_equal(Pinv, (4.8, -2.8, 0, 0.4,
                                          3.2, 0.8, 0, -0.4,
                                          0, 0, -0.5, 0,
                                          -3.2, 1.2, 0, 0.4))

    def test_identity(self):
        U = flowstate(rho=2, velocity=(2, 2))
        unit_vector = ngs.CF((1, 1))/ngs.sqrt(2)

        M = self.eos.conservative_from_primitive(U)
        Minv = self.eos.primitive_from_conservative(U)
        nptest.assert_almost_equal((M * Minv)(mip), ngs.Id(4)(mip))

        U = flowstate(rho=2, speed_of_sound=2)
        L = self.eos.primitive_from_characteristic(U, unit_vector)
        Linv = self.eos.characteristic_from_primitive(U, unit_vector)
        nptest.assert_almost_equal((L * Linv)(mip), ngs.Id(4)(mip))

        U = flowstate(rho=2, speed_of_sound=2, velocity=(2, 2))
        P = self.eos.conservative_from_characteristic(U, unit_vector)
        Pinv = self.eos.characteristic_from_conservative(U, unit_vector)
        nptest.assert_almost_equal((P * Pinv)(mip), ngs.Id(4)(mip))

    def test_transformation(self):
        U = flowstate(rho=2, speed_of_sound=2, velocity=(2, 2))
        unit_vector = ngs.CF((1, 1))/ngs.sqrt(2)

        M = self.eos.conservative_from_primitive(U)
        L = self.eos.primitive_from_characteristic(U, unit_vector)
        P = self.eos.conservative_from_characteristic(U, unit_vector)
        nptest.assert_almost_equal((M * L)(mip), P(mip))

        Minv = self.eos.primitive_from_conservative(U)
        Linv = self.eos.characteristic_from_primitive(U, unit_vector)
        Pinv = self.eos.characteristic_from_conservative(U, unit_vector)
        nptest.assert_almost_equal((Linv * Minv)(mip), Pinv(mip))
