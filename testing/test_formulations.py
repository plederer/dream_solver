import unittest
from ngsolve import *
from netgen.occ import OCCGeometry, Rectangle
from formulations import formulation_factory, ConservativeFormulation2D, Formulation
from configuration import SolverConfiguration, MixedMethods
from abc import ABC, abstractmethod


def formulation_tests(suite: unittest.TestSuite = None) -> unittest.TestSuite:

    if suite is None:
        suite = unittest.TestSuite()

    loader = unittest.TestLoader()

    suite._tests.extend(loader.loadTestsFromTestCase(Test2DConservativeFormulation)._tests)

    return suite


class TestFormulation(unittest.TestCase, ABC):

    formulation: Formulation
    testing_U: CF
    testing_gradient_U: CF

    def test_factory_function(self):
        formulation = formulation_factory(self.formulation.mesh, self.formulation.solver_configuration)
        self.assertIsInstance(formulation, type(self.formulation))

    def test_variables(self):

        test_gfu = GridFunction(self.formulation.get_FESpace())
        U = test_gfu.components[0]
        U.Set(self.testing_U)

        # Density
        error_msg = f"Wrong density returned by {self.formulation}"
        is_value, expected_value = self.density_test(U)
        self.assertAlmostEqual(is_value, expected_value, msg=error_msg)

        # Momentum
        error_msg = f"Wrong momentum returned by {self.formulation}"
        is_value, expected_value = self.momentum_test(U)
        self.assertAlmostEqual(is_value, expected_value, msg=error_msg)

        # Energy
        error_msg = f"Wrong energy returned by {self.formulation}"
        is_value, expected_value = self.energy_test(U)
        self.assertAlmostEqual(is_value, expected_value, msg=error_msg)

        # Pressure
        error_msg = f"Wrong pressure returned by {self.formulation}"
        is_value, expected_value = self.pressure_test(U)
        self.assertAlmostEqual(is_value, expected_value, msg=error_msg)

        # Velocity
        error_msg = f"Wrong velocity returned by {self.formulation}"
        is_value, expected_value = self.velocity_test(U)
        self.assertAlmostEqual(is_value, expected_value, msg=error_msg)

        # Temperature
        error_msg = f"Wrong temperature returned by {self.formulation}"
        is_value, expected_value = self.temperature_test(U)
        self.assertAlmostEqual(is_value, expected_value, msg=error_msg)

    def test_variables_gradients(self):

        def mixed_variables_gradients():

            mixed_method = self.formulation.solver_configuration.mixed_method

            test_gfu = GridFunction(self.formulation.get_FESpace())
            U = test_gfu.components[0]
            U.Set(self.testing_U)

            if mixed_method is MixedMethods.NONE:
                Q = None
            elif mixed_method is MixedMethods.GRADIENT:
                Q = test_gfu.components[2]
                Q.Set(self.testing_gradient_U)
            elif mixed_method is MixedMethods.STRAIN_HEAT:
                Q = test_gfu.components[2]
                Q.Set(self.testing_mixed_heat_Q)
            else:
                raise NotImplementedError()

            # Density Gradient
            error_msg = f"Wrong density gradient returned by {self.formulation} - {mixed_method}"
            is_value, expected_value = self.density_gradient_test(U, Q)
            self.assertAlmostEqual(is_value, expected_value, msg=error_msg)

            # Momentum Gradient
            error_msg = f"Wrong momentum gradient returned by {self.formulation} - {mixed_method}"
            is_value, expected_value = self.momentum_gradient_test(U, Q)
            self.assertAlmostEqual(is_value, expected_value, msg=error_msg)

            # Energy Gradient
            error_msg = f"Wrong energy gradient returned by {self.formulation} - {mixed_method}"
            is_value, expected_value = self.energy_gradient_test(U, Q)
            self.assertAlmostEqual(is_value, expected_value, msg=error_msg)

            # Pressure Gradient
            error_msg = f"Wrong pressure gradient returned by {self.formulation} - {mixed_method}"
            is_value, expected_value = self.pressure_gradient_test(U, Q)
            self.assertAlmostEqual(is_value, expected_value, msg=error_msg)

            # Velocity Gradient
            error_msg = f"Wrong velocity gradient returned by {self.formulation} - {mixed_method}"
            is_value, expected_value = self.velocity_gradient_test(U, Q)
            self.assertAlmostEqual(is_value, expected_value, msg=error_msg)

            # Temperature Gradient
            error_msg = f"Wrong temperature gradient returned by {self.formulation} - {mixed_method}"
            is_value, expected_value = self.temperature_gradient_test(U, Q)
            self.assertAlmostEqual(is_value, expected_value, msg=error_msg)

        # Loop over all MixedMethods
        for mixed_method in MixedMethods:
            self.formulation.solver_configuration.mixed_method = mixed_method.value
            mixed_variables_gradients()

    @abstractmethod
    def density_test(self, U): ...

    @abstractmethod
    def momentum_test(self, U): ...

    @abstractmethod
    def energy_test(self, U): ...

    @abstractmethod
    def pressure_test(self, U): ...

    @abstractmethod
    def velocity_test(self, U): ...

    @abstractmethod
    def temperature_test(self, U): ...

    @abstractmethod
    def density_gradient_test(self, U, Q): ...

    @abstractmethod
    def momentum_gradient_test(self, U, Q): ...

    @abstractmethod
    def energy_gradient_test(self, U, Q): ...

    @abstractmethod
    def pressure_gradient_test(self, U, Q): ...

    @abstractmethod
    def velocity_gradient_test(self, U, Q): ...

    @abstractmethod
    def temperature_gradient_test(self, U, Q): ...


class Test2DConservativeFormulation(TestFormulation):

    @classmethod
    def setUpClass(cls):
        config = SolverConfiguration(formulation="conservative", order=10)
        mesh = Mesh(OCCGeometry(Rectangle(2, 1).Face(), dim=2).GenerateMesh(maxh=0.5))
        cls.formulation = ConservativeFormulation2D(mesh, config)

        gamma = config.heat_capacity_ratio.Get()

        cls.testing_U = CF((x*y,
                            x**3*y**2,
                            y**3*x**2,
                            x**2*y**2))
        cls.testing_gradient_U = CF((y, x,
                                     3*x**2*y**2, 2*x**3*y,
                                     2*y**3*x, 3*y**2*x**2,
                                     2*x*y**2, 2*y*x**2), dims=(4, 2))
        cls.testing_mixed_heat_Q = CF((0,
                                       0,
                                       0,
                                       gamma*(y - 2*x**3*y**2 - x*y**4),
                                       gamma*(x - x**4*y - 2*x**2*y**3)))
        cls.testing_vector = CF((2, 1))
        cls.testing_matrix = CF((0.5, 1, 1.5, 2), dims=(2, 2))

    def density_test(self, U):

        density = self.formulation.density(U.components)
        is_value = Integrate(density, self.formulation.mesh)

        expected_value = 1

        return is_value, expected_value

    def momentum_test(self, U):

        momentum = self.formulation.momentum(U.components)
        is_value = Integrate(momentum * self.testing_vector, self.formulation.mesh)

        expected_value = 40/12

        return is_value, expected_value

    def energy_test(self, U):

        energy = self.formulation.energy(U.components)
        is_value = Integrate(energy, self.formulation.mesh)

        expected_value = 8/9

        return is_value, expected_value

    def pressure_test(self, U):

        gamma = self.formulation.solver_configuration.heat_capacity_ratio.Get()

        pressure = self.formulation.pressure(U.components)
        is_value = Integrate(pressure, self.formulation.mesh)

        expected_value = -(gamma - 1)*7/9

        return is_value, expected_value

    def velocity_test(self, U):

        velocity = self.formulation.velocity(U.components)
        is_value = Integrate(velocity * self.testing_vector, self.formulation.mesh)

        expected_value = 10/3

        return is_value, expected_value

    def temperature_test(self, U):

        gamma = self.formulation.solver_configuration.heat_capacity_ratio.Get()

        temperature = self.formulation.temperature(U.components)
        is_value = Integrate(temperature, self.formulation.mesh)

        expected_value = -gamma/3

        return is_value, expected_value

    def density_gradient_test(self, U, Q):

        if Q is not None:
            Q = Q.components

        density_gradient = self.formulation.density_gradient(U, Q)
        is_value = Integrate(density_gradient * self.testing_vector, self.formulation.mesh)

        expected_value = 4

        return is_value, expected_value

    def momentum_gradient_test(self, U, Q):

        if Q is not None:
            Q = Q.components

        momentum_gradient = self.formulation.momentum_gradient(U, Q)
        is_value = Integrate(InnerProduct(momentum_gradient, self.testing_matrix), self.formulation.mesh)

        expected_value = 73/6

        return is_value, expected_value

    def energy_gradient_test(self, U, Q):

        if Q is not None:
            Q = Q.components

        energy_gradient = self.formulation.energy_gradient(U, Q)
        is_value = Integrate(energy_gradient * self.testing_vector, self.formulation.mesh)

        expected_value = 16/3

        return is_value, expected_value

    def pressure_gradient_test(self, U, Q):

        gamma = self.formulation.solver_configuration.heat_capacity_ratio.Get()

        if Q is not None:
            Q = Q.components

        pressure_gradient = self.formulation.pressure_gradient(U, Q)
        is_value = Integrate(pressure_gradient * self.testing_vector, self.formulation.mesh)

        expected_value = -(gamma-1)*136/12

        return is_value, expected_value

    def velocity_gradient_test(self, U, Q):

        if Q is not None:
            Q = Q.components

        velocity_gradient = self.formulation.velocity_gradient(U, Q)
        is_value = Integrate(InnerProduct(velocity_gradient, self.testing_matrix), self.formulation.mesh)

        expected_value = 26/3

        return is_value, expected_value

    def temperature_gradient_test(self, U, Q):

        gamma = self.formulation.solver_configuration.heat_capacity_ratio.Get()

        if Q is not None:
            Q = Q.components

        temperature_gradient = self.formulation.temperature_gradient(U, Q)
        is_value = Integrate(temperature_gradient * self.testing_vector, self.formulation.mesh)

        expected_value = -gamma*20/3

        return is_value, expected_value


if __name__ == "__main__":

    runner = unittest.TextTestRunner(descriptions=True, verbosity=2)
    runner.run(formulation_tests())
