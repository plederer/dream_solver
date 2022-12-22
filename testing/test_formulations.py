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

        cls.testing_U = CF((x*y,
                            x**3*y**2,
                            y**3*x**2,
                            x**2*y**2))
        cls.testing_gradient_U = CF((y, x,
                                     3*x**2*y**2, 2*x**3*y,
                                     2*y**3*x, 3*y**2*x**2,
                                     2*x*y**2, 2*y*x**2), dims=(4, 2))
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

        expected_value = -(gamma-1)*160/12

        return is_value, expected_value

    def velocity_gradient_test(self): ...

    def temperature_gradient_test(self): ...
    # def test_density(self):

    #     mesh = self.formulation.mesh

    #     test_U = CF((x**3 + y*x, x**3, y*x, y**2))
    #     exact_density_integral = 0.5

    #     test_gfu = GridFunction(self.formulation.fes)
    #     U = test_gfu.components[0]
    #     U.Set(test_U)

    #     expected_density = self.formulation.density(U.components)

    #     error_msg = f"Wrong density returned by {self.formulation}"
    #     self.assertAlmostEqual(Integrate(expected_density, mesh),
    #                            exact_density_integral, msg=error_msg)

    # def test_density_gradient(self):

    #     solver_configuration = self.formulation.solver_configuration
    #     mesh = self.formulation.mesh

    #     test_U = CF((x**3 + y*x, x**3, y*x, y**2))
    #     test_Q_gradient = CF((3*x**2 + y, x, 3*x**2, 0, y, x, 0, 2*y), dims=(4, 2))
    #     some_vector = CF((2, 1))
    #     exact_gradient_vector_integral = 3.5

    #     # Test gradient of None and Strain-Heat mixed method
    #     solver_configuration.mixed_method = None
    #     test_gfu = GridFunction(self.formulation.get_FESpace())

    #     U = test_gfu.components[0]
    #     U.Set(test_U)
    #     Q = None

    #     expected_density_gradient = self.formulation.density_gradient(U, Q)

    #     error_msg = f"Wrong density gradient for {self.formulation} - {solver_configuration.mixed_method}"
    #     self.assertAlmostEqual(Integrate(expected_density_gradient * some_vector, mesh),
    #                            exact_gradient_vector_integral, msg=error_msg)

    #     # Test gradient of Gradient mixed method
    #     solver_configuration.mixed_method = "gradient"
    #     test_gfu = GridFunction(self.formulation.get_FESpace())

    #     U = test_gfu.components[0]
    #     U.Set(test_U)
    #     Q = test_gfu.components[2]
    #     Q.Set(test_Q_gradient)

    #     expected_density_gradient = self.formulation.density_gradient(U, Q.components)

    #     error_msg = f"Wrong density gradient for {self.formulation} - {solver_configuration.mixed_method}"
    #     self.assertAlmostEqual(Integrate(expected_density_gradient * some_vector, mesh),
    #                            exact_gradient_vector_integral, msg=error_msg)

    # def test_momentum(self):

    #     test_U = CF((x**3 + y*x, x**3, y*x, y**2))
    #     some_vector = CF((2, 1))
    #     exact_momentum_vector_integral = 0.75

    #     test_gfu = GridFunction(self.formulation.fes)
    #     U = test_gfu.components[0]
    #     U.Set(test_U)

    #     expected_momentum = self.formulation.momentum(U.components)

    #     error_msg = f"Wrong momentum returned by {self.formulation}"
    #     self.assertAlmostEqual(Integrate(expected_momentum * some_vector, self.formulation.mesh),
    #                            exact_momentum_vector_integral, msg=error_msg)

    # def test_momentum_gradient(self):

    #     solver_configuration = self.formulation.solver_configuration

    #     test_U = CF((x**3 + y*x, x**3, y*x, y**2))
    #     test_Q_gradient = CF((3*x**2 + y, x, 3*x**2, 0, y, x, 0, 2*y), dims=(4, 2))
    #     some_matrix = CF((0.5, 1, 1.5, 2), dims=(2, 2))
    #     exact_gradient_matrix_integral = 2.25

    #     # Test gradient of None and Strain-Heat mixed method
    #     solver_configuration.mixed_method = None
    #     test_gfu = GridFunction(self.formulation.get_FESpace())

    #     U = test_gfu.components[0]
    #     U.Set(test_U)
    #     Q = None

    #     momentum_gradient = self.formulation.momentum_gradient(U, Q)

    #     error_msg = f"Wrong momentum gradient for {self.formulation} - {solver_configuration.mixed_method}"
    #     self.assertAlmostEqual(Integrate(InnerProduct(momentum_gradient, some_matrix), self.formulation.mesh),
    #                            exact_gradient_matrix_integral, msg=error_msg)

    #     # Test gradient of Gradient mixed method
    #     solver_configuration.mixed_method = "gradient"
    #     test_gfu = GridFunction(self.formulation.get_FESpace())

    #     U = test_gfu.components[0]
    #     U.Set(test_U)
    #     Q = test_gfu.components[2]
    #     Q.Set(test_Q_gradient)

    #     momentum_gradient = self.formulation.momentum_gradient(U, Q.components)

    #     error_msg = f"Wrong momentum gradient for {self.formulation} - {solver_configuration.mixed_method}"
    #     self.assertAlmostEqual(Integrate(InnerProduct(momentum_gradient, some_matrix), self.formulation.mesh),
    #                            exact_gradient_matrix_integral, msg=error_msg)

    # def test_energy(self):

    #     test_U = CF((x**3 + y*x, x**3, y*x, y**2))
    #     exact_energy_integral = 1/3

    #     test_gfu = GridFunction(self.formulation.fes)
    #     U = test_gfu.components[0]
    #     U.Set(test_U)

    #     expected_energy = self.formulation.energy(U.components)

    #     error_msg = f"Wrong energy returned by {self.formulation}"
    #     self.assertAlmostEqual(Integrate(expected_energy, self.formulation.mesh),
    #                            exact_energy_integral, msg=error_msg)

    # def test_energy_gradient(self):

    # solver_configuration = self.formulation.solver_configuration

    # test_U = CF((x**3 + y*x, x**3, y*x, y**2))
    # test_Q_gradient = CF((3*x**2 + y, x, 3*x**2, 0, y, x, 0, 2*y), dims=(4, 2))
    # some_vector = CF((2, 1))
    # exact_gradient_vector_integral = 1

    # # Test gradient of None and Strain-Heat mixed method
    # solver_configuration.mixed_method = None
    # test_gfu = GridFunction(self.formulation.get_FESpace())

    # U = test_gfu.components[0]
    # U.Set(test_U)
    # Q = None

    # expected_density_gradient = self.formulation.energy_gradient(U, Q)

    # error_msg = f"Wrong density gradient for {self.formulation} - {solver_configuration.mixed_method}"
    # self.assertAlmostEqual(Integrate(expected_density_gradient * some_vector, self.formulation.mesh),
    #                        exact_gradient_vector_integral, msg=error_msg)

    # # Test gradient of Gradient mixed method
    # solver_configuration.mixed_method = "gradient"
    # test_gfu = GridFunction(self.formulation.get_FESpace())

    # U = test_gfu.components[0]
    # U.Set(test_U)
    # Q = test_gfu.components[2]
    # Q.Set(test_Q_gradient)

    # expected_density_gradient = self.formulation.energy_gradient(U, Q.components)

    # error_msg = f"Wrong density gradient for {self.formulation} - {solver_configuration.mixed_method}"
    # self.assertAlmostEqual(Integrate(expected_density_gradient * some_vector, self.formulation.mesh),
    #                        exact_gradient_vector_integral, msg=error_msg)


if __name__ == "__main__":

    runner = unittest.TextTestRunner(descriptions=True, verbosity=2)
    runner.run(formulation_tests())
