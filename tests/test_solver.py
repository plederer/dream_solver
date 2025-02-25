from __future__ import annotations
import unittest

from tests import simplex, ngs, DummySolverConfiguration


class TestSolverConfiguration(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = DummySolverConfiguration(mesh=simplex())
        self.cfg.time = "transient"

    def test_initialize_finite_element_spaces_dictionary(self):
        self.cfg.initialize_finite_element_spaces()

        self.assertTupleEqual(tuple(self.cfg.spaces), ('U', 'Uhat'))

        for space in self.cfg.spaces.values():
            self.assertIsInstance(space, ngs.L2)

    def test_initialize_fininte_element_space(self):
        self.cfg.initialize_finite_element_spaces()

        self.assertIsInstance(self.cfg.fes, ngs.ProductSpace)

    def test_trial_and_test_functions(self):
        self.cfg.initialize_finite_element_spaces()
        self.cfg.initialize_trial_and_test_functions()

        self.assertTupleEqual(tuple(self.cfg.TnT), ('U', 'Uhat'))

        for trial, test in self.cfg.TnT.values():
            self.assertIsInstance(trial, ngs.comp.ProxyFunction)
            self.assertIsInstance(test, ngs.comp.ProxyFunction)

    def test_initialize_gridfunction_components(self):
        self.cfg.initialize_finite_element_spaces()
        self.cfg.initialize_trial_and_test_functions()
        self.cfg.initialize_gridfunctions()

        self.assertTupleEqual(tuple(self.cfg.gfus), ('U', 'Uhat'))
        for gfu in self.cfg.gfus.values():
            self.assertIsInstance(gfu, ngs.GridFunction)

    def test_initialize_gridfunction(self):
        self.cfg.initialize_finite_element_spaces()
        self.cfg.initialize_trial_and_test_functions()
        self.cfg.initialize_gridfunctions()
        self.assertIsInstance(self.cfg.gfu, ngs.GridFunction)

    def test_initialze_transient_gridfunctions(self):
        self.cfg.initialize_finite_element_spaces()
        self.cfg.initialize_trial_and_test_functions()
        self.cfg.initialize_gridfunctions()

        self.cfg.time.initialize()
        self.assertIn('U', self.cfg.time.scheme.gfus)
        self.assertIn('Uhat', self.cfg.time.scheme.gfus)
        self.assertIsInstance(self.cfg.time.scheme.gfus['U']['n+1'], ngs.GridFunction)
        self.assertIsInstance(self.cfg.time.scheme.gfus['U']['n+1'].space, ngs.L2)
        self.assertIsInstance(self.cfg.time.scheme.gfus['Uhat']['n+1'], ngs.GridFunction)
        self.assertIsInstance(self.cfg.time.scheme.gfus['Uhat']['n+1'].space, ngs.L2)

    def test_initialize_symbolic_forms(self):
        self.cfg.initialize_finite_element_spaces()
        self.cfg.initialize_trial_and_test_functions()
        self.cfg.initialize_symbolic_forms()

        self.assertIn('test', self.cfg.blf)
        self.assertIn('test', self.cfg.lf)
