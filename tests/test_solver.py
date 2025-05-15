from __future__ import annotations
import unittest

from tests import simplex, ngs, DummySolverConfiguration


class TestSolverConfiguration(unittest.TestCase):

    def setUp(self) -> None:
        self.cfg = DummySolverConfiguration(mesh=simplex())
        # self.cfg.time = "transient"

    def test_initialize_finite_element_spaces_dictionary(self):
        self.cfg.fem.initialize_finite_element_spaces()

        self.assertTupleEqual(tuple(self.cfg.fem.spaces), ('U', 'Uhat'))

        for space in self.cfg.fem.spaces.values():
            self.assertIsInstance(space, ngs.L2)

    def test_initialize_fininte_element_space(self):
        self.cfg.fem.initialize_finite_element_spaces()

        self.assertIsInstance(self.cfg.fem.fes, ngs.ProductSpace)

    def test_trial_and_test_functions(self):
        self.cfg.fem.initialize_finite_element_spaces()
        self.cfg.fem.initialize_trial_and_test_functions()

        self.assertTupleEqual(tuple(self.cfg.fem.TnT), ('U', 'Uhat'))

        for trial, test in self.cfg.fem.TnT.values():
            self.assertIsInstance(trial, ngs.comp.ProxyFunction)
            self.assertIsInstance(test, ngs.comp.ProxyFunction)

    def test_initialize_gridfunction_components(self):
        self.cfg.fem.initialize_finite_element_spaces()
        self.cfg.fem.initialize_trial_and_test_functions()
        self.cfg.fem.initialize_gridfunctions()

        self.assertTupleEqual(tuple(self.cfg.fem.gfus), ('U', 'Uhat'))
        for gfu in self.cfg.fem.gfus.values():
            self.assertIsInstance(gfu, ngs.GridFunction)

    def test_initialize_gridfunction(self):
        self.cfg.fem.initialize_finite_element_spaces()
        self.cfg.fem.initialize_trial_and_test_functions()
        self.cfg.fem.initialize_gridfunctions()
        self.assertIsInstance(self.cfg.fem.gfu, ngs.GridFunction)

    def test_initialze_transient_gridfunctions(self):
        self.cfg.fem.initialize_finite_element_spaces()
        self.cfg.fem.initialize_trial_and_test_functions()
        self.cfg.fem.initialize_gridfunctions()
        self.cfg.fem.initialize_time_scheme_gridfunctions()

        self.assertIn('U', self.cfg.fem.scheme.gfus)
        self.assertIn('Uhat', self.cfg.fem.scheme.gfus)
        self.assertIsInstance(self.cfg.fem.scheme.gfus['U']['n+1'], ngs.GridFunction)
        self.assertIsInstance(self.cfg.fem.scheme.gfus['U']['n+1'].space, ngs.L2)
        self.assertIsInstance(self.cfg.fem.scheme.gfus['Uhat']['n+1'], ngs.GridFunction)
        self.assertIsInstance(self.cfg.fem.scheme.gfus['Uhat']['n+1'].space, ngs.L2)

    def test_initialize_symbolic_forms(self):
        self.cfg.fem.initialize_finite_element_spaces()
        self.cfg.fem.initialize_trial_and_test_functions()
        self.cfg.fem.initialize_symbolic_forms()

        self.assertIn('test', self.cfg.fem.blf['U'])
        self.assertIn('test', self.cfg.fem.lf['U'])
