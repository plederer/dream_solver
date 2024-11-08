from __future__ import annotations
import unittest

from tests import simplex, ngs
from dream.pde import PDEConfiguration, FiniteElementMethod
from dream.config import interface


class DummyFiniteElementMethod(FiniteElementMethod):

    name: str = 'dummy'
    cfg: DummyPDE

    def add_finite_element_spaces(self, spaces: dict[str, ngs.FESpace]):
        spaces['U'] = ngs.L2(self.mesh, order=0)
        spaces['Uhat'] = ngs.L2(self.mesh, order=0)

    def add_transient_gridfunctions(self, gfus: dict[str, dict[str, ngs.GridFunction]]):
        gfus['n+1'] = ngs.GridFunction(self.cfg.spaces['U'])

    def add_symbolic_forms(self, blf: dict[str, ngs.comp.SumOfIntegrals],
                           lf: dict[str, ngs.comp.SumOfIntegrals]):
        u, v = self.cfg.TnT['U']

        blf['test'] = u * v * ngs.dx
        lf['test'] = v * ngs.dx


class DummyPDE(PDEConfiguration):

    name: str = 'dummy'

    @interface(default=DummyFiniteElementMethod)
    def fem(self, fem):
        return fem


class TestPDEConfiguration(unittest.TestCase):

    def setUp(self) -> None:
        self.pde = DummyPDE(mesh=simplex())

    def test_initialize_finite_element_spaces_dictionary(self):
        self.pde.initialize_finite_element_spaces()

        self.assertTupleEqual(tuple(self.pde.spaces), ('U', 'Uhat'))

        for space in self.pde.spaces.values():
            self.assertIsInstance(space, ngs.L2)

    def test_initialize_fininte_element_space(self):
        self.pde.initialize_finite_element_spaces()

        self.assertIsInstance(self.pde.fes, ngs.ProductSpace)

    def test_trial_and_test_functions(self):
        self.pde.initialize_finite_element_spaces()
        self.pde.initialize_trial_and_test_functions()

        self.assertTupleEqual(tuple(self.pde.TnT), ('U', 'Uhat'))

        for trial, test in self.pde.TnT.values():
            self.assertIsInstance(trial, ngs.comp.ProxyFunction)
            self.assertIsInstance(test, ngs.comp.ProxyFunction)

    def test_initialize_gridfunction_components(self):
        self.pde.initialize_finite_element_spaces()
        self.pde.initialize_trial_and_test_functions()
        self.pde.initialize_gridfunctions()

        self.assertTupleEqual(tuple(self.pde.gfus), ('U', 'Uhat'))
        for gfu in self.pde.gfus.values():
            self.assertIsInstance(gfu, ngs.GridFunction)

    def test_initialize_gridfunction(self):
        self.pde.initialize_finite_element_spaces()
        self.pde.initialize_trial_and_test_functions()
        self.pde.initialize_gridfunctions()
        self.assertIsInstance(self.pde.gfu, ngs.GridFunction)

    def test_initialze_transient_gridfunctions(self):
        self.pde.initialize_finite_element_spaces()
        self.pde.initialize_transient_gridfunctions()

        self.assertIn('n+1', self.pde.transient_gfus)
        self.assertIsInstance(self.pde.transient_gfus['n+1'], ngs.GridFunction)
        self.assertIsInstance(self.pde.transient_gfus['n+1'].space, ngs.L2)

    def test_initialize_symbolic_forms(self):
        self.pde.initialize_finite_element_spaces()
        self.pde.initialize_trial_and_test_functions()
        self.pde.initialize_symbolic_forms()

        self.assertIn('test', self.pde.blf)
        self.assertIn('test', self.pde.lf)
