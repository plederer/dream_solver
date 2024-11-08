from __future__ import annotations
import unittest
import numpy.testing as nptest
from tests import simplex
import ngsolve as ngs

from dream.compressible.config import flowstate
from tests.compressible.setup import cfg, mip


class TestConservativeHDGFiniteElementMethod(unittest.TestCase):

    def setUp(self) -> None:

        cfg.pde.fem = "conservative"
        cfg.pde.fem.method = "hdg"
        cfg.pde.fem.mixed_method = "inactive"
        cfg.pde.dynamic_viscosity = "inviscid"
        cfg.time = "stationary"

    def test_inviscid_finite_element_spaces(self):

        spaces = {}
        cfg.pde.dynamic_viscosity = "inviscid"
        cfg.pde.fem.mixed_method = 'inactive'
        cfg.pde.fem.add_finite_element_spaces(spaces)

        for space, expected in zip(spaces, ('U', 'Uhat'), strict=True):
            self.assertEqual(space, expected)
            self.assertIsInstance(spaces[space], ngs.ProductSpace)

    def test_strain_heat_finite_element_spaces(self):

        spaces = {}
        cfg.pde.dynamic_viscosity = "constant"
        cfg.pde.fem.mixed_method = "strain_heat"
        cfg.pde.fem.add_finite_element_spaces(spaces)

        for space, expected in zip(spaces, ('U', 'Uhat', 'Q'), strict=True):
            self.assertEqual(space, expected)
            self.assertIsInstance(spaces[space], ngs.ProductSpace)

    def test_gradient_finite_element_spaces(self):

        spaces = {}
        cfg.pde.dynamic_viscosity = "constant"
        cfg.pde.fem.mixed_method = "gradient"
        cfg.pde.fem.add_finite_element_spaces(spaces)

        for space, expected in zip(spaces, ('U', 'Uhat', 'Q'), strict=True):
            self.assertEqual(space, expected)
            self.assertIsInstance(spaces[space], ngs.ProductSpace)

    def test_test_and_trial_functions(self):

        cfg.pde.initialize_finite_element_spaces()
        cfg.pde.initialize_trial_and_test_functions()

        self.assertTupleEqual(tuple(cfg.pde.TnT), ('U', 'Uhat'))
        for trial, test in cfg.pde.TnT.values():
            self.assertIsInstance(trial, ngs.comp.ProxyFunction)
            self.assertIsInstance(test, ngs.comp.ProxyFunction)
