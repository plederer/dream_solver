from __future__ import annotations
import unittest
import numpy.testing as nptest
from tests import simplex
import ngsolve as ngs

from dream.compressible.config import flowfields
from tests.compressible.setup import cfg, mip


class TestConservativeHDGFiniteElementMethod(unittest.TestCase):

    def setUp(self) -> None:

        cfg.fem = "conservative"
        cfg.fem.method = "hdg"
        cfg.fem.mixed_method = "inactive"
        cfg.dynamic_viscosity = "inviscid"
        cfg.time = "stationary"

    def test_inviscid_finite_element_spaces(self):

        spaces = {}
        cfg.dynamic_viscosity = "inviscid"
        cfg.fem.mixed_method = 'inactive'
        cfg.fem.add_finite_element_spaces(spaces)

        for space, expected in zip(spaces, ('U', 'Uhat'), strict=True):
            self.assertEqual(space, expected)
            self.assertIsInstance(spaces[space], ngs.ProductSpace)

    def test_strain_heat_finite_element_spaces(self):

        spaces = {}
        cfg.dynamic_viscosity = "constant"
        cfg.fem.mixed_method = "strain_heat"
        cfg.fem.add_finite_element_spaces(spaces)

        for space, expected in zip(spaces, ('U', 'Uhat', 'Q'), strict=True):
            self.assertEqual(space, expected)
            self.assertIsInstance(spaces[space], ngs.ProductSpace)

    def test_gradient_finite_element_spaces(self):

        spaces = {}
        cfg.dynamic_viscosity = "constant"
        cfg.fem.mixed_method = "gradient"
        cfg.fem.add_finite_element_spaces(spaces)

        for space, expected in zip(spaces, ('U', 'Uhat', 'Q'), strict=True):
            self.assertEqual(space, expected)
            self.assertIsInstance(spaces[space], ngs.ProductSpace)

    def test_test_and_trial_functions(self):

        cfg.initialize_finite_element_spaces()
        cfg.initialize_trial_and_test_functions()

        self.assertTupleEqual(tuple(cfg.TnT), ('U', 'Uhat'))
        for trial, test in cfg.TnT.values():
            self.assertIsInstance(trial, ngs.comp.ProxyFunction)
            self.assertIsInstance(test, ngs.comp.ProxyFunction)
