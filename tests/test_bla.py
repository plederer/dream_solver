from __future__ import annotations
import unittest
import numpy.testing as nptest
from tests import unit_square

import ngsolve as ngs
import dream.bla as bla


class BasicLinearAlgebra(unittest.TestCase):

    def setUp(self):
        self.mesh = unit_square()
        self.mip = self.mesh()

    def test_absolute_value_scalar(self):
        a = ngs.CF(-2)
        result = bla.abs(a)(self.mip)
        expected = 2
        self.assertAlmostEqual(result, expected)

    def test_absolute_value_function(self):
        result = ngs.Integrate(bla.abs(ngs.x), self.mesh)
        expected = 0.25
        self.assertAlmostEqual(result, expected)

    def test_maximum_value_default(self):
        a = ngs.CF(-2)

        result = bla.max(a)(self.mip)
        expected = 0
        self.assertAlmostEqual(result, expected)

    def test_maximum_value(self):
        a = ngs.CF(-2)
        b = ngs.CF(-4)

        result = bla.max(a, b)(self.mip)
        expected = -2
        self.assertAlmostEqual(result, expected)

    def test_minimum_value_default(self):
        a = ngs.CF(2)

        result = bla.min(a)(self.mip)
        expected = 0
        self.assertAlmostEqual(result, expected)

    def test_minimum_value(self):
        a = ngs.CF(2)
        b = ngs.CF(-2)

        result = bla.min(a, b)(self.mip)
        expected = -2
        self.assertAlmostEqual(result, expected)

    def test_interval(self):
        a = ngs.CF(2)

        result = bla.interval(a, 0, 4)(self.mip)
        expected = 2
        self.assertAlmostEqual(result, expected)

        result = bla.interval(a, 3, 4)(self.mip)
        expected = 3
        self.assertAlmostEqual(result, expected)

        result = bla.interval(a, 0, 1)(self.mip)
        expected = 1
        self.assertAlmostEqual(result, expected)

        result = ngs.Integrate(bla.interval(ngs.x, 0.0, 0.25), self.mesh)
        expected = 0.0625 + 0.03125
        self.assertAlmostEqual(result, expected)

    def test_trace(self):
        a = ngs.CF(2)
        b = ngs.CF((2, 3, 4, 5), dims=(2, 2))

        with self.assertRaises(ValueError):
            bla.trace(a)

        result = bla.trace(b)(self.mip)
        expected = 7

        self.assertAlmostEqual(result, expected)

    def test_diagonal(self):
        a = [i for i in range(1, 4)]

        result = bla.diagonal(a)(self.mip)
        expected = (1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0)
        nptest.assert_almost_equal(result, expected)

    def test_unit_vector(self):
        a = ngs.CF((10, 10))

        result = bla.unit_vector(a)(self.mip)
        expected = (1/ngs.sqrt(2), 1/ngs.sqrt(2))
        nptest.assert_almost_equal(result, expected)

    def test_fixpoint_iteration(self):

        result = bla.fixpoint_iteration(0.25, lambda x: ngs.sqrt(2-x))
        expected = 1
        self.assertAlmostEqual(result, expected)

    def test_symmetric_matrix_from_vector(self):

        a = [i for i in range(1,4)]
        result = bla.symmetric_matrix_from_vector(a)(self.mip)
        expected = (1.0, 2.0, 2.0, 3.0)
        nptest.assert_almost_equal(result, expected)
    
        a = [i for i in range(1,7)]
        result = bla.symmetric_matrix_from_vector(a)(self.mip)
        expected = (1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0)
        nptest.assert_almost_equal(result, expected)


        with self.assertRaises(ValueError):
            a = [i for i in range(1,8)]
            result = bla.symmetric_matrix_from_vector(a)(self.mip)

    def test_skewsymmetric_matrix_from_vector(self):

        a = 1
        result = bla.skewsymmetric_matrix_from_vector(a)(self.mip)
        expected = (0.0, -1.0, 1.0, 0.0)
        nptest.assert_almost_equal(result, expected)
    
        a = [i for i in range(1,4)]
        result = bla.skewsymmetric_matrix_from_vector(a)(self.mip)
        expected = (0, -1.0, 2.0, 1.0, 0.0, -3.0, -2.0, 3.0, 0.0)
        nptest.assert_almost_equal(result, expected)


        with self.assertRaises(ValueError):
            a = [i for i in range(1,8)]
            result = bla.skewsymmetric_matrix_from_vector(a)(self.mip)