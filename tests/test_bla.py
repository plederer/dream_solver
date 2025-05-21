import pytest
import numpy.testing as nptest
import ngsolve as ngs
import dream.bla as bla

mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=0.5))


def test_absolute_value_scalar():
    a = ngs.CF(-2)
    assert bla.abs(a)(mesh()) == pytest.approx(2)


def test_absolute_value_function():
    assert ngs.Integrate(bla.abs(ngs.x), mesh) == pytest.approx(0.5)


def test_maximum_value_default():
    a = ngs.CF(-2)
    assert bla.max(a)(mesh()) == pytest.approx(0)


def test_maximum_value():
    a = ngs.CF(-2)
    b = ngs.CF(-4)
    assert bla.max(a, b)(mesh()) == pytest.approx(-2)


def test_minimum_value_default():
    a = ngs.CF(2)
    assert bla.min(a)(mesh()) == pytest.approx(0)


def test_minimum_value():
    a = ngs.CF(2)
    b = ngs.CF(-2)
    assert bla.min(a, b)(mesh()) == pytest.approx(-2)


def test_interval():
    a = ngs.CF(2)
    assert bla.interval(a, 0, 4)(mesh()) == pytest.approx(2)
    assert bla.interval(a, 3, 4)(mesh()) == pytest.approx(3)
    assert bla.interval(a, 0, 1)(mesh()) == pytest.approx(1)
    assert ngs.Integrate(bla.interval(ngs.x, 0.5, 1), mesh, order=10) == pytest.approx(0.625)


def test_trace():
    a = ngs.CF(2)
    b = ngs.CF((2, 3, 4, 5), dims=(2, 2))

    with pytest.raises(ValueError):
        bla.trace(a)
    assert bla.trace(b)(mesh()) == pytest.approx(7)


def test_diagonal():
    a = [i for i in range(1, 4)]
    nptest.assert_almost_equal(bla.diagonal(a)(mesh()), (1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0))


def test_unit_vector():
    a = ngs.CF((10, 10))
    nptest.assert_almost_equal(bla.unit_vector(a)(mesh()), (1/ngs.sqrt(2), 1/ngs.sqrt(2)))


def test_fixpoint_iteration():
    assert bla.fixpoint_iteration(0.25, lambda x: ngs.sqrt(2-x)) == pytest.approx(1)


def test_symmetric_matrix_from_vector():
    a = [i for i in range(1, 4)]
    nptest.assert_almost_equal(bla.symmetric_matrix_from_vector(a)(mesh()), (1.0, 2.0, 2.0, 3.0))

    a = [i for i in range(1, 7)]
    nptest.assert_almost_equal(bla.symmetric_matrix_from_vector(a)(mesh()),
                               (1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0))

    with pytest.raises(ValueError):
        a = [i for i in range(1, 8)]
        bla.symmetric_matrix_from_vector(a)(mesh())


def test_skewsymmetric_matrix_from_vector():
    a = 1
    nptest.assert_almost_equal(bla.skewsymmetric_matrix_from_vector(a)(mesh()), (0.0, -1.0, 1.0, 0.0))

    a = [i for i in range(1, 4)]
    nptest.assert_almost_equal(bla.skewsymmetric_matrix_from_vector(
        a)(mesh()), (0, -1.0, 2.0, 1.0, 0.0, -3.0, -2.0, 3.0, 0.0))

    with pytest.raises(ValueError):
        a = [i for i in range(1, 8)]
        bla.skewsymmetric_matrix_from_vector(a)(mesh())
