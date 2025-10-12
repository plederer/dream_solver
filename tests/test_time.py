import pytest
import numpy.testing as nptest

import ngsolve as ngs
from dream.time import Scheme

mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))

def test_parse_sum_of_integrals_basic():
    scheme = Scheme(None)
    integrals = {
        "U": {"time": "time_tree", "convection": "convection_tree"},
        "Uhat": {"convection": "convection_tree"}
    }
    result = scheme.parse_sum_of_integrals(integrals)
    assert result == integrals


def test_parse_sum_of_integrals_include_spaces():

    scheme = Scheme(None)
    integrals = {
        "U": {"time": "time_tree", "convection": "convection_tree"},
        "Uhat": {"convection": "convection_tree"},
        "Q": {"diffusion": "diffusion_tree"}
    }
    result = scheme.parse_sum_of_integrals(integrals, include_spaces=['U'])
    assert result == {"U": {"time": "time_tree", "convection": "convection_tree"}}

    result = scheme.parse_sum_of_integrals(integrals, include_spaces=['Uhat'])
    assert result == {"Uhat": {"convection": "convection_tree"}}

    with pytest.raises(KeyError):
        scheme.parse_sum_of_integrals(integrals, include_spaces=['NonExistentSpace'])


def test_parse_sum_of_integrals_exclude_spaces():

    scheme = Scheme(None)
    integrals = {
        "U": {"time": "time_tree", "convection": "convection_tree"},
        "Uhat": {"convection": "convection_tree"},
        "Q": {"diffusion": "diffusion_tree"}
    }
    result = scheme.parse_sum_of_integrals(integrals, exclude_spaces=['U'])
    assert result == {"Uhat": {"convection": "convection_tree"}, "Q": {"diffusion": "diffusion_tree"}}

    result = scheme.parse_sum_of_integrals(integrals, exclude_spaces=['Uhat'])
    assert result == {"U": {"time": "time_tree", "convection": "convection_tree"}, "Q": {"diffusion": "diffusion_tree"}}


def test_parse_sum_of_integrals_exclude_terms():
    scheme = Scheme(None)

    integrals = {
        "U": {"time": "time_tree", "convection": "convection_tree"},
        "Uhat": {"convection": "convection_tree"}
    }

    result = scheme.parse_sum_of_integrals(integrals, exclude_terms=("time",))
    assert result == {'U': {"convection": "convection_tree"}, 'Uhat': {"convection": "convection_tree"}}

    result = scheme.parse_sum_of_integrals(integrals, exclude_terms=("convection",))
    assert result == {"U": {"time": "time_tree"}}


def test_parse_sum_of_integrals_include_terms():
    scheme = Scheme(None)

    integrals = {
        "U": {"time": "time_tree", "convection": "convection_tree"},
        "Uhat": {"convection": "convection_tree"},
        "Q": {"diffusion": "diffusion_tree"}
    }

    result = scheme.parse_sum_of_integrals(integrals, include_spaces=('U',), include_terms=("time",))
    assert result == {'U': {"time": "time_tree"}}

    result = scheme.parse_sum_of_integrals(integrals, include_terms=("convection",))
    assert result == {'U': {"convection": "convection_tree"}, 'Uhat': {"convection": "convection_tree"}}

    # with pytest.raises(KeyError):
    #     result = scheme.parse_sum_of_integrals(integrals, include_terms=("time",))


def test_add_sum_of_integrals():
    """ This test checks that the sum of integrals are correctly added to the ngsolve form. 

    Since the ngsolve forms use the operator +=, we can simply check that the
    values are correctly added to a mutable object like a list.

    """

    scheme = Scheme(None)
    fes = ngs.L2(mesh, order=0)
    u, v = fes.TnT()

    integrals = {
        "U": {"time": u * v * ngs.dx, "convection": 2 * u * v * ngs.dx},
        "Uhat": {"convection": 4* u * v * ngs.dx}
    }

    blf = ngs.BilinearForm(fes)
    scheme.add_sum_of_integrals(blf, integrals)
    blf.Assemble()

    nptest.assert_almost_equal(tuple(blf.mat.AsVector()), (3.5, 3.5))

    integrals = {
        "U": {"time": 1 * v * ngs.dx, "convection": 2 * v * ngs.dx},
        "Uhat": {"convection": 4 *  v * ngs.dx}
    }

    lf = ngs.LinearForm(fes)
    scheme.add_sum_of_integrals(lf, integrals)
    lf.Assemble()

    nptest.assert_almost_equal(tuple(lf.vec), (3.5, 3.5))
