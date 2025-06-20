import pytest
from dream.time import Scheme


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

    integrals = {
        "U": {"time": [0], "convection": [1, 2]},
        "Uhat": {"convection": [3, 4, 5]}
    }

    blf = []
    scheme.add_sum_of_integrals(blf, integrals)

    assert blf == [0, 1, 2, 3, 4, 5]
