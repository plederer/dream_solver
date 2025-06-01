import pytest
import ngsolve as ngs

from dream.compressible import CompressibleFlowSolver

mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))
mip = mesh(0.5, 0.5)


@pytest.fixture()
def cfg():
    cfg = CompressibleFlowSolver(mesh)
    cfg.time = "transient"

    cfg.fem = "conservative"
    cfg.fem.method = "hdg"
    cfg.fem.mixed_method = "inactive"

    cfg.mach_number = 0.3
    cfg.equation_of_state = "ideal"
    cfg.equation_of_state.heat_capacity_ratio = 1.4
    cfg.dynamic_viscosity = "inviscid"
    cfg.scaling = "aerodynamic"
    cfg.scaling.dimensionful_values = {'length': 1.0, 'density': 1.293, 'velocity': 1.0,
                                       'speed_of_sound': 343.0, 'temperature': 293.15, 'pressure': 101325.0}
    cfg.riemann_solver = "lax_friedrich"

    yield cfg


def test_inviscid_finite_element_spaces(cfg):
    spaces = {}
    cfg.dynamic_viscosity = "inviscid"
    cfg.fem.mixed_method = 'inactive'
    cfg.fem.add_finite_element_spaces(spaces)

    for space, expected in zip(spaces, ('U', 'Uhat'), strict=True):
        assert space == expected
        assert isinstance(spaces[space], ngs.ProductSpace)


def test_strain_heat_finite_element_spaces(cfg):
    spaces = {}
    cfg.dynamic_viscosity = "constant"
    cfg.fem.mixed_method = "strain_heat"
    cfg.fem.add_finite_element_spaces(spaces)

    for space, expected in zip(spaces, ('U', 'Uhat', 'Q'), strict=True):
        assert space == expected
        assert isinstance(spaces[space], ngs.ProductSpace)


def test_gradient_finite_element_spaces(cfg):
    spaces = {}
    cfg.dynamic_viscosity = "constant"
    cfg.fem.mixed_method = "gradient"
    cfg.fem.add_finite_element_spaces(spaces)

    for space, expected in zip(spaces, ('U', 'Uhat', 'Q'), strict=True):
        assert space == expected
        assert isinstance(spaces[space], ngs.ProductSpace)


def test_test_and_trial_functions(cfg):
    cfg.fem.initialize_finite_element_spaces()
    cfg.fem.initialize_trial_and_test_functions()

    assert tuple(cfg.fem.TnT) == ('U', 'Uhat')
    for trial, test in cfg.fem.TnT.values():
        assert isinstance(trial, ngs.comp.ProxyFunction)
        assert isinstance(test, ngs.comp.ProxyFunction)
