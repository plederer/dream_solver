import pytest
import ngsolve as ngs

from dream.compressible import CompressibleFlowSolver, flowfields, dimensionalfields

mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))
mip = mesh(0.5, 0.5)


@pytest.fixture
def inviscid():
    cfg = CompressibleFlowSolver(mesh)
    cfg.dynamic_viscosity = "inviscid"
    return cfg.dynamic_viscosity


def test_is_inviscid(inviscid):
    assert inviscid.is_inviscid


def test_viscosity_inviscid(inviscid):
    with pytest.raises(TypeError):
        inviscid.viscosity(flowfields())


@pytest.fixture
def constant():
    cfg = CompressibleFlowSolver(mesh)
    cfg.dynamic_viscosity = "constant"
    return cfg.dynamic_viscosity


def test_is_not_inviscid(constant):
    assert not constant.is_inviscid


def test_viscosity_constant(constant):
    assert pytest.approx(constant.viscosity(flowfields())) == 1


@pytest.fixture
def sutherland():
    cfg = CompressibleFlowSolver(mesh)
    cfg.mach_number = 1
    cfg.equation_of_state = "ideal"
    cfg.equation_of_state.heat_capacity_ratio = 1.4

    cfg.dynamic_viscosity = "sutherland"
    cfg.dynamic_viscosity.sutherland_temperature = 1

    cfg._dim_fields = dimensionalfields(rho_inf=1, u_inf=1, T_inf=1)

    return cfg


def test_is_not_inviscid_sutherland(sutherland):
    mu = sutherland.dynamic_viscosity
    assert not mu.is_inviscid


def test_viscosity_sutherland(sutherland):

    mu = sutherland.dynamic_viscosity
    fields = flowfields()
    assert mu.viscosity(fields) is None

    fields = flowfields(temperature=1)

    sutherland.scaling = "aerodynamic"
    assert mu.viscosity(fields)(mip) == pytest.approx((0.4)**(3/2) * (2/0.4)/(1+1/0.4))

    sutherland.scaling = "acoustic"
    assert mu.viscosity(fields)(mip) == pytest.approx((0.4)**(3/2) * (2/0.4)/(1+1/0.4))

    sutherland.scaling = "aeroacoustic"
    assert mu.viscosity(fields)(mip) == pytest.approx((1.6)**(3/2) * (2/1.6)/(1+1/1.6))
