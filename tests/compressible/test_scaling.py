import pytest
import ngsolve as ngs

from dream.compressible import CompressibleFlowSolver

mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))
mip = mesh(0.5, 0.5)


def cfg(scaling_name, Mach_number=0.1):
    cfg = CompressibleFlowSolver(mesh)
    cfg.scaling = scaling_name
    cfg.mach_number = Mach_number
    return cfg.scaling


@pytest.mark.parametrize("scaling, expected_velocity", [
    (cfg("aerodynamic", 0.1), 1),
    (cfg("acoustic", 0.1), 0.1),
    (cfg("aeroacoustic", 0.1), 0.1 / (1 + 0.1)),
])
def test_velocity(scaling, expected_velocity):
    u = scaling.velocity
    if isinstance(u, ngs.CF):
        u = u(mip)
    assert u == pytest.approx(expected_velocity)


@pytest.mark.parametrize("scaling, expected_sound", [
    (cfg("aerodynamic", 0.2), 5),
    (cfg("acoustic", 0.2),  1),
    (cfg("aeroacoustic", 0.2), 1 / (1 + 0.2)),
])
def test_speed_of_sound(scaling, expected_sound):
    c = scaling.speed_of_sound
    if isinstance(c, ngs.CF):
        c = c(mip)
    assert c == pytest.approx(expected_sound)


def test_aerodynamic_speed_of_sound_raises():
    scaling = cfg("aerodynamic", 0.0)
    with pytest.raises(ValueError):
        scaling.speed_of_sound
