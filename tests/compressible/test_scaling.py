import pytest
import ngsolve as ngs

from dream.compressible import CompressibleFlowSolver

mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))
mip = mesh(0.5, 0.5)


def cfg(scaling_name):
    cfg = CompressibleFlowSolver(mesh)
    cfg.scaling = scaling_name
    return cfg.scaling


@pytest.mark.parametrize("scaling, velocity_input, expected_velocity", [
    (cfg("aerodynamic"), 0.1, 1),
    (cfg("acoustic"), 0.1, 0.1),
    (cfg("aeroacoustic"), 0.1, 0.1 / (1 + 0.1)),
])
def test_velocity_magnitude(scaling, velocity_input, expected_velocity):
    assert scaling.velocity_magnitude(velocity_input) == pytest.approx(expected_velocity)


@pytest.mark.parametrize("scaling, sound_input, expected_sound", [
    (cfg("aerodynamic"), 0.2, 5),
    (cfg("acoustic"), 0.2, 1),
    (cfg("aeroacoustic"), 0.2, 1 / (1 + 0.2)),
])
def test_speed_of_sound(scaling, sound_input, expected_sound):
    assert scaling.speed_of_sound(sound_input) == pytest.approx(expected_sound)


def test_aerodynamic_speed_of_sound_raises():
    scaling = cfg("aerodynamic")
    with pytest.raises(ValueError):
        scaling.speed_of_sound(0)
