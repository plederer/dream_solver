from __future__ import annotations
from ngsolve import Redraw, CF
from typing import TYPE_CHECKING


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        elif shell == 'TerminalInteractiveShell':
            return False
        else:
            return False
    except NameError:
        return False


if is_notebook():
    from ngsolve.webgui import Draw, WebGLScene
else:
    from ngsolve import Draw

if TYPE_CHECKING:
    from ..formulations import _Formulation


class Drawer:

    def __init__(self, formulation: _Formulation):
        self._formulation = formulation
        self._scenes: list[WebGLScene] = []

    @property
    def formulation(self):
        return self._formulation

    def draw(self,
             density: bool = True,
             velocity: bool = True,
             pressure: bool = True,
             energy: bool = False,
             temperature: bool = False,
             momentum: bool = False):

        if density:
            self.draw_density()

        if velocity:
            self.draw_velocity()

        if energy:
            self.draw_energy()

        if pressure:
            self.draw_pressure()

        if temperature:
            self.draw_temperature()

        if momentum:
            self.draw_momentum()

    def redraw(self, blocking: bool = False):
        for scene in self._scenes:
            scene.Redraw()
        Redraw(blocking)

    def draw_density(self, label: str = "rho", **kwargs):
        scene = Draw(self.formulation.density(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_momentum(self, label: str = "rho_u", **kwargs):
        scene = Draw(self.formulation.momentum(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_energy(self, label: str = "rho_E", **kwargs):
        scene = Draw(self.formulation.energy(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_pressure(self, label: str = "p", **kwargs):
        scene = Draw(self.formulation.pressure(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_temperature(self, label: str = "T", **kwargs):
        scene = Draw(self.formulation.temperature(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_velocity(self, label: str = "u", **kwargs):
        scene = Draw(self.formulation.velocity(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_vorticity(self, label: str = "omega", **kwargs):
        scene = Draw(self.formulation.vorticity(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_mach_number(self, label: str = "Mach", **kwargs):
        scene = Draw(self.formulation.mach_number(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_speed_of_sound(self, label: str = "c", **kwargs):
        scene = Draw(self.formulation.speed_of_sound(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_deviatoric_strain_tensor(self, label: str = "epsilon", **kwargs):
        scene = Draw(self.formulation.deviatoric_strain_rate_tensor(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_deviatoric_stress_tensor(self, label: str = "tau", **kwargs):
        scene = Draw(self.formulation.deviatoric_stress_tensor(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_heat_flux(self, label: str = "q", **kwargs):
        scene = Draw(self.formulation.heat_flux(), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_acoustic_density(self, mean_density: float, label: str = "rho'", **kwargs):
        scene = Draw(self.formulation.density() - mean_density, self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def draw_acoustic_pressure(self, mean_pressure: float, label: str = "p'", **kwargs):
        acc_pressure = self.formulation.pressure() - mean_pressure
        scene = Draw(acc_pressure, self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)
        return acc_pressure

    def draw_particle_velocity(self, mean_velocity: tuple[float, ...], label: str = "u'", **kwargs):
        scene = Draw(self.formulation.velocity() - CF(mean_velocity), self.formulation.mesh, label, **kwargs)
        self._append_scene(scene)

    def _append_scene(self, scene):
        if scene is not None:
            self._scenes.append(scene)
