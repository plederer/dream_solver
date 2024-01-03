from tests import ngs, unit_square, unit_circle
import unittest
import dream.mesh as dmesh
import logging

logging.disable(50)


class Mesh(unittest.TestCase):

    def setUp(self) -> None:
        self.cmesh = dmesh.DreamMesh(unit_square(0.1, periodic=True))
        self.pmesh = dmesh.DreamMesh(unit_circle(0.1, shift=(1, 1)))
        self.pmesh.ngsmesh.Curve(5)

    def test_boundaries(self):
        self.assertSequenceEqual(("bottom", "left", "right", "top"), sorted(self.cmesh.boundaries))

    def test_domains(self):
        self.assertSequenceEqual(("layer_0", "layer_1", "layer_2", "layer_3"), sorted(self.cmesh.domains))

    def test_dim(self):
        self.assertEqual(self.cmesh.dim, 2)

    def test_periodic(self):
        self.assertTrue(self.cmesh.is_periodic)

    def test_pattern(self):
        self.assertEqual(dmesh.pattern(["left", "right"]), "left|right")
        self.assertEqual(dmesh.pattern("left"), "left")

    def test_cartesian_grid_deformation(self):
        x = dmesh.BufferCoord.x(0.25, 0.5)
        x = dmesh.GridMapping.linear(5, x)

        dom = {"layer_3": dmesh.GridDeformation(x=x, order=1)}
        gfu = self.cmesh.get_grid_deformation(dom)

        self.assertIsInstance(gfu, ngs.GridFunction)
        self.assertAlmostEqual(ngs.Integrate(ngs.Norm(gfu), self.cmesh.ngsmesh, order=1), 0.125)

    def test_cartesian_sponge_layer(self):

        dom = {}
        for i, x_ in enumerate([-0.25, 0, 0.25]):
            x = dmesh.BufferCoord.x(x_, x_ + 0.25)
            sigma = dmesh.SpongeFunction.polynomial(2, x, order=5)
            dom[f"layer_{i + 1}"] = dmesh.SpongeLayer(sigma, None, order=5)

        gfu = self.cmesh.get_sponge_function(dom)

        self.assertIsInstance(gfu, ngs.GridFunction)
        self.assertAlmostEqual(ngs.Integrate(ngs.Norm(gfu), self.cmesh.ngsmesh, order=5), 0.25)

    def test_polar_grid_deformation(self):
        r = dmesh.BufferCoord.polar(0.125, 0.5, shift=(1, 1))
        map = dmesh.GridMapping.linear(5, r)
        x, y = map.polar_to_cartesian()

        d = dmesh.GridDeformation(x=x, y=y, order=5)
        dom = {"layer_1": d, "layer_2": d, "layer_3": d}

        gfu = self.pmesh.get_grid_deformation(dom)

        self.assertIsInstance(gfu, ngs.GridFunction)
        self.assertAlmostEqual(ngs.Integrate(ngs.Norm(gfu), self.pmesh.ngsmesh, order=5), 0.6626797003665968)

    def test_polar_sponge_layer(self):

        dom = {}
        for i, r_ in enumerate([0.125, 0.25, 0.375]):
            r = dmesh.BufferCoord.polar(r_, r_ + 0.125, shift=(1, 1))
            sigma = dmesh.SpongeFunction.polynomial(2, r, order=5)
            dom[f"layer_{i + 1}"] = dmesh.SpongeLayer(sigma, None, order=5)

        gfu = self.pmesh.get_sponge_function(dom)

        self.assertIsInstance(gfu, ngs.GridFunction)
        result = 2/3 * ngs.pi * 0.125 * (0.5 + 0.375 + 0.25 - 3/7 * 0.125)
        self.assertAlmostEqual(ngs.Integrate(ngs.Norm(gfu), self.pmesh.ngsmesh, order=5), result)


class BoundaryConditions(unittest.TestCase):

    def setUp(self) -> None:
        self.bcs = dmesh.BoundaryConditions(("a", "a", "b", "c"))

    def test_label_uniqueness(self):
        self.assertTupleEqual(self.bcs.regions, ("a", "b", "c"))

    def test_set(self):
        a = dmesh.Periodic()
        d = dmesh.Domain()

        self.bcs.set(a, "d")
        self.assertDictEqual(self.bcs.data, {})

        self.bcs.set(a, "c")
        self.assertDictEqual(self.bcs.data, {'Periodic': {'c': a}})
        self.bcs.clear()

        self.bcs.set(a, "a|b|c")
        self.assertDictEqual(self.bcs.data, {'Periodic': {'a': a, 'b': a, 'c': a}})
        self.bcs.clear()

        with self.assertRaises(TypeError):
            self.bcs.set(d, "a")

    def test_get(self):
        a = dmesh.Periodic()
        d = dmesh.Domain()

        self.bcs.set(a, "c")
        self.assertDictEqual(self.bcs.get(a), {'c': a})
        self.bcs.clear()

        with self.assertRaises(TypeError):
            self.bcs.get(d, "c")

    def test_to_pattern(self):
        a = dmesh.Periodic()

        self.bcs.set(a, "a|b|c")
        pattern = dmesh.pattern(self.bcs.get(a))
        self.assertDictEqual(self.bcs.to_unique_pattern().data, {'Periodic': {pattern: a}})
        self.bcs.clear()


class DomainConditions(unittest.TestCase):

    def setUp(self) -> None:
        self.dcs = dmesh.DomainConditions(("a", "a", "b", "c"))

    def test_label_uniqueness(self):
        self.assertTupleEqual(self.dcs.regions, ("a", "b", "c"))

    def test_set(self):
        a = dmesh.Periodic()
        d = dmesh.Domain()

        self.dcs.set(d, "d")
        self.assertDictEqual(self.dcs.data, {})

        self.dcs.set(d, "c")
        self.assertDictEqual(self.dcs.data, {'Domain': {'c': d}})
        self.dcs.clear()

        self.dcs.set(d, "a|b|c")
        self.assertDictEqual(self.dcs.data, {'Domain': {'a': d, 'b': d, 'c': d}})
        self.dcs.clear()

        with self.assertRaises(TypeError):
            self.dcs.set(a, "a")

    def test_get(self):
        a = dmesh.Periodic()
        d = dmesh.Domain()

        self.dcs.set(d, "c")
        self.assertDictEqual(self.dcs.get(d), {'c': d})
        self.dcs.clear()

        with self.assertRaises(TypeError):
            self.dcs.get(a, "c")

    def test_to_pattern(self):
        d = dmesh.Domain()

        self.dcs.set(d, "a|b|c")
        pattern = dmesh.pattern(self.dcs.get(d))
        self.assertDictEqual(self.dcs.to_unique_pattern().data, {'Domain': {pattern: d}})
        self.dcs.clear()


class BufferCoord(unittest.TestCase):

    def setUp(self) -> None:
        self.cmesh = unit_square()
        self.pmesh = unit_circle(shift=(1, 1))
        self.pmesh.Curve(6)

    def test_instance(self):
        x = dmesh.BufferCoord.x(-0.25, 0.25)
        self.assertIsInstance(x, ngs.CF)

    def test_x(self):
        x = dmesh.BufferCoord.x(-0.25, 0.25)
        x_ = x.get_normalised_coordinate()

        self.assertAlmostEqual(x.length, 0.5)
        self.assertAlmostEqual(ngs.Integrate(x, self.cmesh), 0)
        self.assertAlmostEqual(ngs.Integrate(x_, self.cmesh), 0.5)

    def test_y(self):
        y = dmesh.BufferCoord.y(-0.5, 0.5)
        y_ = y.get_normalised_coordinate()

        self.assertAlmostEqual(y.length, 1)
        self.assertAlmostEqual(ngs.Integrate(y, self.cmesh), 0)
        self.assertAlmostEqual(ngs.Integrate(y_, self.cmesh), 0.5)

    def test_r(self):
        r = dmesh.BufferCoord.polar(0.125, 0.375, shift=(1, 1))
        r_ = r.get_normalised_coordinate()

        self.assertAlmostEqual(r.length, 0.25)
        self.assertAlmostEqual(ngs.Integrate(r, self.pmesh), 0.24134631062734085)
        self.assertAlmostEqual(ngs.Integrate(r_, self.pmesh), 0.5726861608106394)


class GridMapping(unittest.TestCase):

    def setUp(self) -> None:
        self.pmesh = unit_circle(shift=(1, 1))
        self.r = dmesh.BufferCoord.polar(0.125, 0.5, shift=(1, 1))

    def test_none(self):
        map = dmesh.GridMapping.none(self.r)

        self.assertAlmostEqual(map.length, 0.375)
        self.assertAlmostEqual(map(0.5), 0.5)

    def test_linear(self):
        map = dmesh.GridMapping.linear(5, self.r)

        self.assertAlmostEqual(map.length, 1.875)
        self.assertAlmostEqual(map(0.5), 2.0)

    def test_exponential(self):
        map = dmesh.GridMapping.exponential(5, self.r)

        self.assertAlmostEqual(map.length, 1.875)
        self.assertAlmostEqual(map(0.5), 2.0)

    def test_tangential(self):
        map = dmesh.GridMapping.tangential(5, self.r)

        self.assertAlmostEqual(map.length, 1.875)
        self.assertAlmostEqual(map(0.5), 2.0)
