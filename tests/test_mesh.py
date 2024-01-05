from tests import ngs, unit_square, unit_circle
import logging
import unittest

import dream.mesh as mesh

logging.disable(50)


class Mesh(unittest.TestCase):

    def setUp(self) -> None:
        self.cmesh = mesh.DreamMesh(unit_square(0.1, periodic=True))
        self.pmesh = mesh.DreamMesh(unit_circle(0.1, shift=(1, 1)))
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
        self.assertEqual(mesh.pattern(["left", "right"]), "left|right")
        self.assertEqual(mesh.pattern("left"), "left")

    def test_cartesian_grid_deformation(self):
        x = mesh.BufferCoord.x(0.25, 0.5)
        x = mesh.GridMapping.linear(5, x)

        dom = {"layer_3": mesh.GridDeformation(x=x, order=1)}
        gfu = self.cmesh.get_grid_deformation(dom)

        self.assertIsInstance(gfu, ngs.GridFunction)
        self.assertAlmostEqual(ngs.Integrate(ngs.Norm(gfu), self.cmesh.ngsmesh, order=1), 0.125)

    def test_cartesian_sponge_layer(self):

        dom = {}
        for i, x_ in enumerate([-0.25, 0, 0.25]):
            x = mesh.BufferCoord.x(x_, x_ + 0.25)
            sigma = mesh.SpongeFunction.polynomial(2, x, order=5)
            dom[f"layer_{i + 1}"] = mesh.SpongeLayer(sigma, None, order=5)

        gfu = self.cmesh.get_sponge_function(dom)

        self.assertIsInstance(gfu, ngs.GridFunction)
        self.assertAlmostEqual(ngs.Integrate(ngs.Norm(gfu), self.cmesh.ngsmesh, order=5), 0.25)

    def test_polar_grid_deformation(self):
        r = mesh.BufferCoord.polar(0.125, 0.5, shift=(1, 1))
        map = mesh.GridMapping.linear(5, r)
        x, y = map.polar_to_cartesian()

        d = mesh.GridDeformation(x=x, y=y, order=5)
        dom = {"layer_1": d, "layer_2": d, "layer_3": d}

        gfu = self.pmesh.get_grid_deformation(dom)

        self.assertIsInstance(gfu, ngs.GridFunction)
        self.assertAlmostEqual(ngs.Integrate(ngs.Norm(gfu), self.pmesh.ngsmesh, order=5), 0.6626797003665968)

    def test_polar_sponge_layer(self):

        dom = {}
        for i, r_ in enumerate([0.125, 0.25, 0.375]):
            r = mesh.BufferCoord.polar(r_, r_ + 0.125, shift=(1, 1))
            sigma = mesh.SpongeFunction.polynomial(2, r, order=5)
            dom[f"layer_{i + 1}"] = mesh.SpongeLayer(sigma, None, order=5)

        gfu = self.pmesh.get_sponge_function(dom)

        self.assertIsInstance(gfu, ngs.GridFunction)
        result = 2/3 * ngs.pi * 0.125 * (0.5 + 0.375 + 0.25 - 3/7 * 0.125)
        self.assertAlmostEqual(ngs.Integrate(ngs.Norm(gfu), self.pmesh.ngsmesh, order=5), result)


class BoundaryConditions(unittest.TestCase):

    def setUp(self) -> None:
        self.bcs = mesh.BoundaryConditions(("a", "a", "b", "c"))

    def test_label_uniqueness(self):
        self.assertTupleEqual(tuple(self.bcs), ("a", "b", "c"))

    def test_domain_boundaries(self):
        a = mesh.Boundary()
        b = mesh.Periodic()

        self.bcs.set(a, "a|b")
        self.bcs.set(b, "c")

        self.assertDictEqual(self.bcs.data, {"a": a, "b": a, "c": b})
        self.assertListEqual(self.bcs.get_domain_boundaries(), ["a", "b"])
        self.assertEqual(self.bcs.get_domain_boundaries(True), "a|b")

    def test_set(self):
        a = mesh.Periodic()
        d = mesh.Domain()

        self.bcs.set(a, "d")
        self.assertDictEqual(self.bcs.data, {'a': None, 'b': None, 'c': None})

        self.bcs.set(a, "c")
        self.assertDictEqual(self.bcs.data, {'a': None, 'b': None, 'c': a})
        self.bcs.clear()

        self.bcs.set(a, "a|b|c")
        self.assertDictEqual(self.bcs.data, {'a': a, 'b': a, 'c': a})
        self.bcs.clear()

        with self.assertRaises(TypeError):
            self.bcs.set(d, "a")

    def test_get(self):
        a = mesh.Periodic()
        b = mesh.Periodic()
        d = mesh.Domain()

        self.bcs.set(a, "a")
        self.assertDictEqual(self.bcs.get(a), {'a': a})
        self.bcs.clear()

        self.bcs.set(a, "a|b")
        self.bcs.set(b, "c")
        self.assertDictEqual(self.bcs.get(a), {'a': a, 'b': a, 'c': b})
        self.assertDictEqual(self.bcs.get(a, True), {'a|b': a, 'c': b})
        self.bcs.clear()

        with self.assertRaises(TypeError):
            self.bcs.get(d, "c")


class DomainConditions(unittest.TestCase):

    def setUp(self) -> None:
        self.dcs = mesh.DomainConditions(("a", "a", "b", "c"))

    def test_label_uniqueness(self):
        self.assertTupleEqual(tuple(self.dcs), ("a", "b", "c"))

    def test_set(self):
        a = mesh.Periodic()
        d = mesh.Domain()

        self.dcs.set(d, "d")
        self.assertDictEqual(self.dcs.data, {'a': {}, 'b': {}, 'c': {}})

        self.dcs.set(d, "c")
        self.assertDictEqual(self.dcs.data, {'a': {}, 'b': {}, 'c': {'Domain': d}})
        self.dcs.clear()

        self.dcs.set(d, "a|b|c")
        self.assertDictEqual(self.dcs.data, {'a': {'Domain': d}, 'b': {'Domain': d}, 'c': {'Domain': d}})
        self.dcs.clear()

        with self.assertRaises(TypeError):
            self.dcs.set(a, "a")

    def test_get(self):
        a = mesh.Periodic()
        c = mesh.Domain()
        d = mesh.Domain()

        self.dcs.set(d, "c")
        self.assertDictEqual(self.dcs.get(d), {'c': d})
        self.dcs.clear()

        self.dcs.set(c, "a|b")
        self.dcs.set(d, "c")
        self.assertDictEqual(self.dcs.get(d, True), {'a|b': c, 'c': d})
        self.dcs.clear()

        with self.assertRaises(TypeError):
            self.dcs.get(a)


class BufferCoord(unittest.TestCase):

    def setUp(self) -> None:
        self.cmesh = unit_square()
        self.pmesh = unit_circle(shift=(1, 1))
        self.pmesh.Curve(6)

    def test_instance(self):
        x = mesh.BufferCoord.x(-0.25, 0.25)
        self.assertIsInstance(x, ngs.CF)

    def test_x(self):
        x = mesh.BufferCoord.x(-0.25, 0.25)
        x_ = x.get_normalised_coordinate()

        self.assertAlmostEqual(x.length, 0.5)
        self.assertAlmostEqual(ngs.Integrate(x, self.cmesh), 0)
        self.assertAlmostEqual(ngs.Integrate(x_, self.cmesh), 0.5)

    def test_y(self):
        y = mesh.BufferCoord.y(-0.5, 0.5)
        y_ = y.get_normalised_coordinate()

        self.assertAlmostEqual(y.length, 1)
        self.assertAlmostEqual(ngs.Integrate(y, self.cmesh), 0)
        self.assertAlmostEqual(ngs.Integrate(y_, self.cmesh), 0.5)

    def test_r(self):
        r = mesh.BufferCoord.polar(0.125, 0.375, shift=(1, 1))
        r_ = r.get_normalised_coordinate()

        self.assertAlmostEqual(r.length, 0.25)
        self.assertAlmostEqual(ngs.Integrate(r, self.pmesh), 0.24134631062734085)
        self.assertAlmostEqual(ngs.Integrate(r_, self.pmesh), 0.5726861608106394)


class GridMapping(unittest.TestCase):

    def setUp(self) -> None:
        self.pmesh = unit_circle(shift=(1, 1))
        self.r = mesh.BufferCoord.polar(0.125, 0.5, shift=(1, 1))

    def test_none(self):
        map = mesh.GridMapping.none(self.r)

        self.assertAlmostEqual(map.length, 0.375)
        self.assertAlmostEqual(map(0.5), 0.5)

    def test_linear(self):
        map = mesh.GridMapping.linear(5, self.r)

        self.assertAlmostEqual(map.length, 1.875)
        self.assertAlmostEqual(map(0.5), 2.0)

    def test_exponential(self):
        map = mesh.GridMapping.exponential(5, self.r)

        self.assertAlmostEqual(map.length, 1.875)
        self.assertAlmostEqual(map(0.5), 2.0)

    def test_tangential(self):
        map = mesh.GridMapping.tangential(5, self.r)

        self.assertAlmostEqual(map.length, 1.875)
        self.assertAlmostEqual(map(0.5), 2.0)
