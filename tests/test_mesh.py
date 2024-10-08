from tests import ngs, unit_square, unit_circle
import logging
import unittest
from netgen.occ import WorkPlane, OCCGeometry
from dream.mesh import (DomainConditions, BoundaryConditions, BufferCoord, GridMapping, GridDeformation, Periodic,
                         Condition, get_pattern_from_sequence, SpongeFunction, SpongeLayer)


logging.disable(50)

square = unit_square(0.1, periodic=True)
circle = unit_circle(0.1, shift=(1, 1))
circle.Curve(5)


class TestBoundaryConditions(unittest.TestCase):

    def setUp(self) -> None:
        geo = WorkPlane().LineTo(1, 0, "a").LineTo(1, 1, "a").LineTo(0, 1, "b").LineTo(0, 0, "c").Face()
        mesh = ngs.Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=1))
        self.bcs = BoundaryConditions(mesh)

    def test_label_uniqueness(self):
        self.assertTupleEqual(tuple(self.bcs), ("a", "b", "c"))

    def test_set_object(self):
        a = Condition()
        b = Periodic()

        self.bcs['d'] = a
        self.assertDictEqual(self.bcs.data, {'a': [], 'b': [], 'c': []})

        self.bcs['c'] = a
        self.assertDictEqual(self.bcs.data, {'a': [], 'b': [], 'c': [a]})
        self.bcs.clear()

        self.bcs['a|b'] = a
        self.bcs['c'] = b
        self.assertDictEqual(self.bcs.data, {"a": [a], "b": [a], "c": [b]})

    def test_set_pattern(self):
        self.bcs.register_condition(Periodic)

        self.bcs['d'] = "periodic"
        self.assertDictEqual(self.bcs.data, {'a': [], 'b': [], 'c': []})

        self.bcs['c'] = "periodic"
        self.assertDictEqual(self.bcs.data, {'a': [], 'b': [], 'c': self.bcs['c']})
        self.bcs.clear()

        self.bcs['a|b'] = 'periodic'
        self.bcs['c'] = 'periodic'
        self.assertDictEqual(self.bcs.data, {"a": self.bcs['a'], "b": self.bcs['a'], "c": self.bcs['c']})

        self.assertIsInstance(self.bcs['a'][0], Periodic)
        self.assertIsInstance(self.bcs['c'][0], Periodic)


class TestDomainConditions(unittest.TestCase):

    def setUp(self) -> None:
        self.dcs = DomainConditions(square)
        self.pcs = DomainConditions(circle)

    def test_domains(self):
        self.assertSequenceEqual(("layer_0", "layer_1", "layer_2", "layer_3"), sorted(self.dcs))

    def test_dim(self):
        self.assertEqual(self.dcs.mesh.dim, 2)

    def test_periodic(self):
        self.assertTrue(self.dcs.mesh.is_periodic)

    def test_pattern(self):
        self.assertEqual(get_pattern_from_sequence(["left", "right"]), "left|right")
        self.assertEqual(get_pattern_from_sequence("left"), "left")

    def test_cartesian_grid_deformation(self):
        x = BufferCoord.x(0.25, 0.5)
        x = GridMapping.linear(5, x)

        dom = {"layer_3": GridDeformation(x=x, order=1)}
        gfu = self.dcs.get_grid_deformation(dom)

        self.assertIsInstance(gfu, ngs.GridFunction)
        self.assertAlmostEqual(ngs.Integrate(ngs.Norm(gfu), self.dcs.mesh, order=1), 0.125)

    def test_cartesian_sponge_layer(self):

        dom = {}
        for i, x_ in enumerate([-0.25, 0, 0.25]):
            x = BufferCoord.x(x_, x_ + 0.25)
            sigma = SpongeFunction.polynomial(2, x, order=5)
            dom[f"layer_{i + 1}"] = SpongeLayer(function=sigma, target_state=1, order=5)

        gfu = self.dcs.get_sponge_function(dom)

        self.assertIsInstance(gfu, ngs.GridFunction)
        self.assertAlmostEqual(ngs.Integrate(ngs.Norm(gfu), self.dcs.mesh, order=5), 0.25)

    def test_polar_grid_deformation(self):
        r = BufferCoord.polar(0.125, 0.5, shift=(1, 1))
        map = GridMapping.linear(5, r)
        x, y = map.polar_to_cartesian()

        d = GridDeformation(x=x, y=y, order=5)
        dom = {"layer_1": d, "layer_2": d, "layer_3": d}

        gfu = self.pcs.get_grid_deformation(dom)

        self.assertIsInstance(gfu, ngs.GridFunction)
        self.assertAlmostEqual(ngs.Integrate(ngs.Norm(gfu), self.pcs.mesh, order=5), 0.6626797003665968)

    def test_polar_sponge_layer(self):

        dom = {}
        for i, r_ in enumerate([0.125, 0.25, 0.375]):
            r = BufferCoord.polar(r_, r_ + 0.125, shift=(1, 1))
            sigma = SpongeFunction.polynomial(2, r, order=5)
            dom[f"layer_{i + 1}"] = SpongeLayer(function=sigma, target_state=1, order=5)

        gfu = self.pcs.get_sponge_function(dom)

        self.assertIsInstance(gfu, ngs.GridFunction)
        result = 2/3 * ngs.pi * 0.125 * (0.5 + 0.375 + 0.25 - 3/7 * 0.125)
        self.assertAlmostEqual(ngs.Integrate(ngs.Norm(gfu), self.pcs.mesh, order=5), result)


class TestBufferCoord(unittest.TestCase):

    def setUp(self) -> None:
        self.cmesh = unit_square()
        self.pmesh = unit_circle(shift=(1, 1))
        self.pmesh.Curve(6)

    def test_instance(self):
        x = BufferCoord.x(-0.25, 0.25)
        self.assertIsInstance(x, ngs.CF)

    def test_x(self):
        x = BufferCoord.x(-0.25, 0.25)
        x_ = x.get_normalised_coordinate()

        self.assertAlmostEqual(x.length, 0.5)
        self.assertAlmostEqual(ngs.Integrate(x, self.cmesh), 0)
        self.assertAlmostEqual(ngs.Integrate(x_, self.cmesh), 0.5)

    def test_y(self):
        y = BufferCoord.y(-0.5, 0.5)
        y_ = y.get_normalised_coordinate()

        self.assertAlmostEqual(y.length, 1)
        self.assertAlmostEqual(ngs.Integrate(y, self.cmesh), 0)
        self.assertAlmostEqual(ngs.Integrate(y_, self.cmesh), 0.5)

    def test_r(self):
        r = BufferCoord.polar(0.125, 0.375, shift=(1, 1))
        r_ = r.get_normalised_coordinate()

        self.assertAlmostEqual(r.length, 0.25)
        self.assertAlmostEqual(ngs.Integrate(r, self.pmesh), 0.24134631062734085)
        self.assertAlmostEqual(ngs.Integrate(r_, self.pmesh), 0.5726861608106394)


class TestGridMapping(unittest.TestCase):

    def setUp(self) -> None:
        self.pmesh = unit_circle(shift=(1, 1))
        self.r = BufferCoord.polar(0.125, 0.5, shift=(1, 1))

    def test_none(self):
        map = GridMapping.none(self.r)

        self.assertAlmostEqual(map.length, 0.375)
        self.assertAlmostEqual(map(0.5), 0.5)

    def test_linear(self):
        map = GridMapping.linear(5, self.r)

        self.assertAlmostEqual(map.length, 1.875)
        self.assertAlmostEqual(map(0.5), 2.0)

    def test_exponential(self):
        map = GridMapping.exponential(5, self.r)

        self.assertAlmostEqual(map.length, 1.875)
        self.assertAlmostEqual(map(0.5), 2.0)

    def test_tangential(self):
        map = GridMapping.tangential(5, self.r)

        self.assertAlmostEqual(map.length, 1.875)
        self.assertAlmostEqual(map(0.5), 2.0)
