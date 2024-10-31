# %%
from __future__ import annotations
import unittest

import dream.config as cfg

# ------- Setup ------- #


class DummyState(cfg.State):

    rho = cfg.quantity(lambda x: 2*x, 'density')
    p = cfg.quantity(lambda x: 5*x, 'pressure')


class MultipleConfiguration(cfg.MultipleConfiguration, is_interface=True):
    ...


class MultipleA(MultipleConfiguration):
    name = "child_A"
    aliases = ("A", )

    @cfg.any(default=0.0)
    def x(self, x: float):
        if x < 0.0:
            raise ValueError("Non negative x not allowed!")
        return float(x)

    @cfg.any(default=0.0)
    def y(self, y: float):
        return float(y)

    @y.getter_check
    def y(self):
        if self.x >= 10:
            ValueError("x is too large!")


class MultipleB(MultipleConfiguration):
    name = "child_B"

    @cfg.any(default="abc")
    def alphabet(self, abc: str):
        return str(abc)

    @cfg.parameter(default=2.0)
    def number(self, number: float):
        return number


class UniqueConfiguration_(cfg.UniqueConfiguration):

    other = cfg.multiple(default=MultipleA)
    param = cfg.parameter(default=2.0)

# ------- Tests ------- #


class TestVariable(unittest.TestCase):

    def setUp(self):
        self.obj = DummyState()

    def test_variable_name(self):
        self.assertEqual(type(self.obj).rho.name, "density")

    def test_variable_symbol(self):
        self.assertEqual(type(self.obj).rho.symbol, "rho")

    def tearDown(self) -> None:
        self.obj.clear()

    def test_label(self):
        self.obj.p = 10
        self.assertIn("pressure", self.obj)

    def test_getter(self):
        self.assertEqual(self.obj.p, None)

    def test_setter(self):
        self.obj.p = 10
        self.assertEqual(self.obj.p, 50)

        self.obj.p = None
        self.assertEqual(self.obj.p, 50)

    def test_deleter(self):
        self.obj.p = 20
        del self.obj.p
        self.assertEqual(self.obj.p, None)


class TestState(unittest.TestCase):

    def setUp(self) -> None:
        self.obj = DummyState()

    def tearDown(self) -> None:
        self.obj.clear()

    def test_update(self):
        self.obj.update({"rho": 2, "p": 5, "u": 5})

        self.assertDictEqual(self.obj.data, {"density": 4, "pressure": 25, 'u': 5})
        self.assertDictEqual(self.obj.__dict__, {"data": self.obj.data})

    def test_keys(self):
        self.obj.update({"rho": 2, "p": 5, "u": 5})
        self.assertTupleEqual(tuple(self.obj.keys()), ("density", "pressure", 'u'))

    def test_values(self):
        self.obj.update({"rho": 2, "p": 5, "u": 5})
        self.assertTupleEqual(tuple(self.obj.values()), (4, 25, 5))

    def test_items(self):
        self.obj.update({"rho": 2, "p": 5, "u": 5})
        self.assertTupleEqual(tuple(self.obj.items()), (('density', 4), ('pressure', 25), ('u', 5)))

    def test_set_item(self):
        self.obj["rho"] = 2
        self.obj["p"] = 5
        self.obj["u"] = 5

        self.assertDictEqual(self.obj.data, {"density": 4, "pressure": 25, 'u': 5})
        self.assertDictEqual(self.obj.__dict__, {"data": self.obj.data})

    def test_get_item(self):
        self.assertEqual(self.obj["p"], None)

    def test_iterator(self):
        self.obj.update({"rho": 2, "p": 5, "u": 5})
        self.assertTupleEqual(tuple(self.obj), ("density", "pressure", 'u'))

    def test_length(self):
        self.obj.update({"rho": 2, "p": 5, "u": 5})
        self.assertEqual(len(self.obj), 3)


class TestDescriptorConfigurationChild(unittest.TestCase):

    def setUp(self):
        self.obj = MultipleA()

    def tearDown(self) -> None:
        self.obj.clear()

    def test_inheritance_tree(self):
        self.assertDictEqual(self.obj.leafs.data, {'child_a': MultipleA, 'a': MultipleA,  'child_b': MultipleB})

    def test_configurations_id_after_clear(self):
        old = [self.obj.x, self.obj.y]

        self.obj.clear()
        new = [self.obj.x, self.obj.y]

        for new, old in zip(old, new):
            self.assertEqual(id(new), id(old))

    def test_configurations_id_after_update(self):
        old = [self.obj.x, self.obj.y]

        self.obj.update({'x': 2, 'y': 0.3})
        new = [self.obj.x, self.obj.y]

        for new, old in zip(old, new):
            self.assertNotEqual(id(new), id(old))

    def test_clear(self):
        self.obj["z"] = 2
        self.obj.clear()

        self.assertDictEqual(self.obj.data, {'x': 0.0, 'y': 0.0})
        self.assertDictEqual(self.obj.__dict__, {'cfg': {'x': 0.0, 'y': 0.0}, 'data': {'x': 0.0, 'y': 0.0}, 'z': 2})

    def test_root_id(self):
        self.assertEqual(id(self.obj.cfg), id(self.obj))

    def test_export_default(self):
        self.assertDictEqual(self.obj.to_flat_dict(), {'x': 0.0, 'y': 0.0})

    def test_export_parent_argument(self):
        self.assertDictEqual(self.obj.to_flat_dict(root="parent"), {'parent.x': 0.0, 'parent.y': 0.0})

    def test_export_data_argument(self):
        self.assertDictEqual(self.obj.to_flat_dict(data={'z': 2}), {'x': 0.0, 'y': 0.0, 'z': 2})


class TestUniqueConfiguration(unittest.TestCase):

    def setUp(self):
        self.obj = UniqueConfiguration_()

    def tearDown(self) -> None:
        self.obj.clear()

    def test_root_id(self):
        self.assertEqual(id(self.obj), id(self.obj.other.cfg))

    def test_export_default(self):
        self.assertDictEqual(self.obj.to_flat_dict(), {'other': 'child_a',
                             'other.x': 0.0, 'other.y': 0.0, 'param': 2.0})

    def test_export_parent(self):
        self.assertDictEqual(
            self.obj.to_flat_dict(root='unique'),
            {'unique.other.x': 0.0, 'unique.other.y': 0.0, 'unique.param': 2.0})

    def test_export_data_argument(self):
        self.assertDictEqual(
            self.obj.to_flat_dict(data={'z': 2}),
            {'other.x': 0.0, 'other.y': 0.0, 'param': 2.0, 'z': 2})

    def test_update_subconfiguration(self):
        self.obj.update({'other.x': 3.0, 'other.y': 5.0, 'param': 5})
        self.assertDictEqual(self.obj.to_flat_dict(), {'other': 'child_a',
                             'other.x': 3.0, 'other.y': 5.0, 'param': 5.0})

    def test_export_parent(self):
        self.assertDictEqual(
            self.obj.to_flat_dict(root='unique'),
            {'unique.other': 'child_a', 'unique.other.x': 0.0, 'unique.other.y': 0.0, 'unique.param': 2.0})

    def test_export_data_argument(self):
        self.assertDictEqual(
            self.obj.to_flat_dict(data={'z': 2}),
            {'other': 'child_a', 'other.x': 0.0, 'other.y': 0.0, 'param': 2.0, 'z': 2})

    def test_set_subconfiguration_by_string(self):
        self.obj.other = "child_b"
        self.assertDictEqual(
            self.obj.to_flat_dict(),
            {'other': 'child_b', 'other.alphabet': 'abc', 'other.number': 2.0, 'param': 2.0})

    def test_set_subconfiguration_by_instance(self):
        self.obj.other = MultipleB()
        self.assertDictEqual(self.obj.to_flat_dict(), {'other.alphabet': 'abc', 'other.number': 2.0, 'param': 2.0})

    def test_set_subconfiguration_by_dict(self):
        self.obj.other = {'x': 2.0, 'y': 3.0, 'z': 5}
        self.assertDictEqual(self.obj.to_flat_dict(), {'other': 'child_a',
                             'other.x': 2.0, 'other.y': 3.0, 'param': 2.0})

    def test_set_subconfiguration_by_string_id(self):
        old = id(self.obj.other)
        self.obj.other = "child_a"
        new = id(self.obj.other)
        self.assertEqual(old, new)

    def test_set_subconfiguration_by_instance(self):
        old = id(self.obj.other)
        self.obj.other = MultipleA()
        new = id(self.obj.other)
        self.assertNotEqual(old, new)

    def test_set_subconfiguration_by_dict(self):
        old = id(self.obj.other)
        self.obj.other = {'x': 2.0, 'y': 3.0, 'z': 5}
        new = id(self.obj.other)
        self.assertEqual(old, new)


# %%
