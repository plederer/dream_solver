# %%
from __future__ import annotations
import unittest

from dream.config import configuration, ngsdict, quantity, InterfaceConfiguration, UniqueConfiguration, parameter, interface

# ------- Setup ------- #


class teststate(ngsdict):

    rho = quantity('density')
    p = quantity('pressure')


class Interface(InterfaceConfiguration, is_interface=True):
    ...


class OptionA(Interface):
    name = "option_a"
    aliases = ("A", )

    @configuration(default=0.0)
    def x(self, x: float):
        if x < 0.0:
            raise ValueError("Non negative x not allowed!")
        return float(x)

    @configuration(default=0.0)
    def y(self, y: float):
        return float(y)

    @y.getter_check
    def y(self):
        if self.x >= 10:
            ValueError("x is too large!")


class OptionB(Interface):
    name = "option_b"

    @configuration(default="abc")
    def alphabet(self, abc: str):
        return str(abc)

    @parameter(default=2.0)
    def number(self, number: float):
        return number


class MainConfiguration(UniqueConfiguration):

    other = interface(default=OptionA)
    param = parameter(default=2.0)

# ------- Tests ------- #


class TestVariable(unittest.TestCase):

    def setUp(self):
        self.obj = teststate()

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
        self.assertEqual(self.obj.to_py()['pressure'], 10)

        self.obj.p = None
        self.assertEqual(self.obj.to_py()['pressure'], 10)

    def test_deleter(self):
        self.obj.p = 20
        del self.obj.p
        self.assertEqual(self.obj.p, None)


class TestState(unittest.TestCase):

    def setUp(self) -> None:
        self.obj = teststate()

    def tearDown(self) -> None:
        self.obj.clear()

    def test_update(self):
        self.obj.update({"rho": 2, "p": 5, "u": 5})

        self.assertDictEqual(self.obj.to_py(), {"density": 2, "pressure": 5, 'u': 5})

    def test_keys(self):
        self.obj.update({"rho": 2, "p": 5, "u": 5})
        self.assertTupleEqual(tuple(self.obj.keys()), ("density", "pressure", 'u'))

    def test_values(self):
        self.obj.update({"rho": 2, "p": 5, "u": 5})
        self.assertTupleEqual(tuple(self.obj.to_py().values()), (2, 5, 5))

    def test_items(self):
        self.obj.update({"rho": 2, "p": 5, "u": 5})
        self.assertTupleEqual(tuple(self.obj.to_py().items()), (('density', 2), ('pressure', 5), ('u', 5)))

    def test_set_item(self):
        self.obj["rho"] = 2
        self.obj["p"] = 5
        self.obj["u"] = 5

        self.assertDictEqual(self.obj.to_py(), {"density": 2, "pressure": 5, 'u': 5})

    def test_get_item(self):
        self.assertEqual(self.obj.p, None)

        with self.assertRaises(KeyError):
            self.obj["p"]

    def test_iterator(self):
        self.obj.update({"rho": 2, "p": 5, "u": 5})
        self.assertTupleEqual(tuple(self.obj), ("density", "pressure", 'u'))

    def test_length(self):
        self.obj.update({"rho": 2, "p": 5, "u": 5})
        self.assertEqual(len(self.obj), 3)


class TestDescriptorConfigurationChild(unittest.TestCase):

    def setUp(self):
        self.obj = OptionA()

    def tearDown(self) -> None:
        self.obj.clear()

    def test_inheritance_tree(self):
        self.assertDictEqual(self.obj.tree.leafs, {'option_a': OptionA, 'a': OptionA,  'option_b': OptionB})

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
        self.assertDictEqual(self.obj.__dict__, {'cfg': self.obj.cfg, 'data': {'x': 0.0, 'y': 0.0}, 'z': 2})

    def test_root_id(self):
        self.assertEqual(id(self.obj.cfg), id(self.obj))

    def test_export_default(self):
        self.assertDictEqual(self.obj.to_tree(), {'x': 0.0, 'y': 0.0})

    def test_export_parent_argument(self):
        self.assertDictEqual(self.obj.to_tree(root="parent"), {'parent.x': 0.0, 'parent.y': 0.0})

    def test_export_data_argument(self):
        self.assertDictEqual(self.obj.to_tree(data={'z': 2}), {'x': 0.0, 'y': 0.0, 'z': 2})


class TestUniqueConfiguration(unittest.TestCase):

    def setUp(self):
        self.obj = MainConfiguration()

    def tearDown(self) -> None:
        self.obj.clear()

    def test_root_id(self):
        self.assertEqual(id(self.obj), id(self.obj.other.cfg))

    def test_export_default(self):
        self.assertDictEqual(self.obj.to_tree(), {'other': 'option_a',
                             'other.x': 0.0, 'other.y': 0.0, 'param': 2.0})

    def test_export_parent(self):
        self.assertDictEqual(
            self.obj.to_tree(root='unique'),
            {'unique.other.x': 0.0, 'unique.other.y': 0.0, 'unique.param': 2.0})

    def test_export_data_argument(self):
        self.assertDictEqual(
            self.obj.to_tree(data={'z': 2}),
            {'other.x': 0.0, 'other.y': 0.0, 'param': 2.0, 'z': 2})

    def test_update_subconfiguration(self):
        self.obj.update({'other.x': 3.0, 'other.y': 5.0, 'param': 5})
        self.assertDictEqual(self.obj.to_tree(), {'other': 'option_a',
                             'other.x': 3.0, 'other.y': 5.0, 'param': 5.0})

    def test_export_parent(self):
        self.assertDictEqual(
            self.obj.to_tree(root='unique'),
            {'unique.other': 'option_a', 'unique.other.x': 0.0, 'unique.other.y': 0.0, 'unique.param': 2.0})

    def test_export_data_argument(self):
        self.assertDictEqual(
            self.obj.to_tree(data={'z': 2}),
            {'other': 'option_a', 'other.x': 0.0, 'other.y': 0.0, 'param': 2.0, 'z': 2})

    def test_set_subconfiguration_by_string(self):
        self.obj.other = "option_b"
        self.assertDictEqual(
            self.obj.to_tree(),
            {'other': 'option_b', 'other.alphabet': 'abc', 'other.number': 2.0, 'param': 2.0})

    def test_set_subconfiguration_by_instance(self):
        self.obj.other = OptionB()
        self.assertDictEqual(self.obj.to_tree(), {'other.alphabet': 'abc', 'other.number': 2.0, 'param': 2.0})

    def test_set_subconfiguration_by_dict(self):
        self.obj.other = {'x': 2.0, 'y': 3.0, 'z': 5}
        self.assertDictEqual(self.obj.to_tree(), {'other': 'option_a',
                             'other.x': 2.0, 'other.y': 3.0, 'param': 2.0})

    def test_set_subconfiguration_by_string_id(self):
        old = id(self.obj.other)
        self.obj.other = "option_a"
        new = id(self.obj.other)
        self.assertEqual(old, new)

    def test_set_subconfiguration_by_instance(self):
        old = id(self.obj.other)
        self.obj.other = OptionA()
        new = id(self.obj.other)
        self.assertNotEqual(old, new)

    def test_set_subconfiguration_by_dict(self):
        old = id(self.obj.other)
        self.obj.other = {'x': 2.0, 'y': 3.0, 'z': 5}
        new = id(self.obj.other)
        self.assertEqual(old, new)


# %%
