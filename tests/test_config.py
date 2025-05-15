# %%
from __future__ import annotations
import unittest
import ngsolve as ngs
from tests import simplex

from dream.config import ngsdict, quantity, Configuration, dream_configuration

# ------- Setup ------- #


class teststate(ngsdict):

    rho = quantity('density')
    p = quantity('pressure')


class DummyConfiguration(Configuration, is_interface=True):

    name = "point"

    def __init__(self, mesh, root=None, **default):
        DEFAULT = {
            "x": 0.0,
            "y": 0.0,
        }
        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = float(x)

    @dream_configuration
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = float(y)


class OptionConfiguration(Configuration, is_interface=True):
    ...


class OptionA(OptionConfiguration):
    name = "option_a"

    def __init__(self, mesh, root=None, **default):
        DEFAULT = {
            "x": 0.0,
            "y": 0.0
        }
        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        if x < 0.0:
            raise ValueError("Non negative x not allowed!")
        self._x = float(x)

    @dream_configuration
    def y(self):
        if self.x >= 10:
            ValueError("x is too large!")
        return self._y

    @y.setter
    def y(self, y):
        self._y = float(y)


class OptionB(OptionConfiguration):
    name = "option_b"

    def __init__(self, mesh, root=None, **default):

        self._number = ngs.Parameter(2.0)

        DEFAULT = {
            "alphabet": "abc",
            "number": 2.0,
            "sub": DummyConfiguration(mesh, root),
        }

        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def alphabet(self):
        return self._alphabet

    @alphabet.setter
    def alphabet(self, abc: str):
        self._alphabet = str(abc)

    @dream_configuration
    def number(self) -> ngs.Parameter:
        return self._number

    @number.setter
    def number(self, number: float):
        if isinstance(number, ngs.Parameter):
            self._number = number
        else:
            self._number.Set(number)

    @dream_configuration
    def sub(self):
        return self._sub

    @sub.setter
    def sub(self, sub):
        OPTIONS = [DummyConfiguration]
        self._sub = self._get_configuration_option(sub, OPTIONS, DummyConfiguration)


class MainConfiguration(Configuration):

    def __init__(self, mesh, root=None, **default):

        self._param = ngs.Parameter(2.0)

        DEFAULT = {
            "other": OptionA(mesh, root),
            "param": 2.0
        }
        DEFAULT.update(default)
        super().__init__(mesh, root, **DEFAULT)

    @dream_configuration
    def other(self) -> OptionConfiguration:
        return self._other

    @other.setter
    def other(self, other):
        OPTIONS = [OptionA, OptionB]
        self._other = self._get_configuration_option(other, OPTIONS, OptionConfiguration)

    @dream_configuration
    def param(self):
        return self._param

    @param.setter
    def param(self, param):
        self._param.Set(param)


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
        mesh = simplex()
        self.obj.p = 10
        self.assertEqual(self.obj.to_py(mesh)['pressure'], 10)

        self.obj.p = None
        self.assertEqual(self.obj.to_py(mesh)['pressure'], 10)

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
        mesh = simplex()
        self.obj.update({"rho": 2, "p": 5, "u": 5})

        self.assertDictEqual(self.obj.to_py(mesh), {"rho": 2, "p": 5, 'u': 5})

    def test_keys(self):
        self.obj.update({"rho": 2, "p": 5, "u": 5})
        self.assertTupleEqual(tuple(self.obj.keys()), ("rho", "p", 'u'))

    def test_values(self):
        mesh = simplex()
        self.obj.update({"rho": 2, "p": 5, "u": 5})
        self.assertTupleEqual(tuple(self.obj.to_py(mesh).values()), (2, 5, 5))

    def test_items(self):
        mesh = simplex()
        self.obj.update({"rho": 2, "p": 5, "u": 5})
        self.assertTupleEqual(tuple(self.obj.to_py(mesh).items()), (('rho', 2), ('p', 5), ('u', 5)))

    def test_set_item(self):
        mesh = simplex()
        self.obj["rho"] = 2
        self.obj["p"] = 5
        self.obj["u"] = 5

        self.assertDictEqual(self.obj.to_py(mesh), {"rho": 2, "p": 5, 'u': 5})

    def test_set_quantity(self):
        mesh = simplex()
        self.obj.rho = 2
        self.obj.p = 5
        self.obj["u"] = 5


        self.assertDictEqual(self.obj.to_py(mesh), {"density": 2, "pressure": 5, 'u': 5})


    def test_get_item(self):
        self.assertEqual(self.obj.p, None)

        with self.assertRaises(KeyError):
            self.obj["p"]

    def test_iterator(self):
        self.obj.update({"rho": 2, "p": 5, "u": 5})
        self.assertTupleEqual(tuple(self.obj), ("rho", "p", 'u'))

    def test_length(self):
        self.obj.update({"rho": 2, "p": 5, "u": 5})
        self.assertEqual(len(self.obj), 3)


class TestOptionConfiguration(unittest.TestCase):

    def setUp(self):
        self.obj = OptionA(None)

    def tearDown(self) -> None:
        self.obj.clear()

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
        self.obj.z = 2
        self.obj.x = 5

        self.assertDictEqual(self.obj.to_dict(), {'x': 5.0, 'y': 0.0})
        self.assertDictEqual(self.obj.__dict__, {'root': self.obj.root,
                             'mesh': None, '_x': 5.0, '_y': 0.0, 'z': 2})

        self.obj.clear()

        self.assertDictEqual(self.obj.to_dict(), {'x': 0.0, 'y': 0.0})
        self.assertDictEqual(self.obj.__dict__, {'root': self.obj.root,
                             'mesh': None, '_x': 0.0, '_y': 0.0, 'z': 2})

    def test_root_id(self):
        self.assertEqual(id(self.obj.root), id(self.obj))

    def test_export_default(self):
        self.assertDictEqual(self.obj.to_dict(), {'x': 0.0, 'y': 0.0})

    def test_export_parent_argument(self):
        self.assertDictEqual(self.obj.to_dict(root="parent"), {'parent.x': 0.0, 'parent.y': 0.0})

    def test_export_data_argument(self):
        self.assertDictEqual(self.obj.to_dict(dict={'z': 2}), {'x': 0.0, 'y': 0.0, 'z': 2})


class TestUniqueConfiguration(unittest.TestCase):

    def setUp(self):
        self.obj = MainConfiguration(None)

    def tearDown(self) -> None:
        self.obj.clear()

    def test_root_id(self):
        self.assertEqual(id(self.obj), id(self.obj.other.root))

    def test_export_default(self):
        self.assertDictEqual(self.obj.to_dict(), {'other': 'option_a',
                             'other.x': 0.0, 'other.y': 0.0, 'param': 2.0})

    def test_export_parent(self):
        self.assertDictEqual(
            self.obj.to_dict(root='unique'),
            {'unique.other.x': 0.0, 'unique.other.y': 0.0, 'unique.param': 2.0})

    def test_export_data_argument(self):
        self.assertDictEqual(
            self.obj.to_dict(dict={'z': 2}),
            {'other.x': 0.0, 'other.y': 0.0, 'param': 2.0, 'z': 2})

    def test_update_subconfiguration(self):
        self.obj.update({'other.x': 3.0, 'other.y': 5.0, 'param': 5})
        self.assertDictEqual(self.obj.to_dict(), {'other': 'option_a',
                             'other.x': 3.0, 'other.y': 5.0, 'param': 5.0})

    def test_export_parent(self):
        self.assertDictEqual(
            self.obj.to_dict(root='unique'),
            {'unique.other': 'option_a', 'unique.other.x': 0.0, 'unique.other.y': 0.0, 'unique.param': 2.0})

    def test_export_data_argument(self):
        self.assertDictEqual(
            self.obj.to_dict(dict={'z': 2}),
            {'other': 'option_a', 'other.x': 0.0, 'other.y': 0.0, 'param': 2.0, 'z': 2})

    def test_set_subconfiguration_by_string(self):
        self.obj.other = "option_b"
        self.assertDictEqual(
            self.obj.to_dict(),
            {'other': 'option_b', 'other.alphabet': 'abc', 'other.number': 2.0, 'other.sub': 'point',
             'other.sub.x': 0.0,
             'other.sub.y': 0.0, 'param': 2.0})

    def test_set_subconfiguration_by_instance(self):
        self.obj.other = OptionB(None)
        self.assertDictEqual(self.obj.to_dict(), {'other.alphabet': 'abc', 'other.number': 2.0, 'param': 2.0})

    def test_set_subconfiguration_by_dict(self):
        self.obj.other = {'x': 2.0, 'y': 3.0, 'z': 5}
        self.assertDictEqual(self.obj.to_dict(), {'other': 'option_a',
                             'other.x': 2.0, 'other.y': 3.0, 'param': 2.0})

    def test_set_subconfiguration_by_instance(self):
        old = id(self.obj.other)
        self.obj.other = OptionA(None)
        new = id(self.obj.other)
        self.assertNotEqual(old, new)

    def test_set_subconfiguration_by_dict(self):
        old = id(self.obj.other)
        self.obj.other.update({'x': 2.0, 'y': 3.0, 'z': 5})
        new = id(self.obj.other)
        self.assertEqual(old, new)

    def test_descriptor_set_recursive_update_subconfigurations(self):

        sub_config = OptionB(None)

        self.obj.other = sub_config

        self.assertEqual(id(self.obj), id(self.obj.other.root))


# %%
