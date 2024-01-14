from __future__ import annotations
import unittest

import dream.config as cfg
import ngsolve as ngs


class Descriptor(unittest.TestCase):

    def setUp(self):

        class Test:
            test = cfg.descriptor()

        self.obj = Test()

    def test_descriptor_public_label(self):
        self.assertEqual(type(self.obj).test.label, "test")


class Variable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
    
        class Test(cfg.DescriptorDict):
            double = cfg.variable(lambda x: 2*x)
        
        cls.class_ = Test

    def setUp(self):
        self.obj = self.class_()

    def tearDown(self) -> None:
        self.obj.clear()

    def test_label(self):
        self.obj.double = 10
        self.assertIn("double", self.obj)

    def test_getter(self):
        self.assertEqual(self.obj.double, None)

    def test_setter(self):
        self.obj.double = 10
        self.assertEqual(self.obj.double, 20)

        self.obj.double = None
        self.assertEqual(self.obj.double, 20)

    def test_deleter(self):
        self.obj.double = 20
        del self.obj.double
        self.assertEqual(self.obj.double, None)


class DescriptorDict(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        class Test(cfg.DescriptorDict):
            double = cfg.variable(lambda x: 2*x)
            join = cfg.variable(lambda x: "".join(x))

        self.class_ = Test

    def setUp(self) -> None:
        self.obj = self.class_()

    def tearDown(self) -> None:
        self.obj.clear()

    def test_update(self):
        kwargs = {"double": 2, "join": ['a', 'b', 'c'], "non_default": 2}
        self.obj.update(**kwargs)

        self.assertDictEqual(self.obj.data, {"double": 4, "join": "abc"})
        self.assertDictEqual(self.obj.__dict__, {"data": self.obj.data, "non_default": 2})

    def test_keys(self):
        self.obj.update(double=2, join=['a', 'b', 'c'], c=3)
        self.assertTupleEqual(tuple(self.obj.keys()), ("double", "join"))

    def test_values(self):
        self.obj.update(double=2, join=['a', 'b', 'c'], c=3)
        self.assertTupleEqual(tuple(self.obj.values()), (4, 'abc'))

    def test_items(self):
        self.obj.update(double=2, join="abc", c=3)
        self.assertTupleEqual(tuple(self.obj.items()), (('double', 4), ('join', 'abc')))

    def test_set_item(self):
        self.obj["double"] = 2
        self.obj["join"] = ['a', 'b', 'c']
        self.obj["non_default"] = 2

        self.assertDictEqual(self.obj.data, {"double": 4, "join": "abc"})
        self.assertDictEqual(self.obj.__dict__, {"data": self.obj.data, "non_default": 2})

    def test_get_item(self):
        self.assertEqual(self.obj["double"], None)

        with self.assertRaises(AttributeError):
            self.obj["non_default"]

    def test_iterator(self):
        self.obj.update(double=2, join="abc", c=3)
        self.assertTupleEqual(tuple(self.obj), ("double", "join"))

    def test_length(self):
        self.obj.update(double=2, join="abc", c=3)
        self.assertEqual(len(self.obj), 2)


class UserConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        class SubTest(cfg.UserConfig, is_interface=True):
            ...

        class Test_A(SubTest):

            label: str = "A"

            @cfg.cfg(default="abc")
            def sub_a(self, c: str):
                return str(c)

        class Test_B(SubTest):

            @cfg.cfg(default=0.3)
            def sub_b(self, b):
                return float(b)

        class Test(cfg.UserConfig, is_interface=True):

            @cfg.cfg(default=2)
            def a(self, a: int):
                """ Polynomial Order """
                if a < 0:
                    raise ValueError("Order must be greater > 0")
                return int(a)

            @cfg.parameter(default=2)
            def b(self, b):
                """ Heat Capacity Ratio """
                return b

            @b.get_check
            def b(self):
                if self.a > 10:
                    raise ValueError("Polynomial Order to high!")

            @cfg.user(default=Test_A)
            def c(self, cfg: SubTest):
                return cfg
            
            c: SubTest

        cls.subclass_ = SubTest
        cls.subclass_A = Test_A
        cls.subclass_B = Test_B
        cls.class_ = Test

    def setUp(self):
        self.obj = self.class_()

    def tearDown(self) -> None:
        self.obj.clear()

    def test_types(self):
        self.assertDictEqual(self.obj.types.data, {})
        self.assertDictEqual(self.obj.c.types.data, {'A': self.subclass_A, 'Test_B': self.subclass_B})

    def test_nested_configuration_string_constructor(self):
        self.obj.c = "A"
        self.assertDictEqual(self.obj.c.data, {'sub_a': 'abc'})

        self.obj.c = "Test_B"
        self.assertDictEqual(self.obj.c.data, {'sub_b': 0.3})

    def test_nested_configuration_dict_constructor(self):
        self.obj.c = {'type': 'A', 'sub_a': 'art', '_private': 2}
        self.assertDictEqual(self.obj.c.data, {'sub_a': 'art'})

        self.obj.c = {'type': 'Test_B', 'sub_b': 0.1, '_private': 2}
        self.assertDictEqual(self.obj.c.data, {'sub_b': 0.1})

    def test_flatten(self):
        print(self.obj.flatten())
        self.assertDictEqual(self.obj.flatten(), {'type': 'Test', 'a': 2, 'b': self.obj.b, 'c': {'type': 'A', 'sub_a': 'abc'}})

    def test_clear_id(self):
        old = [self.obj.b, self.obj.c]

        self.obj.clear()
        new = [self.obj.b, self.obj.c]

        for new, old in zip(old, new):
            self.assertNotEqual(id(new), id(old))

    def test_update_id(self):
        old = [self.obj.b, self.obj.c]

        self.obj.update(**{'a': 2, 'b': 0.3, 'c': {'type': 'A', '_sub_b': 3}})
        new = [self.obj.b, self.obj.c]

        for new, old in zip(old, new):
            self.assertEqual(id(new), id(old))

    def test_clear(self):
        self.obj["test"] = 2
        self.obj.clear()

        self.assertListEqual(list(self.obj.keys()), ['a', 'b', 'c'])
