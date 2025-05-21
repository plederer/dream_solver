import pytest
import ngsolve as ngs
from dream.config import (ngsdict,
                          quantity,
                          Configuration,
                          dream_configuration)

mesh = ngs.Mesh(ngs.unit_square.GenerateMesh(maxh=1))


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


@pytest.fixture
def state():
    obj = teststate()
    yield obj
    obj.clear()


def test_variable_name(state):
    assert type(state).rho.name == "density"


def test_variable_symbol(state):
    assert type(state).rho.symbol == "rho"


def test_label(state):
    state.p = 10
    assert "pressure" in state


def test_getter(state):
    assert state.p is None


def test_setter(state):
    state.p = 10
    assert state.to_py(mesh)['pressure'] == 10
    state.p = None
    assert state.to_py(mesh)['pressure'] == 10


def test_deleter(state):
    state.p = 20
    del state.p
    assert state.p is None


def test_update(state):
    state.update({"rho": 2, "p": 5, "u": 5})
    assert state.to_py(mesh) == {"rho": 2, "p": 5, 'u': 5}


def test_keys(state):
    state.update({"rho": 2, "p": 5, "u": 5})
    assert tuple(state.keys()) == ("rho", "p", 'u')


def test_values(state):
    state.update({"rho": 2, "p": 5, "u": 5})
    assert tuple(state.to_py(mesh).values()) == (2, 5, 5)


def test_items(state):
    state.update({"rho": 2, "p": 5, "u": 5})
    assert tuple(state.to_py(mesh).items()) == (('rho', 2), ('p', 5), ('u', 5))


def test_set_item(state):
    state["rho"] = 2
    state["p"] = 5
    state["u"] = 5
    assert state.to_py(mesh) == {"rho": 2, "p": 5, 'u': 5}


def test_set_quantity(state):
    state.rho = 2
    state.p = 5
    state["u"] = 5
    assert state.to_py(mesh) == {"density": 2, "pressure": 5, 'u': 5}


def test_get_item(state):
    assert state.p is None
    with pytest.raises(KeyError):
        _ = state["p"]


def test_iterator(state):
    state.update({"rho": 2, "p": 5, "u": 5})
    assert tuple(state) == ("rho", "p", 'u')


def test_length(state):
    state.update({"rho": 2, "p": 5, "u": 5})
    assert len(state) == 3


@pytest.fixture
def option_a():
    obj = OptionA(None)
    yield obj
    obj.clear()


def test_configurations_id_after_clear(option_a):
    old = [option_a.x, option_a.y]
    option_a.clear()
    new = [option_a.x, option_a.y]
    for n, o in zip(new, old):
        assert id(n) == id(o)


def test_configurations_id_after_update(option_a):
    old = [option_a.x, option_a.y]
    option_a.update({'x': 2, 'y': 0.3})
    new = [option_a.x, option_a.y]
    for n, o in zip(new, old):
        assert id(n) != id(o)


def test_clear(option_a):
    option_a.z = 2
    option_a.x = 5
    assert option_a.to_dict() == {'x': 5.0, 'y': 0.0}
    assert option_a.__dict__ == {'root': option_a.root, 'mesh': None, '_x': 5.0, '_y': 0.0, 'z': 2}
    option_a.clear()
    assert option_a.to_dict() == {'x': 0.0, 'y': 0.0}
    assert option_a.__dict__ == {'root': option_a.root, 'mesh': None, '_x': 0.0, '_y': 0.0, 'z': 2}


def test_root_id(option_a):
    assert id(option_a.root) == id(option_a)


def test_export_default(option_a):
    assert option_a.to_dict() == {'x': 0.0, 'y': 0.0}


def test_export_parent_argument(option_a):
    assert option_a.to_dict(root="parent") == {'parent.x': 0.0, 'parent.y': 0.0}


def test_export_data_argument(option_a):
    assert option_a.to_dict(dict={'z': 2}) == {'x': 0.0, 'y': 0.0, 'z': 2}


@pytest.fixture
def main_config():
    obj = MainConfiguration(None)
    yield obj
    obj.clear()


def test_main_root_id(main_config):
    assert id(main_config) == id(main_config.other.root)


def test_main_export_default(main_config):
    assert main_config.to_dict() == {'other': 'option_a', 'other.x': 0.0, 'other.y': 0.0, 'param': 2.0}


def test_main_export_parent(main_config):
    assert main_config.to_dict(root='unique') == {'unique.other': 'option_a',
                                                  'unique.other.x': 0.0, 'unique.other.y': 0.0, 'unique.param': 2.0}


def test_main_export_data_argument(main_config):
    assert main_config.to_dict(dict={'z': 2}) == {'other': 'option_a',
                                                  'other.x': 0.0, 'other.y': 0.0, 'param': 2.0, 'z': 2}


def test_update_subconfiguration(main_config):
    main_config.update({'other.x': 3.0, 'other.y': 5.0, 'param': 5})
    assert main_config.to_dict() == {'other': 'option_a', 'other.x': 3.0, 'other.y': 5.0, 'param': 5.0}


def test_main_export_parent(main_config):
    assert main_config.to_dict(root='unique') == {'unique.other': 'option_a',
                                                  'unique.other.x': 0.0, 'unique.other.y': 0.0, 'unique.param': 2.0}


def test_main_export_data_argument(main_config):
    assert main_config.to_dict(dict={'z': 2}) == {'other': 'option_a',
                                                  'other.x': 0.0, 'other.y': 0.0, 'param': 2.0, 'z': 2}


def test_set_subconfiguration_by_string(main_config):
    main_config.other = "option_b"
    assert main_config.to_dict() == {
        'other': 'option_b', 'other.alphabet': 'abc', 'other.number': 2.0, 'other.sub': 'point',
        'other.sub.x': 0.0, 'other.sub.y': 0.0, 'param': 2.0
    }


def test_set_subconfiguration_by_instance(main_config):
    main_config.other = OptionB(None)
    assert main_config.to_dict() == {'other': 'option_b', 'other.alphabet': 'abc', 'other.number': 2.0,
                                     'other.sub': 'point', 'other.sub.x': 0.0, 'other.sub.y': 0.0,
                                     'param': 2.0}


def test_set_subconfiguration_by_dict(main_config):
    old = id(main_config.other)
    main_config.other.update({'x': 2.0, 'y': 3.0, 'z': 5})
    new = id(main_config.other)
    assert old == new


def test_set_subconfiguration_by_instance_id(main_config):
    old = id(main_config.other)
    main_config.other = OptionA(None)
    new = id(main_config.other)
    assert old != new


def test_descriptor_set_recursive_update_subconfigurations(main_config):
    sub_config = OptionB(None)
    main_config.other = sub_config
    assert id(main_config) == id(main_config.other.root)
