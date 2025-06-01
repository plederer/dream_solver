import pytest
import ngsolve as ngs
from netgen.occ import WorkPlane, OCCGeometry, Glue, IdentificationType
from dream.mesh import (
    DomainConditions,
    BoundaryConditions,
    BufferCoord,
    GridMapping,
    GridDeformation,
    Periodic,
    Condition,
    get_pattern_from_sequence,
    SpongeFunction,
    SpongeLayer
)


def unit_square(maxh=0.25, periodic: bool = False) -> ngs.Mesh:
    wp = WorkPlane()
    faces = []
    for i, x_ in enumerate([-0.375, -0.125, 0.125, 0.375]):
        face = wp.MoveTo(x_, 0).RectangleC(0.25, 1).Face()

        face.name = f"layer_{i}"

        for edge, bnd in zip(face.edges, ("bottom", "right", "top", "left")):
            edge.name = bnd

        if periodic:
            periodic_edge = face.edges[0]
            periodic_edge.Identify(face.edges[2], f"periodic_{i}", IdentificationType.PERIODIC)

        faces.append(face)

    face = Glue(faces)
    return ngs.Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=maxh))


def unit_circle(maxh=0.125, shift=(0, 0)) -> ngs.Mesh:
    wp = WorkPlane()
    faces = []
    for i, r_ in enumerate([0.125, 0.25, 0.375, 0.5]):
        face = wp.Circle(*shift, r_).Face()

        face.name = f"layer_{i}"
        faces.append(face)

    face = Glue(faces)
    return ngs.Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=maxh))


square = unit_square(0.1, periodic=True)
circle = unit_circle(0.1, shift=(1, 1))
circle.Curve(5)


@pytest.fixture
def bcs():
    geo = WorkPlane().LineTo(1, 0, "a").LineTo(1, 1, "a").LineTo(0, 1, "b").LineTo(0, 0, "c").Face()
    mesh = ngs.Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=1))
    return BoundaryConditions(mesh, options=[Periodic])


def test_label_uniqueness(bcs):
    assert tuple(bcs) == ("a", "b", "c")


def test_set_object(bcs):
    a = Condition()
    b = Periodic()

    bcs['d'] = a
    assert bcs.data == {'a': [], 'b': [], 'c': []}

    bcs['c'] = a
    assert bcs.data == {'a': [], 'b': [], 'c': [a]}
    bcs.clear()

    bcs['a|b'] = a
    bcs['c'] = b
    assert bcs.data == {"a": [a], "b": [a], "c": [b]}


def test_set_pattern(bcs):
    bcs['d'] = "periodic"
    assert bcs.data == {'a': [], 'b': [], 'c': []}

    bcs['c'] = "periodic"
    assert bcs.data == {'a': [], 'b': [], 'c': bcs['c']}
    bcs.clear()

    bcs['a|b'] = 'periodic'
    bcs['c'] = 'periodic'
    assert bcs.data == {"a": bcs['a'], "b": bcs['a'], "c": bcs['c']}

    assert isinstance(bcs['a'][0], Periodic)
    assert isinstance(bcs['c'][0], Periodic)


def test_get_boundaries(bcs):
    bcs['a'] = Periodic()
    bcs['b'] = Condition()
    bcs['c'] = Condition()

    assert bcs.get_region(Periodic) == ['a']
    assert bcs.get_region(Periodic, as_pattern=True) == 'a'

    assert bcs.get_region(Periodic, Condition) == ['a', 'b', 'c']
    assert bcs.get_region(Periodic, Condition, as_pattern=True) == 'a|b|c'


def test_get_domain_boundaries(bcs):
    bcs['a'] = Periodic()
    bcs['b'] = Condition()

    assert bcs.get_domain_boundaries() == ['b']
    assert bcs.get_domain_boundaries(as_pattern=True) == 'b'

    bcs.clear()

    bcs['a'] = Periodic()
    bcs['b'] = Condition()
    bcs['c'] = Condition()

    assert bcs.get_domain_boundaries() == ['b', 'c']
    assert bcs.get_domain_boundaries(as_pattern=True) == 'b|c'


@pytest.fixture
def dcs():
    return DomainConditions(square, options=[SpongeLayer, GridDeformation])


@pytest.fixture
def pcs():
    return DomainConditions(circle, options=[SpongeLayer, GridDeformation])


def test_domains(dcs):
    assert tuple(sorted(dcs)) == ("layer_0", "layer_1", "layer_2", "layer_3")


def test_dim(dcs):
    assert dcs.mesh.dim == 2


def test_pattern():
    assert get_pattern_from_sequence(["left", "right"]) == "left|right"
    assert get_pattern_from_sequence("left") == "left"


def test_cartesian_grid_deformation(dcs):
    x = BufferCoord.x(0.25, 0.5)
    x = GridMapping.linear(5, x)

    dom = {"layer_3": GridDeformation(x=x, order=1)}
    gfu = dcs.get_grid_deformation_function(dom)

    assert isinstance(gfu, ngs.GridFunction)
    assert ngs.Integrate(ngs.Norm(gfu), dcs.mesh, order=1) == pytest.approx(0.125)


def test_cartesian_sponge_layer(dcs):
    dom = {}
    for i, x_ in enumerate([-0.25, 0, 0.25]):
        x = BufferCoord.x(x_, x_ + 0.25)
        sigma = SpongeFunction.polynomial(2, x, order=5)
        dom[f"layer_{i + 1}"] = SpongeLayer(function=sigma, target_state={"rho": 1}, order=5)

    gfu = dcs.get_sponge_layer_function(dom)

    assert isinstance(gfu, ngs.GridFunction)
    assert ngs.Integrate(ngs.Norm(gfu), dcs.mesh, order=5) == pytest.approx(0.25)


def test_polar_grid_deformation(pcs):
    r = BufferCoord.polar(0.125, 0.5, shift=(1, 1))
    map = GridMapping.linear(5, r)
    x, y = map.polar_to_cartesian()

    d = GridDeformation(x=x, y=y, order=5)
    dom = {"layer_1": d, "layer_2": d, "layer_3": d}

    gfu = pcs.get_grid_deformation_function(dom)

    assert isinstance(gfu, ngs.GridFunction)
    assert ngs.Integrate(ngs.Norm(gfu), pcs.mesh, order=5) == pytest.approx(0.6626797003665968)


def test_polar_sponge_layer(pcs):
    dom = {}
    for i, r_ in enumerate([0.125, 0.25, 0.375]):
        r = BufferCoord.polar(r_, r_ + 0.125, shift=(1, 1))
        sigma = SpongeFunction.polynomial(2, r, order=5)
        dom[f"layer_{i + 1}"] = SpongeLayer(function=sigma, target_state={"rho": 1}, order=5)

    gfu = pcs.get_sponge_layer_function(dom)

    assert isinstance(gfu, ngs.GridFunction)
    result = 2/3 * ngs.pi * 0.125 * (0.5 + 0.375 + 0.25 - 3/7 * 0.125)
    assert ngs.Integrate(ngs.Norm(gfu), pcs.mesh, order=5) == pytest.approx(result)


@pytest.fixture
def cmesh():
    return unit_square()


@pytest.fixture
def pmesh():
    mesh = unit_circle(shift=(1, 1))
    mesh.Curve(6)
    return mesh


def test_buffercoord_instance():
    x = BufferCoord.x(-0.25, 0.25)
    assert isinstance(x, ngs.CF)


def test_buffercoord_x(cmesh):
    x = BufferCoord.x(-0.25, 0.25)
    x_ = x.get_normalised_coordinate()

    assert x.length == pytest.approx(0.5)
    assert ngs.Integrate(x, cmesh) == pytest.approx(0)
    assert ngs.Integrate(x_, cmesh) == pytest.approx(0.5)


def test_buffercoord_y(cmesh):
    y = BufferCoord.y(-0.5, 0.5)
    y_ = y.get_normalised_coordinate()

    assert y.length == pytest.approx(1)
    assert ngs.Integrate(y, cmesh) == pytest.approx(0)
    assert ngs.Integrate(y_, cmesh) == pytest.approx(0.5)


def test_buffercoord_r(pmesh):
    r = BufferCoord.polar(0.125, 0.375, shift=(1, 1))
    r_ = r.get_normalised_coordinate()

    assert r.length == pytest.approx(0.25)
    assert ngs.Integrate(r, pmesh) == pytest.approx(0.24134631062734085)
    assert ngs.Integrate(r_, pmesh) == pytest.approx(0.5726861608106394)


@pytest.fixture
def gridmap_r():
    return BufferCoord.polar(0.125, 0.5, shift=(1, 1))


@pytest.fixture
def gridmap_pmesh():
    return unit_circle(shift=(1, 1))


def test_gridmapping_none(gridmap_r):
    map = GridMapping.none(gridmap_r)
    assert map.length == pytest.approx(0.375)
    assert map(0.5) == pytest.approx(0.5)


def test_gridmapping_linear(gridmap_r):
    map = GridMapping.linear(5, gridmap_r)
    assert map.length == pytest.approx(1.875)
    assert map(0.5) == pytest.approx(2.0)


def test_gridmapping_exponential(gridmap_r):
    map = GridMapping.exponential(5, gridmap_r)
    assert map.length == pytest.approx(1.875)
    assert map(0.5) == pytest.approx(2.0)


def test_gridmapping_tangential(gridmap_r):
    map = GridMapping.tangential(5, gridmap_r)
    assert map.length == pytest.approx(1.875)
    assert map(0.5) == pytest.approx(2.0)
