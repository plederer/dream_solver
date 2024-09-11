from __future__ import annotations
import ngsolve as ngs
import builtins as math
import logging

from numbers import Number
from typing import Sequence, Generator, TypeAlias

logger = logging.getLogger(__name__)

SCALAR: TypeAlias = ngs.CF | Number
VECTOR: TypeAlias = ngs.CF
MATRIX: TypeAlias = ngs.CF


def abs(x: SCALAR) -> SCALAR:
    if isinstance(x, ngs.CF):
        return ngs.IfPos(x, x, -x)
    elif isinstance(x, Number):
        return math.abs(x)
    else:
        raise NotImplementedError(f"Can not calculate absolute value for '{type(x)}'!")


def max(x: SCALAR, y: SCALAR = 0) -> SCALAR:

    if isinstance(x, ngs.CF) or isinstance(y, ngs.CF):
        return ngs.IfPos(x-y, x, y)
    elif isinstance(x, Number) and isinstance(y, Number):
        return math.max(x, y)
    else:
        raise NotImplementedError(f"Can not calculate maximum value for '{type(x)}' and '{type(x)}'!")


def min(x: SCALAR, y: SCALAR = 0) -> SCALAR:

    if isinstance(x, ngs.CF) or isinstance(y, ngs.CF):
        return ngs.IfPos(x-y, y, x)
    elif isinstance(x, Number) and isinstance(y, Number):
        return math.min(x, y)
    else:
        raise NotImplementedError(f"Can not calculate minimum value for '{type(x)}' and '{type(x)}'!")


def interval(x: SCALAR, start: SCALAR, end: SCALAR):
    return min(max(x, start), end)


def trace(x: MATRIX) -> SCALAR:

    if is_symmetric(x):
        return sum(x[i, i] for i in range(x.dims[0]))
    else:
        raise ValueError(f"Can not return trace for non symmetric matrix!")


def diagonal(x: VECTOR) -> MATRIX:
    x = as_vector(x)

    mat = [0 for _ in range(x.dim**2)]
    for i in range(x.dim):
        mat[(x.dim + 1) * i] = x[i]

    return as_matrix(mat, dims=(x.dim, x.dim))


def unit_vector(x: ngs.CF | Sequence | Generator) -> VECTOR:
    x = as_vector(x)
    return x/ngs.sqrt(inner(x, x))


def inner(x: VECTOR | MATRIX, y: VECTOR | MATRIX) -> SCALAR:
    return ngs.InnerProduct(x, y)


def outer(x: VECTOR, y: VECTOR) -> MATRIX:
    x = as_vector(x)
    y = as_vector(y)
    return ngs.OuterProduct(x, y)


def as_scalar(x: SCALAR) -> SCALAR:
    if isinstance(x, ngs.CF):
        return x
    elif isinstance(x, Number):
        return ngs.CF(x)
    else:
        raise NotImplementedError(f"Can not cast type '{type(x)}' to a scalar!")


def is_scalar(x: SCALAR) -> bool:
    if isinstance(x, Number):
        return True
    elif isinstance(x, ngs.CF):
        if not x.dims:
            return True
    return False


def as_vector(x: ngs.CF | Sequence | Generator) -> VECTOR:
    if isinstance(x, ngs.CF):
        return x
    elif isinstance(x, (Sequence, Generator)):
        return ngs.CF(tuple(x))
    else:
        raise NotImplementedError(f"Can not cast type '{type(x)}' to a vector!")


def is_vector(x: VECTOR) -> bool:
    if isinstance(x, ngs.CF):
        if len(x.dims) == 1:
            return True
    return False


def as_matrix(x: ngs.CF | Sequence | Generator, dims: tuple[int, ...] | None = None) -> MATRIX:

    if isinstance(x, ngs.CF):
        return x

    elif isinstance(x, (Sequence, Generator)):
        x = tuple(x)

        if dims is None:
            dim = len(x)
            row = int(ngs.sqrt(dim))

            if row**2 != dim:
                raise ValueError("Non symmetric matrix! Can not deduce dims!")

            dims = (row, row)

        return ngs.CF(x, dims=dims)
    else:
        raise NotImplementedError(f"Can not cast type '{type(x)}' to a matrix!")


def is_matrix(x: MATRIX):
    if isinstance(x, ngs.CF):
        if len(x.dims) == 2:
            return True
    return False


def is_symmetric(x: MATRIX):
    if is_matrix(x):
        return x.dims[0] == x.dims[1]
    else:
        return False


def is_zero(x: SCALAR | VECTOR | MATRIX) -> bool:
    if isinstance(x, Number):
        return x == 0.0
    elif isinstance(x, ngs.CF):
        return str(x) == 'ZeroCoefficientFunction'
    else:
        raise TypeError("Can not determine if it is zero!")


def fixpoint_iteration(x0: float, func, it: int = 100, tol: float = 1e-16):

    for i in range(it):

        xn = func(x0)
        err = math.abs(xn - x0)

        logger.debug(f"It: {i:3d} - n+1: {xn:.5e} - n: {x0:.5e} - err: {err:.5e}")

        x0 = xn

        if err < tol:
            break

    return x0


def symmetric_matrix_from_vector(x: ngs.CF, factor: float = 1):

    x = as_vector(x)

    if x.dim == 3:
        return factor * ngs.CF((x[0], x[1], x[1], x[2]), dims=(2, 2))
    elif x.dim == 6:
        return factor * ngs.CF((x[0], x[1], x[2], x[1], x[3], x[4], x[2], x[4], x[5]), dims=(3, 3))
    else:
        raise ValueError(f"Can not create symmetric matrix with component vector of length {x.dim}!")


def skewsymmetric_matrix_from_vector(x: ngs.CF, factor: float = 1):

    if is_scalar(x):
        return factor * ngs.CF((0, -x, x, 0), dims=(2, 2))

    x = as_vector(x)

    if x.dim == 3:
        return factor * ngs.CF((0, -x[0], x[1], x[0], 0, -x[2], -x[1], x[2], 0), dims=(3, 3))
    else:
        raise ValueError(f"Can not create symmetric matrix with component vector of length {x.dim}!")
