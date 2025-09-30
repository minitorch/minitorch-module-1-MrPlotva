"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Mul"""
    return x * y


def id(x: float) -> float:
    """Id"""
    return x


def add(x: float, y: float) -> float:
    """Add"""
    return x + y


def neg(x: float) -> float:
    """Neg"""
    return -x


def lt(x: float, y: float) -> bool:
    """Lt"""
    return x < y


def eq(x: float, y: float) -> bool:
    """Eq"""
    return x == y


def max(x: float, y: float) -> float:
    """Max"""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """is_close"""
    return abs(x - y) < 1e-3


def sigmoid(x: float) -> float:
    """Sigmoid"""
    return 1.0 / (1.0 + math.exp(-x))


def relu(x: float) -> float:
    """Relu"""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Log"""
    return math.log(x)


def exp(x: float) -> float:
    """Exp"""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """log_back"""
    return d / x


def inv(x: float) -> float:
    """Inv"""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """inv_back"""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """relu_back"""
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Map"""

    def _map(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """ZipWith"""

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduce"""

    def _reduce(ls: Iterable[float]) -> float:
        result = start
        for x in ls:
            result = fn(result, x)
        return result

    return _reduce


def negList(ls: Iterable[float]) -> Iterable[float]:
    """NegList"""
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """AddLists"""
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum"""
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Prod"""
    return reduce(mul, 1.0)(ls)
