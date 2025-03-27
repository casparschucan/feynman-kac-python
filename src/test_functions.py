import numpy as np


def non_hom_test(x, y):
    return np.sin(x)*np.sin(y)


def test_phi(x, y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)


def test_bound(x, y):
    return 0


def test_rhs(x, y):
    return np.pi**2 * (np.sin(np.pi * x)*np.sin(np.pi * y))


def test_cos(x, y):
    return np.cos(np.pi*x)*np.cos(np.pi*y)


def test_cos_rhs(x, y):
    return (np.pi**2) * np.cos(np.pi*x)*np.cos(np.pi*y)
