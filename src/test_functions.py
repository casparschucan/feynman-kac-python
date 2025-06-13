import numpy as np

# The following provides a list of functions to test the poisson solver
# where we get [func_name] as boundary condition/analytic solutio
# and [func_name]_rhs as the right hand size for the PDE $-1/2 \Delta phi = g$


def non_hom_test(x, y):
    return np.sin(x)*np.sin(y)


def sin(x, y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)


def sin_rhs(x, y):
    return np.pi**2 * (np.sin(np.pi * x)*np.sin(np.pi * y))


def cos(x, y):
    return np.cos(np.pi*x)*np.cos(np.pi*y)


def cos_rhs(x, y):
    return (np.pi**2) * np.cos(np.pi*x)*np.cos(np.pi*y)


def sq_cos(x, y):
    return 4*(np.cos(np.pi*x)*np.cos(np.pi*y))**2


def sq_cos_rhs(x, y):
    return 4*0.5 * (np.pi**2) * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)
                                 + np.cos(2 * np.pi * (x - y))
                                 + np.cos(2 * np.pi * (x + y)))


def poly(x, y):
    return x**2 + y**2


def poly_rhs(x, y):
    return -2


def gaussian(x, y):
    mu = 0.01
    return np.exp(-((x-0.5)**2 + (y-0.5)**2)/mu)


def gaussian_rhs(x, y):
    mu = 0.01
    # Laplacian of exp(-((x-0.5)²+(y-0.5)²)/0.01)
    r2 = ((x-0.5)**2 + (y-0.5)**2)/mu
    return -1.*(2.*r2 - 2)*np.exp(-r2)/mu


def exp(x, y):
    return np.exp(x+y)


def exp_rhs(x, y):
    return -np.exp(x+y)  # Laplacian of exp(x+y)


def paper(x, y):
    if x < 0.25:
        return 4*(x - 0.25)**2
    if x > 0.75:
        return 4*(x - 0.75)**2
    return 0


def paper_rhs(x, y):
    return 0
