import sympy
import numpy as np

# calculation of the Jacobian matrix
from scipy import linalg


def form_jac_matrix(eqs, vars):
    n = len(eqs)
    m = len(vars)
    Jacobian = np.zeros((n, m))
    for i in range(0, n):
        eq = eqs[i]
        for j in range(0, m):
            var = vars[j]
            Jacobian[i, j] = sympy.N(eq.diff(var))
    return Jacobian


def form_symbolic_jac_matrix(eqs, statevars):
    n = len(statevars)
    Jacobian = sympy.zeros(n, n)
    for i in range(0, n):
        eq = eqs[i]
        for j in range(0, n):
            var = statevars[j]
            Jacobian[i, j] = eq.diff(var)
    return Jacobian


# Analytical solution from 0 to tf with a timestep of h
def analytical_solution_matrix(A, x0, h, tf):
    # Computes solution to dx/dt = A*x
    t_range = np.arange(0, tf + h, h)
    exphA = linalg.expm(h * A)
    x = x0
    xs = [x]
    for _ in t_range[1:]:
        x = exphA.dot(x)
        xs.append(x)
    return (t_range, xs)


def convert_autonomous_ode(A, B, x0):
    n = np.shape(A)[0]
    x0 = np.concatenate((x0,
                         np.array([[1]])), axis=0)  # vertical
    assert np.shape(x0) == (n + 1, 1)
    A_row_1 = np.concatenate((A, B), axis=1)

    # Size of input
    m = B.shape[1]
    assert A_row_1.shape == (n, n + m)

    A_row_2 = np.concatenate((np.zeros((1, n)), np.zeros((1, 1))), axis=1)

    assert A_row_2.shape == (1, n + 1)

    A_expanded = np.concatenate((A_row_1,
                                 A_row_2), axis=0)

    assert A_expanded.shape == (n + 1, n + 1)

    return (A_expanded, x0)


def analytical_solution_matrix_affine(A, B, x0, h, tf):
    # Computes solution to dx/dt = A.dot(x) + B
    n = np.shape(A)[0]
    t_range = np.arange(0, tf + h, h)
    assert len(t_range) > 1, "At least one numerical step should be done."
    (A_expanded, x) = convert_autonomous_ode(A, B, x0)
    exphA = linalg.expm(h * A_expanded)
    xs = [x]
    for _ in t_range[1:]:
        x = exphA.dot(x)
        xs.append(x)
    return (t_range, [x[0:n] for x in xs])
