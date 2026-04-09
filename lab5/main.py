
import math
from typing import Callable, List, Tuple

import numpy as np

# -----------------------------
# Problem definition
# -----------------------------
# Minimize:
#   f(x) = 4 x1^2 + 2 x1 x2 + 3 x2^2 + exp(x3) + exp(x4)
#
# Subject to:
#   g1(x) = x1^2 + x2^2 + x3^2 + x4^2 - 4 <= 0
#   g2(x) = x1 + 2 x2 + x3 - 1 <= 0
#   g3(x) = -x1 + x4 - 1 <= 0
#   h1(x) = x2 - x3 + x4 = 0
#
# The feasible point x0 = (0, 0, 0, 0) is admissible.


def f(x: np.ndarray) -> float:
    x1, x2, x3, x4 = x
    return 4 * x1 * x1 + 2 * x1 * x2 + 3 * x2 * x2 + math.exp(x3) + math.exp(x4)


def grad_f(x: np.ndarray) -> np.ndarray:
    x1, x2, x3, x4 = x
    return np.array([8 * x1 + 2 * x2, 2 * x1 + 6 * x2, math.exp(x3), math.exp(x4)], dtype=float)


def g1(x: np.ndarray) -> float:
    return float(np.dot(x, x) - 4.0)


def grad_g1(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.array(x, dtype=float)


def g2(x: np.ndarray) -> float:
    x1, x2, x3, x4 = x
    return float(x1 + 2 * x2 + x3 - 1.0)


def grad_g2(x: np.ndarray) -> np.ndarray:
    return np.array([1.0, 2.0, 1.0, 0.0], dtype=float)


def g3(x: np.ndarray) -> float:
    x1, x2, x3, x4 = x
    return float(-x1 + x4 - 1.0)


def grad_g3(x: np.ndarray) -> np.ndarray:
    return np.array([-1.0, 0.0, 0.0, 1.0], dtype=float)


def h1(x: np.ndarray) -> float:
    x1, x2, x3, x4 = x
    return float(x2 - x3 + x4)


def grad_h1(x: np.ndarray) -> np.ndarray:
    return np.array([0.0, 1.0, -1.0, 1.0], dtype=float)


def feasible(x: np.ndarray, tol: float = 1e-10) -> bool:
    return (g1(x) <= tol) and (g2(x) <= tol) and (g3(x) <= tol) and (abs(h1(x)) <= tol)


# -----------------------------
# Simplex method
# -----------------------------
# Solve:
#   max c^T z
#   A z <= b
#   z >= 0
#
# Converts to standard form with slack variables and applies primal simplex
# with Bland's rule (to avoid cycling).


def simplex_max(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> Tuple[np.ndarray, float, int]:
    
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    m, n = A.shape

    # Make right-hand sides nonnegative.
    for i in range(m):
        if b[i] < 0:
            A[i, :] *= -1
            b[i] *= -1

    # Tableau: [A | I | b], last row is objective z - c^T x = 0
    T = np.zeros((m + 1, n + m + 1), dtype=float)
    T[:m, :n] = A
    T[:m, n:n + m] = np.eye(m)
    T[:m, -1] = b
    T[m, :n] = -c

    basis = [n + i for i in range(m)]

    for _ in range(max_iter):
        reduced = T[m, :-1]
        entering_candidates = np.where(reduced < -tol)[0]

        if len(entering_candidates) == 0:
            sol = np.zeros(n + m, dtype=float)
            for i, bi in enumerate(basis):
                sol[bi] = T[i, -1]
            return sol[:n], float(T[m, -1]), 0

        entering = int(entering_candidates[0])  # Bland's rule

        column = T[:m, entering]
        leaving_candidates = np.where(column > tol)[0]
        if len(leaving_candidates) == 0:
            raise RuntimeError("Auxiliary LP is unbounded.")

        ratios = T[leaving_candidates, -1] / column[leaving_candidates]
        min_ratio = np.min(ratios)
        tied = leaving_candidates[np.where(np.abs(ratios - min_ratio) <= 1e-12)[0]]
        leaving = int(tied[0])  # Bland's rule

        piv = T[leaving, entering]
        T[leaving, :] /= piv
        for r in range(m + 1):
            if r != leaving:
                T[r, :] -= T[r, entering] * T[leaving, :]

        basis[leaving] = entering

    raise RuntimeError("Simplex iteration limit exceeded.")


# -----------------------------
# Auxiliary LP for Zoutendijk
# -----------------------------
# Variables:
#   z = (d+_1,...,d+_4, d-_1,...,d-_4, gamma), all >= 0
#   d = d+ - d-
#
# LP:
#   max gamma
#   s.t. grad f(x)^T d + gamma <= 0
#        grad g_j(x)^T d + gamma <= 0   for active inequalities
#        grad h1(x)^T d = 0
#        |d_i| <= 1
#
# Equality is written as two inequalities without gamma.


def auxiliary_direction(x: np.ndarray, active_tol: float = 1e-8) -> Tuple[np.ndarray, float]:
    gf = grad_f(x)

    A: List[np.ndarray] = []
    b: List[float] = []

    # grad f(x)^T (d+ - d-) + gamma <= 0
    row = np.zeros(9, dtype=float)
    row[:4] = gf
    row[4:8] = -gf
    row[8] = 1.0
    A.append(row)
    b.append(0.0)

    # Only the nonlinear inequality g1 is nonlinear.
    # Add it to the LP only if active.
    if g1(x) >= -active_tol:
        gg = grad_g1(x)
        row = np.zeros(9, dtype=float)
        row[:4] = gg
        row[4:8] = -gg
        row[8] = 1.0
        A.append(row)
        b.append(0.0)

    # Equality constraint grad h1(x)^T d = 0
    gh = grad_h1(x)
    row = np.zeros(9, dtype=float)
    row[:4] = gh
    row[4:8] = -gh
    A.append(row)
    b.append(0.0)

    row = np.zeros(9, dtype=float)
    row[:4] = -gh
    row[4:8] = gh
    A.append(row)
    b.append(0.0)

    # Bounds |d_i| <= 1
    for i in range(4):
        row = np.zeros(9, dtype=float)
        row[i] = 1.0
        row[4 + i] = 1.0
        A.append(row)
        b.append(1.0)

    c = np.zeros(9, dtype=float)
    c[8] = 1.0

    sol, gamma, _ = simplex_max(np.array(A), np.array(b), c)
    d = sol[:4] - sol[4:8]
    return d, gamma


# -----------------------------
# Step length
# -----------------------------


def max_feasible_alpha(x: np.ndarray, d: np.ndarray) -> float:
    """
    Maximum feasible alpha along x + alpha d.
    The feasible set is convex in this example, so we can find alpha_max
    by expanding and then bisection.
    """
    lo = 0.0
    hi = 1.0

    # Expand until infeasible
    for _ in range(60):
        if feasible(x + hi * d, tol=1e-12):
            lo = hi
            hi *= 2.0
        else:
            break
    else:
        return lo

    # Bisection on [lo, hi]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if feasible(x + mid * d, tol=1e-12):
            lo = mid
        else:
            hi = mid

    return lo


def golden_section_min(phi: Callable[[float], float], a: float, b: float, tol: float = 1e-10) -> float:
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = phi(c)
    fd = phi(d)

    while abs(b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = phi(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = phi(d)

    return 0.5 * (a + b)


# -----------------------------
# Zoutendijk method
# -----------------------------


def zoutendijk(
    x0: np.ndarray,
    eps: float,
    max_iter: int = 100,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, float, float, np.ndarray]]]:
    x = np.array(x0, dtype=float)
    history: List[Tuple[np.ndarray, float, float, np.ndarray]] = []

    for _ in range(max_iter):
        d, gamma = auxiliary_direction(x)
        history.append((x.copy(), f(x), gamma, d.copy()))

        if gamma <= eps:
            return x, history

        alpha_max = max_feasible_alpha(x, d)
        if alpha_max <= 1e-14:
            return x, history

        phi = lambda a: f(x + a * d)
        alpha = golden_section_min(phi, 0.0, alpha_max, tol=1e-10)
        x = x + alpha * d

    return x, history


def main() -> None:
    x0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    print("Initial point:", x0)
    print("Feasible:", feasible(x0))
    print("Exact solution for this example: x* = (0, 0, -sqrt(2), -sqrt(2))")
    print("Exact value:", 2.0 * math.exp(-math.sqrt(2.0)))
    print()

    for eps in [1e-1, 1e-2, 1e-3]:
        x_opt, hist = zoutendijk(x0, eps=eps)
        print(f"epsilon = {eps:g}")
        print("  iterations:", len(hist) - 1)
        print("  x =", np.round(x_opt, 10))
        print("  f(x) =", f(x_opt))
        print("  g1 =", g1(x_opt))
        print("  g2 =", g2(x_opt))
        print("  g3 =", g3(x_opt))
        print("  h1 =", h1(x_opt))
        print("  last gamma =", hist[-1][2])
        print()


if __name__ == "__main__":
    main()