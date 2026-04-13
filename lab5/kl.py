import math
from typing import Callable, List, Tuple

import numpy as np

# ============================================================
# Èñõîäíàÿ çàäà÷à
# ============================================================
def f(x: np.ndarray) -> float:
    x1, x2, x3, x4 = x
    return 4.0 * x1 * x1 + 2.0 * x1 * x2 + 3.0 * x2 * x2 + math.exp(x3) + math.exp(x4)

def grad_f(x: np.ndarray) -> np.ndarray:
    x1, x2, x3, x4 = x
    return np.array([8.0 * x1 + 2.0 * x2, 2.0 * x1 + 6.0 * x2, math.exp(x3), math.exp(x4)], dtype=float)

def g1(x: np.ndarray) -> float: return float(np.dot(x, x) - 4.0)
def grad_g1(x: np.ndarray) -> np.ndarray: return 2.0 * np.array(x, dtype=float)

def g2(x: np.ndarray) -> float:
    x1, x2, x3, x4 = x
    return float(x1 + 2.0 * x2 + x3 - 1.0)
def grad_g2(x: np.ndarray) -> np.ndarray: return np.array([1.0, 2.0, 1.0, 0.0], dtype=float)

def g3(x: np.ndarray) -> float:
    x1, x2, x3, x4 = x
    return float(-x1 + x4 - 1.0)
def grad_g3(x: np.ndarray) -> np.ndarray: return np.array([-1.0, 0.0, 0.0, 1.0], dtype=float)

def h1(x: np.ndarray) -> float:
    x1, x2, x3, x4 = x
    return float(x2 - x3 + x4)
def grad_h1(x: np.ndarray) -> np.ndarray: return np.array([0.0, 1.0, -1.0, 1.0], dtype=float)

def feasible(x: np.ndarray, tol: float = 1e-10) -> bool:
    return (g1(x) <= tol) and (g2(x) <= tol) and (g3(x) <= tol) and (abs(h1(x)) <= tol)

INEQ_CONSTRAINTS = [("g1", g1, grad_g1), ("g2", g2, grad_g2), ("g3", g3, grad_g3)]

# ============================================================
# Ñèìïëåêñ-ìåòîä (áåç èçìåíåíèé)
# ============================================================
def simplex_max(A: np.ndarray, b: np.ndarray, c: np.ndarray, tol: float = 1e-10, max_iter: int = 500) -> Tuple[np.ndarray, float, int]:
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    m, n = A.shape
    for i in range(m):
        if b[i] < 0: A[i, :] *= -1.0; b[i] *= -1.0
    T = np.zeros((m + 1, n + m + 1), dtype=float)
    T[:m, :n] = A; T[:m, n:n + m] = np.eye(m); T[:m, -1] = b; T[m, :n] = -c
    basis = [n + i for i in range(m)]
    for _ in range(max_iter):
        reduced = T[m, :-1]
        entering_candidates = np.where(reduced < -tol)[0]
        if len(entering_candidates) == 0:
            sol = np.zeros(n + m, dtype=float)
            for i, bi in enumerate(basis): sol[bi] = T[i, -1]
            return sol[:n], float(T[m, -1]), 0
        entering = int(entering_candidates[0])
        column = T[:m, entering]
        leaving_candidates = np.where(column > tol)[0]
        if len(leaving_candidates) == 0: raise RuntimeError("ËÏ-çàäà÷à íåîãðàíè÷åíà.")
        ratios = T[leaving_candidates, -1] / column[leaving_candidates]
        min_ratio = np.min(ratios)
        tied = leaving_candidates[np.where(np.abs(ratios - min_ratio) <= 1e-12)[0]]
        leaving = int(tied[0])
        piv = T[leaving, entering]
        T[leaving, :] /= piv
        for r in range(m + 1):
            if r != leaving: T[r, :] -= T[r, entering] * T[leaving, :]
        basis[leaving] = entering
    raise RuntimeError("Ïðåâûøåíî ÷èñëî èòåðàöèé ñèìïëåêñ-ìåòîäà.")

# ============================================================
# Âñïîìîãàòåëüíàÿ çàäà÷à Çîéòåíäåéêà (ÈÇÌÅÍÅÍÎ: delta ïåðåäàžòñÿ ÿâíî)
# ============================================================
def auxiliary_direction(x: np.ndarray, delta: float) -> Tuple[np.ndarray, float]:
    gf = grad_f(x)
    A: List[np.ndarray] = []
    b: List[float] = []

    row = np.zeros(9, dtype=float)
    row[:4] = gf; row[4:8] = -gf; row[8] = 1.0
    A.append(row); b.append(0.0)

    for name, g_fun, grad_fun in INEQ_CONSTRAINTS:
        # Èñïîëüçóåì äèíàìè÷åñêèé äîïóñê delta âìåñòî ôèêñèðîâàííîãî tol
        if g_fun(x) >= -delta:
            gg = grad_fun(x)
            row = np.zeros(9, dtype=float)
            row[:4] = gg; row[4:8] = -gg; row[8] = 1.0
            A.append(row); b.append(0.0)

    gh = grad_h1(x)
    row = np.zeros(9, dtype=float)
    row[:4] = gh; row[4:8] = -gh
    A.append(row); b.append(0.0)
    row = np.zeros(9, dtype=float)
    row[:4] = -gh; row[4:8] = gh
    A.append(row); b.append(0.0)

    for i in range(4):
        row = np.zeros(9, dtype=float)
        row[i] = 1.0; row[4 + i] = 1.0
        A.append(row); b.append(1.0)

    c = np.zeros(9, dtype=float); c[8] = 1.0
    sol, gamma, _ = simplex_max(np.array(A), np.array(b), c)
    d = sol[:4] - sol[4:8]
    return d, gamma

# ============================================================
# Ïîèñê ìàêñèìàëüíîãî äîïóñòèìîãî øàãà (áåç èçìåíåíèé)
# ============================================================
def max_feasible_alpha(x: np.ndarray, d: np.ndarray) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(60):
        if feasible(x + hi * d, tol=1e-12): lo = hi; hi *= 2.0
        else: break
    else: return lo
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if feasible(x + mid * d, tol=1e-12): lo = mid
        else: hi = mid
    return lo

# ============================================================
# Çîëîòîå ñå÷åíèå (áåç èçìåíåíèé)
# ============================================================
def golden_section_min(phi: Callable[[float], float], a: float, b: float, tol: float = 1e-10) -> float:
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = phi(c), phi(d)
    while abs(b - a) > tol:
        if fc < fd: b, d, fd = d, c, fc; c = b - gr * (b - a); fc = phi(c)
        else: a, c, fc = c, d, fd; d = a + gr * (b - a); fd = phi(d)
    return 0.5 * (a + b)

# ============================================================
# Ìåòîä Çîéòåíäåéêà (ÈÇÌÅÍÅÍÎ: äèíàìè÷åñêèé delta è âûâîä èñòîðèè)
# ============================================================
def zoutendijk(x0: np.ndarray, eps: float, max_iter: int = 100) -> Tuple[np.ndarray, List[Tuple[np.ndarray, float, float, np.ndarray]]]:
    x = np.array(x0, dtype=float)
    history: List[Tuple[np.ndarray, float, float, np.ndarray]] = []

    print(f"{'k':>3} | {'x1':>9} {'x2':>9} {'x3':>9} {'x4':>9} | {'f(x)':>10} | {'delta':>10} | {'gamma':>10}")
    print("-" * 78)

    for k in range(max_iter):
        # 1. Äèíàìè÷åñêèé ðàñ÷žò delta_k êàê ìàêñèìàëüíîãî íàðóøåíèÿ îãðàíè÷åíèé
        delta = max(0.0, g1(x), g2(x), g3(x), abs(h1(x)))
        # Äëÿ óñòîé÷èâîñòè ËÏ-çàäà÷è íå äàžì delta îïóñòèòüñÿ íèæå eps äî ñõîäèìîñòè
        delta = max(eps, delta)

        # 2. Âñïîìîãàòåëüíàÿ çàäà÷à ñ àêòóàëüíûì delta
        d, gamma = auxiliary_direction(x, delta)
        history.append((x.copy(), f(x), gamma, d.copy()))

        # 3. Âûâîä êîðòåæà (k, x_k, f(x_k), delta_k, gamma_k)
        print(f"{k:3d} | {x[0]:9.5f} {x[1]:9.5f} {x[2]:9.5f} {x[3]:9.5f} | {f(x):10.5f} | {delta:10.2e} | {gamma:10.2e}")

        # 4. Êðèòåðèé îñòàíîâêè
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
    x0 = np.array([0.0, 0.3, 0.3, 0.0], dtype=float)
    print("========================================================================================")
    print(f"Initial point: {x0}")
    print(f"Feasible: {feasible(x0)}")
    print(f"Exact solution: x* = (0, 0, -sqrt(2), -sqrt(2))")
    print(f"Exact value: {2.0 * math.exp(-math.sqrt(2.0)):.6f}")
    print("========================================================================================\n")

    for eps in [1e-1, 1e-2, 1e-3]:
        print(f"\n---  eps = {eps:g} ---")
        x_opt, hist = zoutendijk(x0, eps=eps)
        print(f"\ (eps={eps:g}):")
        print(f"  x*      = {np.round(x_opt, 6)}")
        print(f"  f(x*)   = {f(x_opt):.6f}")
        print(f"  g1      = {g1(x_opt):.2e}, g2 = {g2(x_opt):.2e}, g3 = {g3(x_opt):.2e}, h1 = {h1(x_opt):.2e}")
        print(f"  history: {len(hist)}")
        print("-" * 78)

if __name__ == "__main__":
    main()