import numpy as np
import pandas as pd
import utils
from utils import f, grad_f


#line search вспомогательный
def armijo_backtracking(x, p, g, alpha0=1.0, c=1e-4, rho=0.5, max_iter=50):
    alpha = alpha0
    fx = f(x)
    gp = np.dot(g, p)

    for _ in range(max_iter):
        x_new = x + alpha * p
        if f(x_new) <= fx + c * alpha * gp:
            return alpha
        alpha *= rho

    return alpha

def exploratory_search(x, delta):
    """
    метод идёт по координатам и пробует +/- delta.
    """
    x_best = x.copy()
    f_best = f(x_best)
    n = len(x_best)

    for i in range(n):
        # пробуем увеличить координату
        x_try = x_best.copy()
        x_try[i] += delta
        f_try = f(x_try)
        if f_try < f_best:
            x_best = x_try
            f_best = f_try
        else:
            # пробуем уменьшить координату
            x_try = x_best.copy()
            x_try[i] -= delta
            f_try = f(x_try)
            if f_try < f_best:
                x_best = x_try
                f_best = f_try

    return x_best, f_best

def hooke_jeeves(x0, delta0=1.0, eps=1e-6, gamma=2.0, beta=0.5, max_iter=500):
    """
    Метод Хука–Дживса.
    gamma - коэффициент ускорения при образцовом шаге
    beta  - коэффициент уменьшения шага при неудаче
    """
    x_base = np.array(x0, dtype=float)
    delta = float(delta0)

    history = []

    for k in range(max_iter):
        f_base = f(x_base)

        # Исследующий поиск
        x_new, f_new = exploratory_search(x_base, delta)

        history.append({
            "k": k,
            "x1": x_base[0],
            "x2": x_base[1],
            "f(x)": f_base,
            "delta": delta,
            "x1_new": x_new[0],
            "x2_new": x_new[1],
            "f_new": f_new
        })

        # критерий остановки
        if delta < eps:
            break

        if f_new < f_base:
            # образцовый шаг
            x_pattern = x_new + gamma * (x_new - x_base)

            # проверяем, стал ли образцовый шаг лучше
            if f(x_pattern) < f_new:
                x_base = x_pattern
            else:
                x_base = x_new
        else:
            # уменьшить шаг поиска
            delta *= beta

        if delta < eps:
            break

    return x_base, history


if __name__ == "__main__":
    x0 = np.array([1.0, 1.0])

    x_hj, hist_hj = hooke_jeeves(x0, delta0=1.0, eps=1e-6)
    df_hj = pd.DataFrame(hist_hj)

    print("=== Hooke-Jeeves ===")
    print("x* =", x_hj)
    print("f(x*) =", f(x_hj))
    print(df_hj.to_string(index=False))

    print("\n" + "="*60 + "\n")
