import utils
import numpy as np
import pandas as pd
import utils
from utils import f, grad_f
from hooke_jeevs import armijo_backtracking, hooke_jeeves


def bfgs(x0, eps=1e-6, max_iter=200):
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)   #нач приближение 
    history = []

    for k in range(max_iter):
        g = grad_f(x)
        g_norm = np.linalg.norm(g)

        history.append({
            "k": k,
            "x1": x[0],
            "x2": x[1],
            "f(x)": f(x),
            "||grad||": g_norm
        })

        if g_norm < eps:
            break

        # направление спуска
        p = -H @ g

        #if направление не убывающее- сбрасываем H
        if np.dot(p, g) >= 0:
            H = np.eye(n)
            p = -g

        alpha = armijo_backtracking(x, p, g, alpha0=1.0)

        x_new = x + alpha * p
        g_new = grad_f(x_new)

        s = x_new - x
        y = g_new - g

        ys = np.dot(y, s)

        # обновление BFGS
        if ys > 1e-12:
            rho = 1.0 / ys
            I = np.eye(n)
            Hy = H @ y

            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        else:
            # если обновление плохое, сбрасываем H
            H = np.eye(n)

        x = x_new

        if np.linalg.norm(s) < eps:
            break

    return x, history

if __name__ == "__main__":
    x0 = np.array([1.0, 1.0])

    x_bfgs, hist_bfgs = bfgs(x0, eps=1e-6)
    df_bfgs = pd.DataFrame(hist_bfgs)

    print("=== BFGS ===")
    print("x* =", x_bfgs)
    print("f(x*) =", f(x_bfgs))
    print(df_bfgs.to_string(index=False))