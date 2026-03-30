import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
from utils import grad_f, f

def golden_section_minimize(phi, a, b, tol=1e-8, max_iter=200):
    '''
    нахождение минимума функции на интервале
        - берём две точки внутри интервала
        - сравниваем значения функции
        - отбрасываем «плохую» часть
    '''
    gr = (np.sqrt(5) - 1) / 2  # 0.618 - коэф золотого сечения, позволяет: не пересчитывать 1 из точек на каждом шаге
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = phi(c)
    fd = phi(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = phi(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = phi(d)

    alpha_star = (a + b) / 2
    return alpha_star

def bracket_minimum(phi, alpha0=0.0, h=1.0, expand=2.0, max_steps=50):
    """
    Находит интервал [a, b], содержащий минимум
    """
    a = alpha0 #нач точка
    fa = phi(a)
    b = a + h  #делаем шаг вправо
    fb = phi(b)

    # Если пошли вверх - интервал найден
    if fb > fa:
        return a, b

    for _ in range(max_steps):
        h *= expand #увеличиваем шаг, И идём дальше, пока не найдём:𝑓(𝑏) > 𝑓(𝑎)
        a, fa = b, fb
        b = a + h
        fb = phi(b)
        if fb > fa:
            return a - h / expand, b

    return 0.0, b

def exact_line_search(xk, gk):
    """
    нахождение оптимального шага - alpha = argmin_{alpha>=0} f(xk - alpha*gk)
    """
    #Строит функцию
    phi = lambda alpha: f(xk - alpha * gk)
    a, b = bracket_minimum(phi, alpha0=0.0, h=1.0) #нахождение интервала
    alpha_star = golden_section_minimize(phi, a, b, tol=1e-10)  #нахождение минимума внутри интервала
    return alpha_star

def steepest_descent(x0, eps=1e-6, max_iter=100):
    x = np.array(x0, dtype=float)
    history = []

    for k in range(max_iter):
        g = grad_f(x)
        g_norm = np.linalg.norm(g)

        row = {
            "k": k,
            "x1": x[0],
            "x2": x[1],
            "f(x)": f(x),
            "||grad||": g_norm,
            "alpha": np.nan,
            "step_len": np.nan,
            "dot(grad_next, step)": np.nan,
            "angle_deg": np.nan
        }

        if g_norm < eps:
            history.append(row)
            break

        alpha = exact_line_search(x, g)
        x_next = x - alpha * g
        step = x_next - x

        g_next = grad_f(x_next)
        dot_val = float(np.dot(g_next, step))
        denom = np.linalg.norm(g_next) * np.linalg.norm(step)
        angle = np.degrees(np.arccos(np.clip(abs(dot_val) / denom, -1.0, 1.0))) if denom > 0 else np.nan

        row["alpha"] = alpha
        row["step_len"] = np.linalg.norm(step)
        row["dot(grad_next, step)"] = dot_val
        row["angle_deg"] = angle

        history.append(row)

        x = x_next

        if np.linalg.norm(step) < eps:
            break

    return x, history

def animate_descent( x0, delay=0.2):

    x1_min, x1_max = -3, 3
    x2_min, x2_max = -1, 3

    #строится прямоугольная сетка в плоскости (x1, x2)
    x1 = np.linspace(x1_min, x1_max, 60)
    x2 = np.linspace(x2_min, x2_max, 60)
    X1, X2 = np.meshgrid(x1, x2)
    Z = 2*X1 - 5*X2 + np.exp(X1**2 + 0.5*X2**2)

    plt.ion()
    fig, ax = plt.subplots(figsize=(15, 10))
    
    #отрисовка линий уровня f(x1,x2) = c
    ax.contour(X1, X2, Z, levels=30)

    line, = ax.plot([], [], "o-", linewidth=2, markersize=5, label="Gradient polyline")#ломаная которая потом будет обновляться
    start_pt = ax.scatter([x0[0]], [x0[1]], c="red", s=80, label="x0")                #точка начального приближения
    curr_pt = ax.scatter([], [], c="green", s=80, label="current point")             #точка текущего положения

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Function level lines and gradient polyline")
    ax.legend()
    ax.grid(True)

    for i in range(1, len(path) + 1):
        p = path[:i] #Берётся первые i точек траектории
        line.set_data(p[:, 0], p[:, 1]) #обновление ломаной
        curr_pt.set_offsets([p[-1]]) #зеленая точка переносится в новую вершину ломанной 
        fig.canvas.draw()           
        fig.canvas.flush_events()
        plt.pause(delay)

    plt.ioff()
    plt.show(block=True)




x0 = np.array([1.0, 1.0])  
e = [0.1, 0.01, 0.001]
eps = e[2]

x_star, hist = steepest_descent(x0, eps=eps, max_iter=500)

df = pd.DataFrame(hist)

pd.set_option("display.precision", 8)
pd.set_option("display.max_columns", None)

print("The result of the steepest descent method")
print("x* =", x_star)
print("f(x*) =", f(x_star))
print()

print("Table of iterations:")
print(df.to_string(index=False))

path = np.array([[row["x1"], row["x2"]] for row in hist])
animate_descent( x0, delay=0.4)

print("\nOrthogonality check:")
print("For accurate line search there should be: grad(x_{k+1}) - (x_{k+1}-x_k) = 0")
check_rows = []

#обновление ломанной
for i in range(len(path) - 1):
    xk = path[i] 
    xk1 = path[i + 1]
    step = xk1 - xk
    g_next = grad_f(xk1)
    dot_val = np.dot(g_next, step)
    check_rows.append({
        "k": i,
        "dot(grad_next, step)": dot_val,
        "abs(dot)": abs(dot_val)
    })

check_df = pd.DataFrame(check_rows)
print(check_df.to_string(index=False))

