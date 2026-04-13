import math
from typing import Callable, List, Tuple

import numpy as np

# ============================================================
# Исходная задача
# ============================================================
# Минимизировать:
#   f(x) = 4 x1^2 + 2 x1 x2 + 3 x2^2 + exp(x3) + exp(x4)
#
# При ограничениях:
#   g1(x) = x1^2 + x2^2 + x3^2 + x4^2 - 4 <= 0
#   g2(x) = x1 + 2 x2 + x3 - 1 <= 0
#   g3(x) = -x1 + x4 - 1 <= 0
#   h1(x) = x2 - x3 + x4 = 0
#
# Начальная точка:
#   x0 = (0, 0.3, 0.3, 0)
# ============================================================


# ============================================================
# Целевая функция и её градиент
# ============================================================
def f(x: np.ndarray) -> float:
    """Целевая функция."""
    x1, x2, x3, x4 = x
    return 4.0 * x1 * x1 + 2.0 * x1 * x2 + 3.0 * x2 * x2 + math.exp(x3) + math.exp(x4)


def grad_f(x: np.ndarray) -> np.ndarray:
    """Градиент целевой функции."""
    x1, x2, x3, x4 = x
    return np.array(
        [8.0 * x1 + 2.0 * x2,
         2.0 * x1 + 6.0 * x2,
         math.exp(x3),
         math.exp(x4)],
        dtype=float
    )


# ============================================================
# Ограничения и их градиенты
# ============================================================
def g1(x: np.ndarray) -> float:
    """Нелинейное неравенство g1(x) <= 0."""
    return float(np.dot(x, x) - 4.0)


def grad_g1(x: np.ndarray) -> np.ndarray:
    """Градиент g1."""
    return 2.0 * np.array(x, dtype=float)


def g2(x: np.ndarray) -> float:
    """Линейное неравенство g2(x) <= 0."""
    x1, x2, x3, x4 = x
    return float(x1 + 2.0 * x2 + x3 - 1.0)


def grad_g2(x: np.ndarray) -> np.ndarray:
    """Градиент g2."""
    return np.array([1.0, 2.0, 1.0, 0.0], dtype=float)


def g3(x: np.ndarray) -> float:
    """Линейное неравенство g3(x) <= 0."""
    x1, x2, x3, x4 = x
    return float(-x1 + x4 - 1.0)


def grad_g3(x: np.ndarray) -> np.ndarray:
    """Градиент g3."""
    return np.array([-1.0, 0.0, 0.0, 1.0], dtype=float)


def h1(x: np.ndarray) -> float:
    """Линейное равенство h1(x) = 0."""
    x1, x2, x3, x4 = x
    return float(x2 - x3 + x4)


def grad_h1(x: np.ndarray) -> np.ndarray:
    """Градиент h1."""
    return np.array([0.0, 1.0, -1.0, 1.0], dtype=float)


def feasible(x: np.ndarray, tol: float = 1e-10) -> bool:
    """Проверка допустимости точки."""
    return (g1(x) <= tol) and (g2(x) <= tol) and (g3(x) <= tol) and (abs(h1(x)) <= tol)


# ============================================================
# Описание всех неравенств одной таблицей
# ============================================================
# Это нужно для того, чтобы в вспомогательную задачу автоматически
# попадали все активные ограничения, а не только g1.
INEQ_CONSTRAINTS = [
    ("g1", g1, grad_g1),  # нелинейное
    ("g2", g2, grad_g2),   # линейное
    ("g3", g3, grad_g3),   # линейное
]


# ============================================================
# Симплекс-метод
# ============================================================
# Решается задача:
#   max c^T z
#   A z <= b
#   z >= 0
#
# Табличный симплекс с искусственно не нужен,
# потому что стартовая базисная допустимая точка задаётся слэками.
#
# Используется правило Бланда:
#   - выбираем первый допустимый входящий столбец;
#   - при выборе выходящей строки берём первую из строк с минимальным отношением.
# Это уменьшает риск зацикливания.
# ============================================================


def simplex_max(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> Tuple[np.ndarray, float, int]:
    """
    Решает задачу линейного программирования:
        max c^T z
        A z <= b
        z >= 0

    Возвращает:
        z_opt  - оптимальный план по основным переменным
        opt    - значение целевой функции
        code   - 0, если всё прошло нормально
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    m, n = A.shape

    # Если в какой-то строке правая часть отрицательная,
    # умножаем всю строку на -1, чтобы стандартный базис был допустимым.
    for i in range(m):
        if b[i] < 0:
            A[i, :] *= -1.0
            b[i] *= -1.0

    # Таблица симплекс-метода:
    # [ A | I | b ]
    # Последняя строка — целевая функция.
    T = np.zeros((m + 1, n + m + 1), dtype=float)
    T[:m, :n] = A
    T[:m, n:n + m] = np.eye(m)
    T[:m, -1] = b
    T[m, :n] = -c

    # Начальный базис: все слэки
    basis = [n + i for i in range(m)]

    for _ in range(max_iter):
        # В последней строке ищем отрицательные коэффициенты:
        # если они есть, можно улучшить целевую функцию.
        reduced = T[m, :-1]
        entering_candidates = np.where(reduced < -tol)[0]

        # Если отрицательных коэффициентов нет — оптимум найден.
        if len(entering_candidates) == 0:
            sol = np.zeros(n + m, dtype=float)
            for i, bi in enumerate(basis):
                sol[bi] = T[i, -1]
            return sol[:n], float(T[m, -1]), 0

        # Входящий столбец выбираем по правилу Бланда:
        # берём первый подходящий.
        entering = int(entering_candidates[0])

        # Выбираем строку, которая покинет базис, по правилу минимального отношения.
        column = T[:m, entering]
        leaving_candidates = np.where(column > tol)[0]

        # Если нет положительных элементов в столбце,
        # задача неограничена сверху.
        if len(leaving_candidates) == 0:
            raise RuntimeError("ЛП-задача неограничена.")

        ratios = T[leaving_candidates, -1] / column[leaving_candidates]
        min_ratio = np.min(ratios)
        tied = leaving_candidates[np.where(np.abs(ratios - min_ratio) <= 1e-12)[0]]
        leaving = int(tied[0])

        # Поворот симплекс-таблицы.
        piv = T[leaving, entering]
        T[leaving, :] /= piv

        for r in range(m + 1):
            if r != leaving:
                T[r, :] -= T[r, entering] * T[leaving, :]

        basis[leaving] = entering

    raise RuntimeError("Превышено число итераций симплекс-метода.")


# ============================================================
# Вспомогательная задача Зойтендейка
# ============================================================
# Вводим переменные:
#   d = d_plus - d_minus
#   d_plus >= 0, d_minus >= 0
#   gamma >= 0
#
# Тогда вспомогательная задача:
#   max gamma
#   при
#       grad f(x)^T d + gamma <= 0
#       grad g_i(x)^T d + gamma <= 0   для всех активных неравенств
#       grad h(x)^T d = 0
#       |d_j| <= 1
#
# Равенство записываем как две неравенства:
#       grad h(x)^T d <= 0
#      -grad h(x)^T d <= 0
#
# Ограничение |d_j| <= 1 переписывается как:
#       d_plus_j + d_minus_j <= 1


def auxiliary_direction(x: np.ndarray, active_tol: float = 1e-8) -> Tuple[np.ndarray, float]:
    """
    Строит и решает вспомогательную задачу Зойтендейка.

    Возвращает:
        d     - направление спуска
        gamma - значение вспомогательной переменной
    """
    gf = grad_f(x)

    # Сюда будем собирать все ограничения ЛП.
    A: List[np.ndarray] = []
    b: List[float] = []

    # --------------------------------------------------------
    # 1) Ограничение на убывание целевой функции:
    #    grad f(x)^T d + gamma <= 0
    # --------------------------------------------------------
    row = np.zeros(9, dtype=float)
    row[:4] = gf           # коэффициенты при d_plus
    row[4:8] = -gf         # коэффициенты при d_minus
    row[8] = 1.0           # коэффициент при gamma
    A.append(row)
    b.append(0.0)

    # 2) Активные неравенства:
    #    если g_i(x) почти активно, добавляем его в LP.
    #    Это исправляет недочёт исходной версии кода,
    #    где учитывался только g1.
    for name, g_fun, grad_fun in INEQ_CONSTRAINTS:
        if g_fun(x) >= -active_tol:
            gg = grad_fun(x)

            row = np.zeros(9, dtype=float)
            row[:4] = gg
            row[4:8] = -gg
            row[8] = 1.0

            A.append(row)
            b.append(0.0)

    # 3) Равенство:
    #    grad h(x)^T d = 0
    #    записываем как две противоположные неравенства.
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

    # 4) Ограничения на направление:
    #    |d_j| <= 1, j = 1..4
    #    В форме d_plus_j + d_minus_j <= 1.
    for i in range(4):
        row = np.zeros(9, dtype=float)
        row[i] = 1.0
        row[4 + i] = 1.0
        A.append(row)
        b.append(1.0)

    # Целевая функция вспомогательной задачи:
    #   max gamma
    c = np.zeros(9, dtype=float)
    c[8] = 1.0

    # Решаем LP.
    sol, gamma, _ = simplex_max(np.array(A), np.array(b), c)

    # Восстанавливаем d = d_plus - d_minus
    d = sol[:4] - sol[4:8]
    return d, gamma


# ============================================================
# Поиск максимального допустимого шага
# ============================================================
# Идём по направлению d из точки x и ищем максимальный alpha,
# при котором x + alpha d остаётся допустимой.
#
# Для этой задачи удобно использовать расширение границы + бисекцию:
# сначала быстро находим интервал [lo, hi], где точка на lo допустима,
# а на hi уже недопустима, затем уточняем границу бисекцией.
# ============================================================


def max_feasible_alpha(x: np.ndarray, d: np.ndarray) -> float:
    """
    Максимальный допустимый шаг alpha вдоль направления d.
    """
    lo = 0.0
    hi = 1.0

    # Сначала расширяем правую границу, пока точка остаётся допустимой.
    for _ in range(60):
        if feasible(x + hi * d, tol=1e-12):
            lo = hi
            hi *= 2.0
        else:
            break
    else:
        # Если даже после 60 расширений точка допустима,
        # возвращаем lo как найденную нижнюю оценку.
        return lo

    # Затем уточняем границу методом бисекции.
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if feasible(x + mid * d, tol=1e-12):
            lo = mid
        else:
            hi = mid

    return lo


# ============================================================
# Золотое сечение для одномерного поиска минимума
# ============================================================


def golden_section_min(
    phi: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10
) -> float:
    """
    Минимум функции phi на отрезке [a, b] методом золотого сечения.
    """
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


# ============================================================
# Метод Зойтендейка
# ============================================================
# На каждой итерации:
#   1) строим вспомогательную задачу;
#   2) решаем её симплекс-методом;
#   3) если gamma мало, останавливаемся;
#   4) иначе ищем допустимый шаг;
#   5) делаем шаг и повторяем.
# ============================================================


def zoutendijk(
    x0: np.ndarray,
    eps: float,
    max_iter: int = 100,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, float, float, np.ndarray]]]:
    """
    Метод Зойтендейка.

    Возвращает:
        x_opt  - найденная точка
        history - история итераций:
                  (x_k, f(x_k), gamma_k, d_k)
    """
    x = np.array(x0, dtype=float)
    history: List[Tuple[np.ndarray, float, float, np.ndarray]] = []

    for _ in range(max_iter):
        # Строим вспомогательное направление.
        d, gamma = auxiliary_direction(x)

        # Запоминаем текущую итерацию.
        history.append((x.copy(), f(x), gamma, d.copy()))

        # Если gamma мало, считаем, что точка близка к стационарной.
        if gamma <= eps:
            return x, history

        # Находим максимально возможный допустимый шаг.
        alpha_max = max_feasible_alpha(x, d)
        if alpha_max <= 1e-14:
            return x, history

        # На [0, alpha_max] минимизируем целевую функцию вдоль направления d.
        phi = lambda a: f(x + a * d)
        alpha = golden_section_min(phi, 0.0, alpha_max, tol=1e-10)

        # Делаем шаг.
        x = x + alpha * d

    return x, history

def main() -> None:
    # Начальная допустимая точка.
    x0 = np.array([0.0, 0.3, 0.3, 0.0], dtype=float)
    print("========================================================================================")
    print("========================================================================================")
    print("Initial point:", x0)
    print("Feasible:", feasible(x0))

    # Для этой конкретной задачи можно указать точный ответ.
    print("Exact solution for this example: x* = (0, 0, -sqrt(2), -sqrt(2))")
    print("Exact value:", 2.0 * math.exp(-math.sqrt(2.0)))
    print()

    # Решаем задачу с разной точностью.
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
        # print(hist)

        # for i in hist:

        # print()
   


if __name__ == "__main__":
    main()



    # 