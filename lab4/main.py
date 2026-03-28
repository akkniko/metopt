import pandas as pd

from sgd import steepest_descent
from hooke_jeevs import hooke_jeeves
from BFGS import bfgs
from utils import f

x0 = [1.0, 1.0]

# --- SGD ---
x_sgd, hist_sgd = steepest_descent(x0, eps=1e-3)
df_sgd = pd.DataFrame(hist_sgd)

# --- Hooke-Jeeves ---
x_hj, hist_hj = hooke_jeeves(x0)
df_hj = pd.DataFrame(hist_hj)

# --- BFGS ---
x_bfgs, hist_bfgs = bfgs(x0)
df_bfgs = pd.DataFrame(hist_bfgs)

# --- Итоговая таблица ---
summary = pd.DataFrame([
    ["SGD", x_sgd, f(x_sgd), len(df_sgd)],
    ["Hooke-Jeeves", x_hj, f(x_hj), len(df_hj)],
    ["BFGS", x_bfgs, f(x_bfgs), len(df_bfgs)],
], columns=["Method", "x*", "f(x*)", "Iterations"])

# --- Запись в Excel ---
with pd.ExcelWriter("optimization_results.xlsx") as writer:
    df_sgd.to_excel(writer, sheet_name="SGD", index=False)
    df_hj.to_excel(writer, sheet_name="Hooke_Jeeves", index=False)
    df_bfgs.to_excel(writer, sheet_name="BFGS", index=False)
    summary.to_excel(writer, sheet_name="Summary", index=False)

print("Saved to optimization_results.xlsx")