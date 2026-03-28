import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f(x):
    x1, x2 = x
    s = x1**2 + 0.5 * x2**2
    return 2*x1 - 5*x2 + np.exp(s)

def grad_f(x):
    x1, x2 = x
    s = x1**2 + 0.5 * x2**2
    e = np.exp(s)
    df_dx1 = 2 + 2*x1*e
    df_dx2 = -5 + x2*e
    return np.array([df_dx1, df_dx2], dtype=float)
