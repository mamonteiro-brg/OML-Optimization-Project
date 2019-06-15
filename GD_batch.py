import math

import numpy as np
from generic_helper import loadData, printStatus
from plot_helper import plotPolynomialCurve, plotDataset, plotCostEpoch, plotCostIter, rmse, mae, mse, polynomial

a1 = 'datasets/p5_s1000_r30.csv'
a2 = 'datasets/p6_s1000_r20.csv'
a3 = 'datasets/p7_s1000_r10.csv'

dataset_path = a1
I = 3


# def polynomial(w, x):
#     return np.polyval(w[::-1], x)


def cost(w, x, y, N):
    result = 0
    for n in range(0, N):
        val = polynomial(w, x[n]) - y[n]
        val = val * val
        result = result + val
    return 0.5 * result / N


def gradient(w, x, y, N):
    grad = w * 0.
    I = len(w) - 1
    pw = range(I + 1)
    for n in range(0, N):
        val = polynomial(w, x[n]) - y[n]
        phi = np.power(x[n], pw)
        grad = grad + phi * val
    return grad / N


def step(w, x, y, g, N):
    grad = w * 0.
    I = len(w) - 1
    pw = range(I + 1)
    alpha1 = 0
    alpha2 = 0
    for n in range(0, N):
        val = polynomial(w, x[n]) - y[n]
        phi = np.power(x[n], pw)
        dot = np.dot(phi, g)
        alpha1 = alpha1 + val * dot
        alpha2 = alpha2 + dot * dot
    return alpha1 / alpha2


ITERATION_MAX = 1000  # STOP CASE 1
PRECISION = 1.e-6  # STOP CASE 2

if __name__ == '__main__':
    N, x, y, Nt, xt, yt, _, xv, yv = loadData(dataset_path)
    plotDataset(x, y)

    w = np.array([1] * (I + 1))
    # w=np.array([1/5,-1/2,1])
    print(cost(w, xt, yt, Nt))
    print(gradient(w, xt, yt, Nt))

    ok = True
    iter = 0
    epoch = 0

    costEpochList = []
    costIterList = []

    while ok:
        g = gradient(w, xt, yt, Nt)
        Norm = np.linalg.norm(g)
        alpha = step(w, xt, yt, g, Nt)
        w = w - 0.9 * alpha * g
        # print(w)
        # print(g)
        if iter > ITERATION_MAX:
            ok = False

        custo = cost(w, xt, yt, Nt)

        costEpochList.append(custo)
        costIterList.append(custo)
        # print('DAD ::: ', iter, custo)

        if Norm < PRECISION:
            ok = False
        iter = iter + 1

    printStatus(I, w, cost(w, xt, yt, Nt), cost(w, xv, yv, N - Nt))

    plotPolynomialCurve(I, w, x, y)
    plotCostEpoch(np.array(costEpochList))
    plotCostIter(np.array(costIterList))

    print('MAE  :: ', mae(w, xv, yv, N - Nt))
    print('MSE  :: ', mse(w, xv, yv, N - Nt))
    print('RMSE :: ', rmse(w, xv, yv, N - Nt))

    print('bye')
    exit(0)
