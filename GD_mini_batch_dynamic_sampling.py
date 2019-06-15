import numpy as np
import math
from generic_helper import loadData, printStatus, startTimer, endTimer
from plot_helper import plotPolynomialCurve, plotDataset, plotCostIter, plotCostEpoch, rmse, mae, mse, polynomial
import random
import math

a1 = 'datasets/p5_s1000_r30.csv'
a2 = 'datasets/p6_s1000_r20.csv'
a3 = 'datasets/p7_s1000_r10.csv'

dataset_path = a3
I = 7


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
    alpha = 0.2
    return alpha


def get_data_cycle_new(xt, yt, Nb, pos_actual):
    res_x = []
    res_y = []
    my_index = 0

    for i in range(Nb):

        pos_actual_aux = pos_actual + i

        if (pos_actual_aux) < len(xt):

            res_x.append(xt[pos_actual_aux])
            res_y.append(yt[pos_actual_aux])
            my_index = pos_actual_aux

        else:

            # aux_posicao_actual_mod = (divmod(pos_actual_aux,len(xt))[1])

            aux_posicao_actual_mod = pos_actual_aux % len(xt)

            # print("--------------")
            # print(pos_actual_aux)
            # print(aux_posicao_actual_mod)
            # print("--------------")

            res_x.append(xt[aux_posicao_actual_mod])
            res_y.append(yt[aux_posicao_actual_mod])
            my_index = aux_posicao_actual_mod

    return res_x, res_y, my_index


def get_data(xt, yt, Nt, Nb):
    xb = []
    yb = []
    for n in range(0, Nb):
        pos = math.ceil(np.random.rand() * (Nt - 1))
        xb.append(xt[pos])
        yb.append(yt[pos])
    return xb, yb


ITERATION_MAX = 1000  # STOP CASE 1
PRECISION = 1.e-6  # STOP CASE 2


if __name__ == '__main__':
    timer = startTimer()

    N, x, y, Nt, xt, yt, _, xv, yv = loadData(dataset_path)

    w = np.array([1] * (I + 1))

    plotDataset(x, y)

    w = np.array([1] * (I + 1))
    # w=np.array([1/5,-1/2,1])
    # print(cost(w, xt, yt, Nt))
    # print(gradient(w, xt, yt, Nt))

    ok = True
    iter = 0
    Nb = 1

    nAtual = 0

    costEpochList = []
    costIterList = []

    while ok:
        Theta = random.uniform(1, 2)
        Nb = math.ceil(Theta ** iter)
        Nb = min(Nb, Nt)

        xb, yb = get_data(xt, yt, Nt, Nb)
        g = gradient(w, xb, yb, Nb)
        Norm = np.linalg.norm(g)

        # print("-----------")
        # print("Nt  : ", Nt)
        # print("Nb - valor calculado  : ", Nb)
        # print("-----------")

        gT = gradient(w, xt, yt, Nt)
        Norm = np.linalg.norm(gT)

        alpha = step(w, xb, yb, g, Nb)

        w = w - alpha * g
        # print(w)
        # print(g)
        if iter > ITERATION_MAX:
            ok = False

        # calcula o custo desta iteração
        custo = cost(w, xb, yb, Nb)

        # calcula batchSize acumulado
        nAtual = nAtual + Nb

        # Calc Epoch atual
        epAt = nAtual // Nt

        # Se o epoch avancar > guarda o custo na lista
        if epAt > len(costEpochList):
            costEpochList.append(custo)

        costIterList.append(custo)
        # print(iter, custo)

        if Norm < PRECISION:
            ok = False
        iter = iter + 1

    endTimer(timer)

    printStatus(I, w, cost(w, xt, yt, Nt), cost(w, xv, yv, N - Nt))

    plotPolynomialCurve(I, w, x, y)

    print('EPOCH ::::: \n', costEpochList)
    print('ITER  :::::\n', costIterList)
    plotCostEpoch(np.array(costEpochList))
    plotCostIter(np.array(costIterList))

    print('bye')
    exit(0)
