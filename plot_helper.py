import math

import matplotlib.pyplot as plt
import numpy as np


def plotDataset(x, y):
    fig = plt.figure()
    plt.plot(x, y, 'ro')
    plt.show()


def plotCostEpoch(xList):
    print('Ploting cost per Epoch', xList)
    plotCost(xList, 'Epochs')


def plotCostIter(xList):
    print('Ploting cost per Iteration', xList)
    plotCost(xList, 'Iteration')


def plotPolynomialCurve(I, w, x, y):
    fig = plt.figure()
    plt.plot(x, y, 'ro')
    s = np.linspace(0, 1, 100, endpoint=True)
    if I == 2:
        r = w[0] + w[1] * (s) + w[2] * (s * s)
    elif I == 3:
        r = w[0] + w[1] * (s) + w[2] * (s * s) + w[3] * (s * s * s)
    elif I == 4:
        r = w[0] + w[1] * (s) + w[2] * (s * s) + w[3] * (s * s * s) + w[4] * (s * s * s * s)
    elif I == 5:
        r = w[0] + w[1] * (s) + w[2] * (s * s) + w[3] * (s * s * s) + w[4] * (s * s * s * s) + w[5] * (
                s * s * s * s * s)
    elif I == 6:
        r = w[0] + w[1] * (s) + w[2] * (s * s) + w[3] * (s * s * s) + w[4] * (s * s * s * s) + w[5] * (
                s * s * s * s * s) + \
            w[6] * (s * s * s * s * s * s)
    elif I == 7:
        r = w[0] + w[1] * (s) + w[2] * (s * s) + w[3] * (s * s * s) + w[4] * (s * s * s * s) + w[5] * (
                s * s * s * s * s) + \
            w[6] * (s * s * s * s * s * s) + w[7] * (s * s * s * s * s * s * s)
    else:
        raise ValueError('Unsupported polynomial degree.')

    plt.plot(s, r, '-b', linewidth=3, label=r'$exato$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polynomial Curve')
    plt.show()


def plotCost(xList, xLabel):
    """Plot number of gradient descent iterations verus cost function, J,
    values at values of theta

    Returns
    -----------
    matploblib figure
    """

    plt.figure()
    plt.plot(np.arange(1, len(xList) + 1), xList, label=r'$J(\theta)$')
    plt.xlabel(xLabel)
    plt.ylabel(r'$J(\theta)$')
    plt.title("Cost vs {xLabel} of Gradient Descent".format(xLabel=xLabel))
    plt.legend(loc='best')
    plt.show()


def polynomial(w, x):
    return np.polyval(w[::-1], x)


def mae(w, x, y, N):
    result = 0
    for n in range(0, N):
        val = abs(polynomial(w, x[n]) - y[n])
        # val = val * val
        result = result + val
    return result / N


def mse(w, x, y, N):
    result = 0
    for n in range(0, N):
        val = abs(polynomial(w, x[n]) - y[n])
        # val = val * val
        result = result + pow(val, 2)
    return result / N


def rmse(w, x, y, N):
    return math.sqrt(mse(w, x, y, N))
