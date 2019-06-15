import csv
import numpy as np
import math
import time


def loadData(file_path):
    x = []
    y = []
    N = 0
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for u in reader:
            x = np.append(x, float(u[0]))
            y = np.append(y, float(u[1]))
            N = N + 1
    Nt = math.floor(N * 0.8)
    xt = x[0:Nt]
    yt = y[0:Nt]
    xv = x[Nt:N]
    yv = y[Nt:N]
    Nv = N - Nt
    return N, x, y, Nt, xt, yt, Nv, xv, yv


def printStatus(I, w, in_error, out_error):
    print("======================================")
    print("I={:d}".format(I))
    print("coefficient", w)
    print("in error sample {:e}".format(in_error))
    print("out error sample {:e}".format(out_error))


def startTimer():
    return time.time()


def endTimer(startTimer):
    end = time.time()
    print("Execution time: ", end - startTimer, "seconds")
