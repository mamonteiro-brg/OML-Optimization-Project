import numpy as np
import math
from generic_helper import loadData, printStatus, startTimer, endTimer
from plot_helper import plotPolynomialCurve, plotDataset


import random
import numpy as np

from itertools import chain,repeat


def polynomial(w, x):
    return np.polyval(w[::-1], x)


def cost(w, x, y, N):
    result = 0
    for n in range(0, N):
        val = polynomial(w, x[n]) - y[n]
        val = val * val
        result = result + val
    return (0.5 * result / N)


def gradient(w, x, y, N):
    grad = w * 0.
    I = len(w) - 1
    pw = range(I + 1)
    for n in range(0, N):
     #   print("Cenas: ", n, N)
        val = polynomial(w, x[n]) - y[n]
        phi = np.power(x[n], pw)
        grad = grad + phi * val
    return grad / N


#def step(w, x, y, g, N):
#    alpha = 0.2
#    return alpha

def backtracking(w,xt,yt,Gb,Nt):

    #Initialize
    #alpha = alpha_bar

    alpha=1

    #alpha Gradiente Transposto * gradiente
    #gradiente total = gT
    Gt = gradient(w,xt,yt,Nt)

    #Mudar para produto interno
    #Gt*Gb


    inner_product = np.inner(Gt, np.transpose(Gb))
    cost_k = cost(w,xt,yt,Nt)

    w_aux = w
    while cost(w_aux,xt,yt,Nt) > (cost_k - alpha * inner_product):
        alpha = alpha/2

        if alpha * np.linalg.norm(Gb) <= 1.e-9 :
            break

        w_aux = w - alpha * gradient(w,xt,yt,Nb)

    return alpha



def backtracking_william(w,xt,yt,Gb,Gt):

    #Initialize
    #alpha = alpha_bar

    alpha=1

    inner_product = np.inner(Gt, np.transpose(Gb))
    cost_k = cost(w,xt,yt,Nt)

    #w_aux = w
    while cost_k > (cost_k - alpha * inner_product):
        alpha = alpha/2

        if alpha * np.linalg.norm(Gb) <= 1.e-9 :
            break

        #w_aux = w - alpha * gradient(w,xt,yt,Nb)

    return alpha


'''
def step(w,x,y,g,N):
    grad=w*0.
    I=len(w)-1
    pw = range(I+1)
    alpha1=0;alpha2=0;
    for n in range(0,N):
        val=polynomial(w,x[n])-y[n]
        phi=np.power(x[n],pw)
        dot=np.dot(phi,g)
        alpha1=alpha1+val*dot
        alpha2=alpha2+dot*dot
    return(alpha1/alpha2)
'''

# In this function we need to select the records sequentially and not in a random way.
def ncycles(iterable, n):
    return chain.from_iterable(repeat(tuple(iterable), n))

#def get_data_cycle(xt, yt, position, Nb):
#    size = len(xt)
#    next_pos = position+Nb
#    cicle_step = (size % (next_pos)) + 1
#    cicle_list_x = list(ncycles(xt, cicle_step))
#    cicle_list_y = list(ncycles(yt, cicle_step))

#    return cicle_list_x[position:(next_pos)], cicle_list_y[position:(next_pos)], next_pos


def get_data_cycle(xt, yt, position, Nb):

    size = len(xt)
    next_pos = position+Nb
    cicle_step = (size % (next_pos)) + 1
    cicle_list_x = list(ncycles(xt, cicle_step))
    cicle_list_y = list(ncycles(yt, cicle_step))

    return cicle_list_x[position:(next_pos)], cicle_list_y[position:(next_pos)], next_pos

def get_data_cycle_new (xt,yt,Nb,pos_actual):
    res_x=[]
    res_y=[]
    my_index = 0

    for i in range(Nb):

        pos_actual_aux = pos_actual +i

        if (pos_actual_aux) < len(xt):

            res_x.append(xt[pos_actual_aux])
            res_y.append(yt[pos_actual_aux])
            my_index = pos_actual_aux

        else:

           # aux_posicao_actual_mod = (divmod(pos_actual_aux,len(xt))[1])

            aux_posicao_actual_mod = pos_actual_aux % len(xt)

            #print("--------------")
            #print(pos_actual_aux)
            #print(aux_posicao_actual_mod)
            #print("--------------")

            res_x.append(xt[aux_posicao_actual_mod])
            res_y.append(yt[aux_posicao_actual_mod])
            my_index = aux_posicao_actual_mod


    return res_x,res_y,my_index



def get_data(xt, yt, Nt, Nb):
    xb = []
    yb = []
    for n in range(0, Nb):
        pos = math.ceil(np.random.rand() * (Nt - 1))
        xb.append(xt[pos])
        yb.append(yt[pos])
    return xb, yb


dataset_path = 'datasets/p5_s400_r10.csv'

I = 5

ITERATION_MAX = 200  # STOP CASE 1
PRECISION = 1.e-6  # STOP CASE 2

posicao_actual = 0


if __name__ == '__main__':
    timer = startTimer()

    N, x, y, Nt, xt, yt, _, xv, yv = loadData(dataset_path)

    #Nb = 32


    w = np.array([1] * (I + 1))

    plotDataset(x, y)

    w = np.array([1] * (I + 1))
    # w=np.array([1/5,-1/2,1])
    print(cost(w, xt, yt, Nt))
    print(gradient(w, xt, yt, Nt))

    ok = True
    iter = 0
    while ok:
        Theta = random.uniform(1, 2)
        Nb = math.ceil(Theta**iter)
        Nb=min(Nb,Nt)
        #Nb = 32

        #xb, yb = get_data(xt, yt, Nt, Nb)
        xb,yb,posicao_actual =  get_data_cycle_new (xt,yt,Nb,posicao_actual)

        #xb, yb, posicao_actual = get_data_cycle(xt,yt, posicao_actual,Nb)
        Gb=gradient(w,xb,yb,Nb)
        #g = gradient(w, xb, yb, Nb)

        print("-----------")
        print("Nt  : ", Nt)
        print("Nb - valor calculado  : ", Nb)
        print("-----------")

        Gt = gradient(w, xt, yt, Nt)
        Norm = np.linalg.norm(Gb)

        # calculo da direcao estocastica do subconjunto
        #alpha=backtracking(w,xt,yt,Gb,Gt)

        alpha=backtracking_william(w,xt,yt,Gb,Gt)


        w = w - alpha * Gb
        # print(w)
        # print(g)
        if (iter > ITERATION_MAX):
            ok = False
        print(iter, cost(w, xb, yb, Nb))
        if (Norm < PRECISION):
            ok = False
        iter = iter + 1

    endTimer(timer)

    printStatus(I, w, cost(w, xt, yt, Nt), cost(w, xv, yv, N - Nt))

    plotPolynomialCurve(I, w, x, y)

    print('bye')
    exit(0)

