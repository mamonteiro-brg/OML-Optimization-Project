#!python
'''
optimization algorithms
'''
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import numpy as np


from itertools import chain,repeat

def polynomial(w,x):
     return np.polyval(w[::-1],x)
 
def cost(w,x,y,N):
    result=0;
    for n in range(0,N):
        val=polynomial(w,x[n])-y[n] 
        val=val*val
        result=result+val
    return (0.5*result/N)    

def gradient(w,x,y,N):
    grad=w*0.
    I=len(w)-1
    pw = range(I+1)
    for n in range(0,N):
        #print("n : " , N)
        val=polynomial(w,x[n])-y[n]
        phi=np.power(x[n],pw)
        grad=grad+phi*val
        #print("final")
    return grad/N    
 
def step(w,x,y,g,N):
    alpha=0.2
    return alpha

#Algoritmo - Procura de Armijo com backtracking


# Nt - Numero total de registos usados para treino
# #Nb - Numero de registos que vão ser usadas para a amostra -  este valor está a ser calculado
    # o Nb vai ser calculado com Nb = ...ceil(T**iterador)


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

        if alpha * np.linalg.norm(Gb) <= 1.e-8 :
            break

        w_aux = w - alpha * gradient(w,xt,yt,Nb)

    return alpha




def get_data(xt,yt,Nt,Nb):
    xb=[];yb=[];
    for n in range(0,Nb):
        pos=math.ceil(np.random.rand()*(Nt-1))
        xb.append(xt[pos])
        yb.append(yt[pos])
    return xb,yb

# In this function we need to select the records sequentially and not in a random way.
def ncycles(iterable, n):
    return chain.from_iterable(repeat(tuple(iterable), n))

def get_data_cycle(xt, yt, position, Nb):
    size = len(xt)
    cicle_step = (size % (position+Nb)) + 1
    cicle_list_x = list(ncycles(xt, cicle_step))
    cicle_list_y = list(ncycles(yt, cicle_step))
    return cicle_list_x[position:(position+Nb)], cicle_list_y[position:(position+Nb)], (position+Nb)



def get_data_sequencial(xt,yt,Nt,Nb,posicao_actual,posicao_actual_w_Nb):
    xb=[];yb=[];
    ok = True

    while ok:
        for n in range(0,Nb):
            pos = n + posicao_actual
            #posicao_actual_w_Nb =posicao_actual_w_Nb-1
            Nb =Nb -1
            if pos == Nt-1:
                #print("-----------")
                #print("reiniciou porque pos = ", pos)
                #print("Nb está em = ", Nb)
                #print("posicao_actual_w_Nb está em = ",posicao_actual_w_Nb)
                #print("-----------")
                posicao_actual = 1
                n=0
                pos =  n + posicao_actual
                break # break here

            #print("pos está em = ",pos)
            xb.append(xt[pos])
            yb.append(yt[pos])
        return xb,yb

    if(Nb==0):  ok=False;


#=========== MAIN CODE ===============
# read the data file
x=[];y=[]; N=0
with open('P5-large_yes.csv', 'r') as file:
    reader = csv.reader(file)
    for u in reader:
        x=np.append(x,float(u[0]))
        y=np.append(y,float(u[1]))
        N=N+1
Nt=math.floor(N*0.8)
xt=x[0:Nt]     
yt=y[0:Nt] 
xv=x[Nt:N]     
yv=y[Nt:N] 
I=5;
#Nb=math.floor(Nt*0.1)
Nb=1
 

'''
fig = plt.figure()
plt.plot(x,y,'ro')
s = np.linspace(0, 1,100, endpoint = True)
r = 1/5-1/2*(s)+(s*s)-2/3*(s*s*s)+4*(s*s*s*s)-5*(s*s*s*s*s)
plt.plot(s, r, '-g', label=r'$exato$')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Curve')
plt.show()
'''

w=np.array([1]*(I+1));
#w=np.array([1/5,-1/2,1,-2/3,4,-5])
#xb,yb=get_data(xt,yt,Nt,Nb)
#print(cost(w,xb,yb,Nb))
#print(gradient(w,xb,yb,Nb))


ok=True; iter=0

#posicao actual que vai ser utlizada para verificar em que po
posicao_actual = 0

while ok:
    # O dataset tem de ser modificado e o que teremos de fazer é o seguinte
    # Calcular o T random entre 1 e 2
    import random
    Theta = random.uniform(1, 2)

    # O nosso Nb vai ser calculado de um modo dinamico (nisso consiste este metodo de mini-batch dinamico)
    # o Nb vai ser calculado com Nb = ...ceil(T**iterador)
    #Return the ceiling of x as a float, the smallest integer value greater than or equal to x.
    Nb = math.ceil(Theta**iter)

    # outro dado imporntante é que o Nb terá de ser sempre o minimo entre o calcuo
    # Nb = min(Nb e o conjunto total dos dados)
    # Nt - Numero total de registos usados para treino
    Nb=min(Nb,Nt)

    #print("-----------")
    #print("Nb - iter  : ",iter)
    #print("Nt  : ",Nt)
    #print("Nb - valor calculado  : ",Nb)
    #print("posicao_actual: ", posicao_actual)
    #print("Nb + posicao_actual: ", Nb + posicao_actual)
    #print("-----------")

    # Depois disso teremos de alterar o get_data de modo aos dados serem escolhidos de uma forma sequencial e não de uma forma random
    # para isso teremos de saber sempre em que posição estamos actualmente

    #xt - conjunto de features/preditores de dados de treino
    #yt - conjunto de variaveis de targets/dependentes de dados de treino
    #Nb - Numero de registos que vão ser usadas para a amostra -  este valor está a ser calculado

    posicao_actual_w_Nb= Nb + posicao_actual



    #print(posicao_actual_w_Nb)
    #xb,yb = get_data_sequencial(xt,yt,Nt,Nb,posicao_actual,posicao_actual_w_Nb)

    xb, yb, posicao_actual = get_data_cycle(xt,yt, posicao_actual,Nb)

    #xb,yb=get_data(xt,yt,Nt,Nb)

    # calculo da direcao estocastica do subconjunto
    Gb=gradient(w,xb,yb,Nb)


    #alpha=step(w,xb,yb,g,Nb)


    #Nt - Numero total de registos usados para treino
    alpha=backtracking(w,xt,yt,Gb,Nt)

    w=w-alpha*Gb

    gT=gradient(w,xt,yt,Nt)
    Norm_gT=np.linalg.norm(gT)

    #print(w)
    #print(g)
    # paragem quando o iterador chegar ao valor 20000
    if(iter>5000):  ok=False;
    # Imprime-se o custo
    print(iter,cost(w,xb,yb,Nb))
    print("Norma Gt : ",Norm_gT)

    # Funcao de paragem, é para ser utilizada
    #1 * 10 ^ -8, or .00000001

    if(Norm_gT<1.e-8): ok=False

    iter=iter+1

print("======================================")
print("I={:d}".format(I))
print("coefficient",w)
print("Nt - numero total de registos usados para treino : ",Nt)
print("in error sample {:e}".format(cost(w,xt,yt,Nt)))
print("out error sample {:e}".format(cost(w,xv,yv,N-Nt)))

fig = plt.figure()
plt.plot(x,y,'ro')
s = np.linspace(0, 1,100, endpoint = True)
r = w[0]+w[1]*(s)+w[2]*(s*s)+w[3]*(s*s*s)+w[4]*(s*s*s*s)+w[5]*(s*s*s*s*s)
plt.plot(s, r, '-b', linewidth=3,label=r'$exato$')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Curve')
plt.show()
print('bye')
