#!python
'''
perceptron_v1
'''

import numpy as np
import matplotlib.pyplot as plt
import math
import sys

#========= plotting stuff ==================
def plot_data(X,Y,N):
    for n in range(N):
        if(Y[n]>0):
            plt.scatter(X[n,0], X[n,1], color='red', marker='o',s=50)
        else:
            plt.scatter(X[n,0], X[n,1], color='blue', marker='s',s=50)       
    plt.show()
    return

def plot_error(err):
    plt.plot(range(len(err)), err, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of misclassifications')
    plt.ylim([0,1])
    plt.show()
    return 

def plot_decision_regions(X,Y,N,ew):
    x1_min = X[:,0].min();x1_max = X[:,0].max() 
    x1=np.arange(x1_min,x1_max+0.1,0.1)
    x2=np.arange(x1_min,x1_max+0.1,0.1)
    for n in range(len(x1)):
        x2[n]=-(ew[0]+ew[1]*x1[n])/ew[2]
    x2_min = min(x2.min(),X[:,1].min());x2_max = max(x2.max(),X[:,1].max()) 
    plt.fill_between(x1, x2,x2_max, facecolor='red', alpha=0.4, interpolate=True)
    plt.fill_between(x1, x2_min,x2, facecolor='blue', alpha=0.4, interpolate=True)
    for n in range(N):
        if(Y[n]>0):
            plt.scatter(X[n,0], X[n,1], color='red', marker='o',s=50)
        else:
            plt.scatter(X[n,0], X[n,1], color='blue', marker='s',s=50) 
    plt.plot(x1,x2, color='black')      
    plt.show()
    return
# =============== Perceptron stuff ===================

def predictor(x,ew):
    s=ew[0];
    s=s+np.dot(x,ew[1:])
    sgn=np.sign(s)
    return sgn

def cost(X,Y,N,ew):
    En=0;
    for n in range(N):
        En=En+np.abs(0.5*(predictor(X[n],ew)-Y[n]))
    En=En/N
    return En

def update(x,y,eta,ew):
    s=0.5*(y-predictor(x,ew))
    s=s*eta
    ew[0]=ew[0]+s
    ew[1:]=ew[1:]+s*x
    return ew

def run_epoch(X,Y,N,eta,MAX_EPOCH,ew,err):
    #epsi=2*(1/N);
    epsi=0
    nb_epoch=0;
    while(err[-1]>epsi):
        nb_epoch=nb_epoch+1
        if(nb_epoch>MAX_EPOCH): break
        for n in range(N):
            ew=update(X[n],Y[n],eta,ew)  
            err.append(cost(X,Y,N,ew))
    return ew, err    

def run_stocastic(X,Y,N,eta,MAX_ITER,ew,err):
    epsi=0;
    it=0
    while(err[-1]>epsi):
        n=int(np.random.rand()*N)
        ew=update(X[n],Y[n],eta,ew)  
        err.append(cost(X,Y,N,ew))
        it=it+1
        if(it>MAX_ITER): break
    return ew, err


#=========== MAIN CODE ===============
#data=np.array([[0 ,0 ,-1],[0 ,1 ,-1],[1, 0 ,-1],[1, 1 ,1]])  #AND
#data=np.array([[0 ,0 ,-1],[0 ,1 ,1],[1, 0 ,1],[1, 1 ,1]])  #OR
#data=np.array([[0 ,0 ,-1],[0 ,1 ,1],[1, 0 ,1],[1, 1 ,-1]])  #XOR
#data=np.array([[0,0,-1],[0,1,1],[1,0,-1],[-1,1,-1],[0.5,0.5,1],[1,1.5,1],[1,1,1],[-1,0.5,-1],[-0.5,1.5,1],[-0.4,0.6,-1],[-1,0,-1]])
data=np.array([[0,0,-1],[0,1,1],[1,0,-1],[-1,1,-1],[0.5,0.5,1],[1,1.5,1],[1,1,1],[-1,0.5,-1],[-0.5,1.5,1],[-0.4,0.6,-1],[-1,0,-1],[0.3,0.2,1]])

#---------- Training -----
N,I=data.shape;I=I-1
X=data[:,:-1];Y=data[:,-1]
plot_data(X,Y,N)

ew=np.ones([I+1])
MAX_EPOCH=10
eta=0.4 # learning rate
err=[];err.append(cost(X,Y,N,ew))
#ew,err=run_epoch(X,Y,N,eta,MAX_EPOCH,ew,err)
ew,err=run_stocastic(X,Y,N,eta,250,ew,err)
#print(ew) 
plot_error(err)
plot_decision_regions(X,Y,N,ew)
print('in-samples error=%f ' % (cost(X,Y,N,ew)))

#--------- evaluating -----------
data_eval=np.array([[0.1,-0.1,-1],[0.2,0.8,1],[1.1,-0.1,-1],[-1.3,0.9,-1],[0.6,0.5,1],[0.9,1.4,1],[1.1,1.5,1],[-1.3,0.8,-1],[-0.4,1.2,1],[-0.3,0.8,-1],[-1,0.4,-1],[0.35,0.27,1]])
Ne,Ie=data_eval.shape;
Xe=data_eval[:,:-1];Ye=data_eval[:,-1];
plot_decision_regions(Xe,Ye,Ne,ew)
print('out-samples error=%f' % (cost(Xe,Ye,Ne,ew)))
print('bye')
