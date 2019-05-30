#!python
'''
perceptron
'''
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import sys


# ============ FILE load and write stuff ===========================
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def read_asc_data(filename):    
    f= open(filename,'r') 
    tmp_str=f.readline()
    tmp_arr=tmp_str[:-1].split(' ')
    N=int(tmp_arr[0]);n_row=int(tmp_arr[1]);n_col=int(tmp_arr[2])
    data=np.zeros([N,n_row*n_col+1])
    for n in range(N):
        tmp_str=f.readline()
        tmp_arr=tmp_str[:-1].split(' ')       
        for i in range(n_row*n_col+1):
            data[n][i]=int(tmp_arr[i])
    f.close() 
    return N,n_row,n_col,data

def plot_data(row,col,n_row,n_col,data):
    fig=plt.figure(figsize=(row,col))
    for n in range(1, row*col +1):
        img=np.reshape(data[n-1][:-1],(n_row,n_col))
        fig.add_subplot(row, col, n)
        plt.imshow(img,interpolation='none',cmap='binary')
    plt.show()
    
def plot_tagged_data(row,col,n_row,n_col,X,Y,ew): 
    fig=plt.figure(figsize=(row,col))
    for n in range(row*col):
        img=np.reshape(X[n],(n_row,n_col))
        fig.add_subplot(row, col, n+1)
        #if(Y[n]>0):#exact case
        if(predictor(X[n],ew)>0):
            plt.imshow(img,interpolation='none',cmap='RdPu')
        else:
            plt.imshow(img,interpolation='none',cmap='cool')               
    plt.show()
    
def plot_error(err):
    plt.plot(range(len(err)), err, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of misclassifications')
    plt.ylim([0,1])
    plt.show()
    return 


#============== Perceptron Stuff ==================

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
    ew=ew/np.linalg.norm(ew,1)
    return ew

def run_epoch(X,Y,N,eta,MAX_EPOCH,ew,err):
    epsi=0;
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
        r=ew[0];r=r+np.dot(X[n],ew[1:])
        # para line 1500
        theta=math.exp(-it/1000)
        new_eta=eta*theta/(1+100*r*r)
        # para digit
        #theta=math.exp(-it/1600)
        #new_eta=eta*theta/(1+10*r*r)
        ew=update(X[n],Y[n],new_eta,ew)  
        err.append(cost(X,Y,N,ew))
        print('iter %d, cost=%f, eta=%e \r' % (it,err[-1],new_eta),end='')
        it=it+1    
        if(it>MAX_ITER): break
    return ew, err

#=========== MAIN CODE ===============
# read the data file
N,n_row,n_col,data=read_asc_data('./dataset/line1500.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/my_digit.txt');np.place(data[:,-1], data[:,-1]!=1, [-1])
print('find %d images of %d X %d pixels' % (N,n_row,n_col))

plot_data(10,10,n_row,n_col,data)

Nt=int(N*0.8);
I=n_row*n_col;
Xt=data[:Nt,:-1];Yt=data[:Nt,-1]
np.place(Yt, Yt!=1, [-1])
ew=np.ones([I+1])
err=[];err.append(cost(Xt,Yt,Nt,ew));
print('cost=%f ' % (err[-1]))
'''
MAX_EPOCH=10;MAX_ITER=2000;
eta=0.1 # learning rate
#ew,err=run_stocastic(Xt,Yt,Nt,0.1,1000,ew,err)
ew,err=run_epoch(Xt,Yt,Nt,0.1,5,ew,err)
print('cost=%f ' % (err[-1]))
#plot_error(err)
#ew,err=run_stocastic(Xt,Yt,Nt,0.01,2000,ew,err)
ew,err=run_epoch(Xt,Yt,Nt,0.03,5,ew,err)
print('cost=%f ' % (err[-1]))
#plot_error(err)
#ew,err=run_stocastic(Xt,Yt,Nt,0.001,4000,ew,err)
ew,err=run_epoch(Xt,Yt,Nt,0.01,5,ew,err)
print('cost=%f ' % (err[-1]))
#plot_error(err)
#ew,err=run_stocastic(Xt,Yt,Nt,0.0001,5000,ew,err)
ew,err=run_epoch(Xt,Yt,Nt,0.0003,5,ew,err)
print('cost=%f ' % (err[-1]))
#plot_error(err)
ew,err=run_stocastic(Xt,Yt,Nt,0.00001,5000,ew,err)
#ew,err=run_epoch(Xt,Yt,Nt,0.00002,15,ew,err)
plot_error(err)
#ew,err=run_epoch(Xt,Yt,Nt,0.00001,20,ew,err)
#plot_error(err)
#plot_tagged_data(10,10,n_row,n_col,Xt,Yt,ew) 
'''
ew,err=run_stocastic(Xt,Yt,Nt,0.1,19999,ew,err)
plot_error(err)
print('in-samples error=%f ' % (cost(Xt,Yt,Nt,ew)))

Ne=N-Nt;
Xe=data[Nt:N,:-1];Ye=data[Nt:N,-1];
np.place(Ye, Ye!=1, [-1])
print('out-samples error=%f' % (cost(Xe,Ye,Ne,ew)))
plot_tagged_data(10,10,n_row,n_col,Xe,Ye,ew)
print('bye')

# to improve (1) normalizationof omega, (2) eta with respect to inverse square distance