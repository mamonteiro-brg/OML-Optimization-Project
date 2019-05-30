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


def cost(X,Y,N,ew):


def update(x,y,eta,ew):


def run_epoch(X,Y,N,eta,MAX_EPOCH,ew,err):


def run_stocastic(X,Y,N,eta,MAX_ITER,ew,err):


#=========== MAIN CODE ===============
# read the data file
N,n_row,n_col,data=read_asc_data('./dataset/line1500.txt')
#N,n_row,n_col,data=read_asc_data('./dataset/my_digit.txt');np.place(data[:,-1], data[:,-1]!=1, [-1])
print('find %d images of %d X %d pixels' % (N,n_row,n_col))
plot_data(10,10,n_row,n_col,data)

#Nt=int(N*0.8);
#I=n_row*n_col;
#Xt=data[:Nt,:-1];Yt=data[:Nt,-1]
#np.place(Yt, Yt!=1, [-1])
#ew=np.ones([I+1])
#err=[];err.append(cost(Xt,Yt,Nt,ew));
#print('cost=%f ' % (err[-1]))

#---------------training --------------
#ew,err=run_stocastic(Xt,Yt,Nt,0.1,1000,ew,err)
#ew,err=run_epoch(Xt,Yt,Nt,0.1,5,ew,err)
#print('cost=%f ' % (err[-1]))
#plot_error(err)
#plot_tagged_data(10,10,n_row,n_col,Xt,Yt,ew) 
#print('in-samples error=%f ' % (cost(Xt,Yt,Nt,ew)))

#---- evaluated -----------------
#Ne=N-Nt;
#Xe=data[Nt:N,:-1];Ye=data[Nt:N,-1];
#print('out-samples error=%f' % (cost(Xe,Ye,Ne,ew)))
#plot_tagged_data(10,10,n_row,n_col,Xe,Ye,ew)
print('bye')

# to improve (1) normalization of omega, (2) eta with respect to inverse square distance