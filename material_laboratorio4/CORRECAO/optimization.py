#!python
'''
optimization algorithms
'''
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

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
        val=polynomial(w,x[n])-y[n]
        phi=np.power(x[n],pw)
        grad=grad+phi*val
    return grad/N    
''' 
def step(w,x,y,g,N):
    alpha=1
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

#=========== MAIN CODE ===============
# read the data file
x=[];y=[]; N=0
with open('P3.csv', 'r') as file:
    reader = csv.reader(file)
    for u in reader:
        x=np.append(x,float(u[0]))
        y=np.append(y,float(u[1]))
        N=N+1
I=3;    
#print(N)
#fig = plt.figure()
#plt.plot(x,y,'ro')
#plt.show() 

w=np.array([1]*(I+1));
#w=np.array([1/5,-1/2,1])
print(cost(w,x,y,N))
print(gradient(w,x,y,N))

ok=True; iter=0
while ok:
    g=gradient(w,x,y,N)
    Norm=np.linalg.norm(g)
    alpha=step(w,x,y,g,N)
    w=w-0.9*alpha*g
    #print(w)
    #print(g)
    if(iter>5000):  ok=False;
    print(iter,cost(w,x,y,N))
    if(Norm<1.e-6): ok=False
    iter=iter+1
print("======================================")
print("I={:d}".format(I))
print("coefficient",w)  
print("in error sample {:e}".format(cost(w,x,y,N)))
  
fig = plt.figure()
plt.plot(x,y,'ro')
s = np.linspace(0, 1,100, endpoint = True)
#r = w[0]+w[1]*(s)+w[2]*(s*s)+w[3]*(s*s*s)+w[4]*(s*s*s*s)+w[5]*(s*s*s*s*s)
#r = w[0]+w[1]*(s)+w[2]*(s*s)+w[3]*(s*s*s)+w[4]*(s*s*s*s)
r = w[0]+w[1]*(s)+w[2]*(s*s)+w[3]*(s*s*s)
#r = w[0]+w[1]*(s)+w[2]*(s*s)
plt.plot(s, r, '-b', linewidth=3,label=r'$exato$')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Curve')
plt.show()
print('bye')
