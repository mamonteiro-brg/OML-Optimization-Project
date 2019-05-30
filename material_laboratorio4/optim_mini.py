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
 
def step(w,x,y,g,N):
    alpha=0.2
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
def get_data(xt,yt,Nt,Nb):
    xb=[];yb=[];
    for n in range(0,Nb):
        pos=math.ceil(np.random.rand()*(Nt-1))
        xb.append(xt[pos])
        yb.append(yt[pos])
    return xb,yb    
#=========== MAIN CODE ===============
# read the data file
x=[];y=[]; N=0
with open('P5-large_no.csv', 'r') as file:
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
while ok:
    xb,yb=get_data(xt,yt,Nt,Nb)
    g=gradient(w,xb,yb,Nb)
    Norm=np.linalg.norm(g)
    alpha=step(w,xb,yb,g,Nb)
    w=w-alpha*g
    #print(w)
    #print(g)
    if(iter>20000):  ok=False;
    print(iter,cost(w,xb,yb,Nb))
    if(Norm<1.e-8): ok=False
    iter=iter+1
print("======================================")
print("I={:d}".format(I))
print("coefficient",w)  
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
