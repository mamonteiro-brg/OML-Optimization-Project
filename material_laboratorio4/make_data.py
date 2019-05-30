#!python
'''
optimization algorithms
'''
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import sys

def poly(val,a):
    d=0
    factor=1.0
    result=0
    for coef in a:
        result=result+coef*factor
        factor=factor*val
        d=d+1
    return(result)
#=========== MAIN CODE ===============
N=int(input('Data size:'))
I=int(input('polynomial degree:'))
alpha=float(input('perturbation factor in percent:'))*0.01
a_ref=[1/5,-1/2,1,-2/3,4,-5,1,1,1,1];a=a_ref[0:I+1]  
print(a)
x=np.random.rand(N,1);y=[];
for n in range(0,N):
    scale=(np.random.rand()-0.5)*alpha
    y.append(poly(x[n],a)*(1+scale))
 
fig = plt.figure()
plt.plot(x,y,'ro')
plt.show() 

with open('polynomial.csv', mode='w') as poly_file:
    poly_writer = csv.writer(poly_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for n in range(0,N):
        poly_writer.writerow([x[n][0], y[n][0]])

    
    
print('bye')

'''
# write the data file
with open('dados.txt', 'r') as file:
    reader = csv.reader(file)
    all_data = list(reader)    
    data=all_data
    
# give the set of attribute and values  
fig = plt.figure()
ax.scatter(B[:,0],B[:,1], B[:,2], c='b', marker='.')
ax.scatter(xe, ye, ze, c='r', marker='o')
plt.show()
'''
 
