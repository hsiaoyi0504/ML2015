
# coding: utf-8

# In[1]:

import numpy as np
import math

def logistic(s):
    return 1/(1+math.exp(s))

stepSize=0.01
iterations=2000
dimensions=21
x=[]
for i in range(dimensions):
    x.append([])
y=[]
with open('../hw3_train.dat') as f:
    for line in f:
        for i,value in enumerate(line.split()):
            if i<dimensions-1:
                x[i+1].append(float(value))
            else:
                y.append(int(value))
    f.close()
x[0]=np.ones(len(x[1]))
x=np.array(x)
N=len(x[1])


# In[2]:

w=np.zeros(dimensions)
for i in range(iterations):
    gradient=np.zeros(dimensions)
    for j in range(N):
        gradient+=logistic(y[j]*np.inner(w,x[:,j]))*(-y[j])*x[:,j]
    gradient/=N
    w-=stepSize*gradient


# In[3]:

print(w)


# In[4]:

x=[]
for i in range(dimensions):
    x.append([])
y=[]
with open('../hw3_test.dat') as f:
    for line in f:
        for i,value in enumerate(line.split()):
            if i<dimensions-1:
                x[i+1].append(float(value))
            else:
                y.append(int(value))
x[0]=np.ones(len(x[1]))
x=np.array(x)
N=len(x[1])
Eout=0
for i in range(N):
    if y[i]*np.inner(x[:,i],w)<=0:
        Eout+=1
Eout/=N
print("Eout:",Eout)

