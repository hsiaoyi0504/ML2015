
# coding: utf-8

# In[3]:

import numpy as np
import math

def logistic(s):
    return 1/(1+math.exp(s))

stepSize=0.001
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


# In[4]:

w=np.zeros(dimensions)
num=0
for i in range(iterations):
    gradient=logistic(y[num]*np.inner(w,x[:,num]))*(-y[num])*x[:,num]
    w-=stepSize*gradient
    num+=1
    if num>=N:
        num-=N


# In[5]:

print(w)


# In[6]:

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

