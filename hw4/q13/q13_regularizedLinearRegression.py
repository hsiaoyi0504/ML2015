
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

def sign(num):
    if num>0:
        return 1
    else:
        return -1


# In[3]:

dimensions=2
regularizationWeighting=11.26
x=[]
for i in range(dimensions+1):
    x.append([])
y=[]


# In[4]:

#read training data
with open('../hw4_train.dat') as f:
    for line in f:
        for i,value in enumerate(line.split()):
            if i<dimensions:
                x[i+1].append(float(value))
            else:
                y.append(int(value))
    f.close()
x[0]=np.ones(len(x[1]))


# In[5]:

#training
x=np.matrix(x)
x=np.transpose(x)
y=np.matrix(y)
y=np.transpose(y)
w=np.linalg.inv(np.transpose(x)*x+regularizationWeighting*np.identity(dimensions+1)) * np.transpose(x) * y


# In[6]:

#read testing data
x2=[]
for i in range(dimensions+1):
    x2.append([])
y2=[]
with open('../hw4_test.dat') as f:
    for line in f:
        for i,value in enumerate(line.split()):
            if i<dimensions:
                x2[i+1].append(float(value))
            else:
                y2.append(int(value))
    f.close()
x2[0]=np.ones(len(x2[1]))


# In[7]:

#testing
x=np.transpose(x)
x=np.array(x)
Ein=0
for i in range(len(x[2])):
    result=0
    for j in range(dimensions+1):
        result+=w[j]*x[j][i]
    if sign(result) !=  y[i]:
        Ein+=1
Ein/=len(x[2])
Eout=0
for i in range(len(x2[2])):
    result=0
    for j in range(dimensions+1):
        result+=w[j]*x2[j][i]
    if sign(result) != y2[i]:
        Eout+=1
Eout/=len(x2[2])


# In[8]:

print("Ein:",Ein)
print("Eout:",Eout)

