
# coding: utf-8

# In[65]:

import numpy as np
import math as math


# In[66]:

dimensions=11
gamma_list=[32,2,0.125]
lambda_list=[0.001,1,1000]
train_example_num=400


# In[67]:

def sign(x):
    if x>0:
        return 1
    elif x<0:
        return -1


# In[68]:

x=[]
for i in range(dimensions-1):
    x.append([])
y=[]
with open('../hw2_lssvm_all.dat') as file:
    for line in file:
        for i,value in enumerate(line.split()):
            if i!=dimensions-1:
                x[i].append(float(value))
            else:
                y.append(int(value))
    file.close()


# In[69]:

x=np.array(x)
y=np.array(y)
x_train=x[:,:train_example_num]
y_train=y[:train_example_num]
x_test=x[:,train_example_num:]
y_test=y[train_example_num:]
K=np.zeros((train_example_num,train_example_num))


# In[70]:

for l in lambda_list:
    for g in gamma_list:
        #training
        for i in range(train_example_num):
            for j in range(train_example_num):
                temp=np.linalg.norm(x_train[:,i]-x_train[:,j])**2
                K[i][j]=math.exp(-g*temp)
        K=np.linalg.inv(l*np.identity(train_example_num)+K)
        beta=np.dot(K,y_train)
        #testing
        #in sample
        Ein=0
        for n in range(len(x_train[0])):
            weighting_sum=0
            for i in range(train_example_num):
                temp=np.linalg.norm(x_train[:,i]-x_train[:,n])**2
                weighting_sum+=beta[i]*math.exp(-g*temp)
            if sign(weighting_sum)!=y_train[n]:
                Ein+=1
        Ein/=len(x_train[0])
        print('gamma:',g,'lambda:',l,'Ein:',Ein)
        #out sample
        Eout=0
        for n in range(len(x_test[0])):
            weighting_sum=0
            for i in range(train_example_num):
                temp=np.linalg.norm(x_train[:,i]-x_test[:,n])**2
                weighting_sum+=beta[i]*math.exp(-g*temp)
            if sign(weighting_sum)!=y_test[n]:
                Eout+=1
        Eout/=len(x_test[0])
        print('gamma:',g,'lambda:',l,'Eout:',Eout)

