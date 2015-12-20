
# coding: utf-8

# In[6]:

#python2
#get_ipython().magic(u'matplotlib inline')
from svmutil import *
import numpy as np
import matplotlib.pyplot as plt
import math as math


# In[7]:

dimensions=3


# In[8]:

x=[]
y=[]
lengthW=[]
with open('../features.train') as f:
    for line in f:
        for i,value in enumerate(line.split()):
            if i==0:
                if float(value)==0:
                    y.append(float(1))
                else:
                    y.append(float(-1))
            else:
                x.append(float(value))
    f.close()


# In[9]:

x=np.array(x)
x=np.reshape(x,(-1,2))
x=x.tolist()
prob=svm_problem(y,x)

param=svm_parameter('-t 0 -c 0.000001')
m=svm_train(prob,param)
support_vector_coefficients=m.get_sv_coef()
support_vector=m.get_SV()
support_vector_coefficients=np.array(support_vector_coefficients)
for i in range(len(support_vector)):
    support_vector[i]= [support_vector[i][1], support_vector[i][2]]
support_vector=np.array(support_vector)
w=support_vector_coefficients*support_vector
length=[0,0]
for i in range(len(w)):
    length+=w[i]
lengthW.append(math.sqrt(length[0] ** 2 + length[1] ** 2))


param=svm_parameter('-t 0 -c 0.0001')
m=svm_train(prob,param)
support_vector_coefficients=m.get_sv_coef()
support_vector=m.get_SV()
support_vector_coefficients=np.array(support_vector_coefficients)
for i in range(len(support_vector)):
    support_vector[i]= [support_vector[i][1], support_vector[i][2]]
support_vector=np.array(support_vector)
w=support_vector_coefficients*support_vector
length=[0,0]
for i in range(len(w)):
    length+=w[i]
lengthW.append(math.sqrt(length[0] ** 2 + length[1] ** 2))


param=svm_parameter('-t 0 -c 0.01')
m=svm_train(prob,param)
support_vector_coefficients=m.get_sv_coef()
support_vector=m.get_SV()
support_vector_coefficients=np.array(support_vector_coefficients)
for i in range(len(support_vector)):
    support_vector[i]= [support_vector[i][1], support_vector[i][2]]
support_vector=np.array(support_vector)
w=support_vector_coefficients*support_vector
length=[0,0]
for i in range(len(w)):
    length+=w[i]
lengthW.append(math.sqrt(length[0] ** 2 + length[1] ** 2))

param=svm_parameter('-t 0 -c 1')
m=svm_train(prob,param)
support_vector_coefficients=m.get_sv_coef()
support_vector=m.get_SV()
support_vector_coefficients=np.array(support_vector_coefficients)
for i in range(len(support_vector)):
    support_vector[i]= [support_vector[i][1], support_vector[i][2]]
support_vector=np.array(support_vector)
w=support_vector_coefficients*support_vector

length=[0,0]
for i in range(len(w)):
    length+=w[i]
lengthW.append(math.sqrt(length[0] ** 2 + length[1] ** 2))


param=svm_parameter('-t 0 -c 100')
m=svm_train(prob,param)
support_vector_coefficients=m.get_sv_coef()
support_vector=m.get_SV()
support_vector_coefficients=np.array(support_vector_coefficients)
for i in range(len(support_vector)):
    support_vector[i]= [support_vector[i][1], support_vector[i][2]]
support_vector=np.array(support_vector)
w=support_vector_coefficients*support_vector

length=[0,0]
for i in range(len(w)):
    length+=w[i]
lengthW.append(math.sqrt(length[0] ** 2 + length[1] ** 2))


# In[10]:

c=[0.000001, 0.0001,0.01,1,100]
c=np.array(c)
fig=plt.figure()
ax=plt.gca()
ax.plot(c,lengthW,'o',alpha=0.5,markeredgecolor='none')
ax.set_xscale('log')
plt.ylabel('||w||')
plt.xlabel('C')

