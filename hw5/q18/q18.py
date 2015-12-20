
# coding: utf-8

# In[24]:

#python2
#get_ipython().magic(u'matplotlib inline')
from svmutil import *
import numpy as np
import matplotlib.pyplot as plt
import math as math


# In[25]:

x=[]
y=[]
dist=[]
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


# In[26]:

x=np.array(x)
x=np.reshape(x,(-1,2))
x=x.tolist()
prob=svm_problem(y,x)

param=svm_parameter('-t 2 -g 100 -c 0.001')
m=svm_train(prob,param)
support_vector_coefficients=m.get_sv_coef()
support_vector_coefficients=np.array(support_vector_coefficients)
dist.append(1/math.sqrt(2*(-2.380633+sum(abs(support_vector_coefficients)))))

param=svm_parameter('-t 2 -g 100 -c 0.01')
m=svm_train(prob,param)
support_vector_coefficients=m.get_sv_coef()
support_vector_coefficients=np.array(support_vector_coefficients)
dist.append(1/math.sqrt(2*(-23.144993+sum(abs(support_vector_coefficients)))))

param=svm_parameter('-t 2 -g 100 -c 0.1')
m=svm_train(prob,param)
support_vector_coefficients=m.get_sv_coef()
support_vector_coefficients=np.array(support_vector_coefficients)
dist.append(1/math.sqrt(2*(-178.198592+sum(abs(support_vector_coefficients)))))

param=svm_parameter('-t 2 -g 100 -c 1')
m=svm_train(prob,param)
support_vector_coefficients=m.get_sv_coef()
support_vector_coefficients=np.array(support_vector_coefficients)
dist.append(1/math.sqrt(2*(-1401.258805+sum(abs(support_vector_coefficients)))))

param=svm_parameter('-t 2 -g 100 -c 10')
m=svm_train(prob,param)
support_vector_coefficients=m.get_sv_coef()
support_vector_coefficients=np.array(support_vector_coefficients)
dist.append(1/math.sqrt(2*(-13027.302689+sum(abs(support_vector_coefficients)))))


# In[27]:

c=[0.001, 0.01,0.1,1,10]
c=np.array(c)
fig=plt.figure()
ax=plt.gca()
ax.plot(c,dist,'o',alpha=0.5,markeredgecolor='none')
ax.set_xscale('log')
plt.ylabel('Distance')
plt.xlabel('C')

