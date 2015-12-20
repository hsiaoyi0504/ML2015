
# coding: utf-8

# In[84]:

#python2
#get_ipython().magic(u'matplotlib inline')
from svmutil import *
import numpy as np
import matplotlib.pyplot as plt


# In[85]:

dimensions=3


# In[86]:

x=[]
y=[]
Ein=[]
sumAlpha=[]
with open('../features.train') as f:
    for line in f:
        for i,value in enumerate(line.split()):
            if i==0:
                if float(value)==8:
                    y.append(float(1))
                else:
                    y.append(float(-1))
            else:
                x.append(float(value))
    f.close()


# In[87]:

x=np.array(x)
x=np.reshape(x,(-1,2))
x=x.tolist()
prob=svm_problem(y,x)

param=svm_parameter('-t 1 -g 1 -d 2 -r 1 -c 0.000001')
m=svm_train(prob,param)
support_vector_coefficients=m.get_sv_coef()
p_label, p_acc, p_val = svm_predict(y, x, m, '-b 0')
Ein.append(p_acc[0])
support_vector_coefficients=np.array(support_vector_coefficients)
sumAlpha.append(sum(abs(support_vector_coefficients)))

param=svm_parameter('-t 1 -g 1 -d 2 -r 1 -c 0.0001')
m=svm_train(prob,param)
support_vector_coefficients=m.get_sv_coef()
p_label, p_acc, p_val = svm_predict(y, x, m, '-b 0')
Ein.append(p_acc[0])
support_vector_coefficients=np.array(support_vector_coefficients)
sumAlpha.append(sum(abs(support_vector_coefficients)))

param=svm_parameter('-t 1 -g 1 -d 2 -r 1 -c 0.01')
m=svm_train(prob,param)
support_vector_coefficients=m.get_sv_coef()
p_label, p_acc, p_val = svm_predict(y, x, m, '-b 0')
Ein.append(p_acc[0])
support_vector_coefficients=np.array(support_vector_coefficients)
sumAlpha.append(sum(abs(support_vector_coefficients)))

param=svm_parameter('-t 1 -g 1 -d 2 -r 1 -c 1')
m=svm_train(prob,param)
support_vector_coefficients=m.get_sv_coef()
p_label, p_acc, p_val = svm_predict(y, x, m, '-b 0')
Ein.append(p_acc[0])
support_vector_coefficients=np.array(support_vector_coefficients)
sumAlpha.append(sum(abs(support_vector_coefficients)))

param=svm_parameter('-t 1 -g 1 -d 2 -r 1 -c 100')
m=svm_train(prob,param)
support_vector_coefficients=m.get_sv_coef()
p_label, p_acc, p_val = svm_predict(y, x, m, '-b 0')
Ein.append(p_acc[0])
support_vector_coefficients=np.array(support_vector_coefficients)
sumAlpha.append(sum(abs(support_vector_coefficients)))


# In[88]:

c=[0.000001, 0.0001,0.01,1,100]
c=np.array(c)
Ein=np.array(Ein)
Ein=(100-Ein)/100
fig=plt.figure()
ax=plt.gca()
ax.plot(c,Ein,'o',alpha=0.5,markeredgecolor='none')
ax.set_xscale('log')
plt.ylabel('Ein')
plt.xlabel('C')


# In[91]:

c=[0.000001, 0.0001,0.01,1,100]
c=np.array(c)
fig=plt.figure()
ax=plt.gca()
ax.plot(c,sumAlpha,'o',alpha=0.5,markeredgecolor='none')
ax.set_xscale('log')
plt.ylabel('sum(alpha_n)')
plt.xlabel('C')


# In[93]:

print sumAlpha

