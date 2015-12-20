
# coding: utf-8

# In[17]:

#python2
#get_ipython().magic(u'matplotlib inline')
from svmutil import *
import numpy as np
import matplotlib.pyplot as plt


# In[18]:

dimensions=3


# In[19]:

x=[]
y=[]
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


# In[20]:

x2=[]
y2=[]
Eout=[]
with open('../features.test') as f:
    for line in f:
        for i,value in enumerate(line.split()):
            if i==0:
                if float(value)==0:
                    y2.append(float(1))
                else:
                    y2.append(float(-1))
            else:
                x2.append(float(value))
    f.close()


# In[21]:

x=np.array(x)
x=np.reshape(x,(-1,2))
x=x.tolist()
x2=np.array(x2)
x2=np.reshape(x2,(-1,2))
x2=x2.tolist()

prob=svm_problem(y,x)

param=svm_parameter('-t 2 -g 1 -c 0.1')
m=svm_train(prob,param)
p_label, p_acc, p_val = svm_predict(y2, x2, m, '-b 0')
Eout.append(p_acc[0])


param=svm_parameter('-t 2 -g 10 -c 0.1')
m=svm_train(prob,param)
p_label, p_acc, p_val = svm_predict(y2, x2, m, '-b 0')
Eout.append(p_acc[0])

param=svm_parameter('-t 2 -g 100 -c 0.1')
m=svm_train(prob,param)
p_label, p_acc, p_val = svm_predict(y2, x2, m, '-b 0')
Eout.append(p_acc[0])

param=svm_parameter('-t 2 -g 1000 -c 0.1')
m=svm_train(prob,param)
p_label, p_acc, p_val = svm_predict(y2, x2, m, '-b 0')
Eout.append(p_acc[0])

param=svm_parameter('-t 2 -g 10000 -c 0.1')
m=svm_train(prob,param)
p_label, p_acc, p_val = svm_predict(y2, x2, m, '-b 0')
Eout.append(p_acc[0])


# In[22]:

gamma=[1, 10,100,1000,10000]
gamma=np.array(gamma)
Eout=np.array(Eout)
Eout=(100-Eout)/100
fig=plt.figure()
ax=plt.gca()
ax.plot(gamma,Eout,'o',alpha=0.5,markeredgecolor='none')
ax.set_xscale('log')
plt.ylabel('Eout')
plt.xlabel('gamma')

