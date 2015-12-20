
# coding: utf-8

# In[75]:

#python2
#get_ipython().magic(u'matplotlib inline')
from svmutil import *
import numpy as np
import matplotlib.pyplot as plt


# In[76]:

times=100
valSampleSize=1000


# In[77]:

x=[]
votes=[0]*5
with open('../features.train') as f:
    for line in f:
        for i,value in enumerate(line.split()):
            if i==0:
                if float(value)==0:
                    x.append(float(1))
                else:
                    x.append(float(-1))
            else:
                x.append(float(value))
    f.close()


# In[78]:

x=np.array(x)
x=np.reshape(x,(-1,3))
for iteration in range(times):
    np.random.shuffle(x)
    x_val=x[:1000,1:]
    y_val=x[:1000,0]
    x_train=x[1000:,1:]
    y_train=x[1000:,0]
    x_val=x_val.tolist()
    y_val=y_val.tolist()
    x_train=x_train.tolist()
    y_train=y_train.tolist()
    Eval=1
    tempIndex=0
    for i,gamma in enumerate([1,10,100,1000,10000]):
        prob=svm_problem(y_train,x_train)
        param=svm_parameter('-t 2 -g ' + str(gamma) +  ' -c 0.1')
        m=svm_train(prob,param)
        p_label, p_acc, p_val = svm_predict(y_val, x_val, m, '-b 0')
        if (100-p_acc[0])/100 < Eval:
            Eval=(100-p_acc[0])/100
            tempIndex=i
    votes[tempIndex]+=1


# In[83]:

gamma=[1, 10,100,1000,10000]
gamma=np.array(gamma)
fig=plt.figure()
ax=plt.gca()
ax.plot(gamma,votes,alpha=0.5,markeredgecolor='none')
ax.set_xscale('log')
plt.ylabel('Frequency')
plt.xlabel('gamma')


# In[81]:

# gamma=[1, 10,100,1000,10000]
# gamma=np.array(gamma)
# Eout=np.array(Eout)
# Eout=(100-Eout)/100
# fig=plt.figure()
# ax=plt.gca()
# ax.plot(gamma,Eout,'o',alpha=0.5,markeredgecolor='none')
# ax.set_xscale('log')
# plt.ylabel('Eout')
# plt.xlabel('gamma')

