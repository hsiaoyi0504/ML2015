
# coding: utf-8

# In[13]:

get_ipython().magic(u'matplotlib inline')
import random as rand
import numpy as np
import matplotlib.pyplot as plt
def sign(num):
    if num>0:
        return 1
    else:
        return -1
testTimes=1000
w0Record=[]
w1Record=[]
w2Record=[]
w3Record=[]
w4Record=[]
w5Record=[]
N=1000
for testTime in range(testTimes):
    #generate traning set
    x1=[]
    x2=[]
    x1x2=[]
    x1_square=[]
    x2_square=[]
    y=[]
    for i in range(N):
        x1.append(rand.uniform(-1,1))
        x2.append(rand.uniform(-1,1))
        x1x2.append(x1[i]*x2[i])
        x1_square.append(x1[i]**2)
        x2_square.append(x2[i]**2)
        isFlip=rand.uniform(0,1)
        if isFlip<=0.1:
            y.append(-sign( x1[i]**2 + x2[i]**2 - 0.6 ))
        else:
            y.append(sign(x1[i]**2 + x2[i]**2 - 0.6))
    #linear regression(training)
    x1=np.matrix(x1)
    x1=x1.reshape(N,1)
    x2=np.matrix(x2)
    x2=x2.reshape(N,1)
    x1x2=np.matrix(x1x2)
    x1x2=x1x2.reshape(N,1)
    x1_square=np.matrix(x1_square)
    x1_square=x1_square.reshape(N,1)
    x2_square=np.matrix(x2_square)
    x2_square=x2_square.reshape(N,1)
    temp=np.matrix(np.ones((N,1)))
    X=np.concatenate((temp,x1,x2,x1x2,x1_square,x2_square),axis=1)
    y=np.matrix(y)
    y=y.reshape(N,1)
    w=np.linalg.pinv(X)*y
    #testing
#     Ein=0
#     for i in range(N):
#         if y[i]!=sign(w[0]*1+w[1]*x1[i]+w[2]*x2[i]+w[3]*x1x2[i]+w[4]*x1_square[i]+w[5]*x2_square[i]):
#             Ein+=1
#     Ein=Ein/N
    w0Record.append(w.item(0))
    w1Record.append(w.item(1))
    w2Record.append(w.item(2))
    w3Record.append(w.item(3))
    w4Record.append(w.item(4))
    w5Record.append(w.item(5))
    
print("Average w0:",sum(w0Record)/testTimes)
print("Average w1:",sum(w1Record)/testTimes)
print("Average w2:",sum(w2Record)/testTimes)
print("Average w3:",sum(w3Record)/testTimes)
print("Average w4:",sum(w4Record)/testTimes)
print("Average w5:",sum(w5Record)/testTimes)


# In[14]:

plt.hist(w3Record)
plt.title("w3 distribution")
plt.xlabel("w3")
plt.ylabel("Frequency")
plt.show()


# In[ ]:



