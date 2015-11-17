
# coding: utf-8

# In[1]:

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
EoutRecord=[]
N=1000
#generating out-of-sample data
x1_o=[]
x2_o=[]
x1x2_o=[]
x1_square_o=[]
x2_square_o=[]
y_o=[]
for i in range(N):
    x1_o.append(rand.uniform(-1,1))
    x2_o.append(rand.uniform(-1,1))
    x1x2_o.append(x1_o[i]*x2_o[i])
    x1_square_o.append(x1_o[i]**2)
    x2_square_o.append(x2_o[i]**2)
    isFlip=rand.uniform(0,1)
    if isFlip<=0.1:
        y_o.append(-sign( x1_o[i]**2 + x2_o[i]**2 - 0.6 ))
    else:
        y_o.append(sign(x1_o[i]**2 + x2_o[i]**2 - 0.6))
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
    #testing(Eout)
    Eout=0
    for i in range(N):
        if y_o[i]*(w[0]*1 + w[1]*x1_o[i] + w[2]*x2_o[i] + w[3]*x1x2_o[i] + w[4]*x1_square_o[i]+ w[5]*x2_square_o[i] )<=0:
            Eout+=1
    Eout=Eout/N
    EoutRecord.append(Eout)

print("Average Eout:",sum(EoutRecord)/testTimes)


# In[2]:

plt.hist(EoutRecord)
plt.title("Eout distribution")
plt.xlabel("Eout")
plt.ylabel("Frequency")
plt.show()

