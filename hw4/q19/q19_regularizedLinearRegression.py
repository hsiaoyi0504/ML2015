
# coding: utf-8

# In[35]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# In[36]:

def sign(num):
    if num>0:
        return 1
    else:
        return -1


# In[37]:

dimensions=2
regularizationWeighting=np.arange(-10,3,1,dtype=float)
regularizationWeighting=10 ** regularizationWeighting
x=[]
for i in range(dimensions+1):
    x.append([])
y=[]


# In[38]:

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
x=np.array(x)
# x_train=x[:,0:120]
# y_train=y[0:120]
# x_val=x[:,120:]
# y_val=y[120:]


# In[39]:

EcvRecord=[]


# In[40]:

for k in range(len(regularizationWeighting)):
    Ecv=0
    for i in range(5):
        #training
        x_train=x[:,0:i*40]
        x_train=np.concatenate((x_train,x[:,(i+1)*40:]),1)
        x_train=np.matrix(x_train)
        y_train=y[0:i*40]
        y_train=np.concatenate((y_train,y[(i+1)*40:]))
        y_train=np.matrix(y_train)
        x_train=np.transpose(x_train)
        y_train=np.transpose(y_train)
        x_val=x[:,i*40:(i+1)*40]
        y_val=y[i*40:(i+1)*40]
        w=np.linalg.inv(np.transpose(x_train)*x_train+regularizationWeighting[k]*np.identity(dimensions+1))* np.transpose(x_train) * y_train
        #validation
        for i in range(len(x_val[2])):
            result=0
            for j in range(dimensions+1):
                result+=w[j]*x_val[j][i]
            if sign(result) !=  y_val[i]:
                Ecv+=1
    Ecv/=len(x[2])
    EcvRecord.append(Ecv)


# In[41]:

print("Ecv:",EcvRecord)


# In[42]:

lineEcv,=plt.semilogx(regularizationWeighting, EcvRecord)
plt.title("Ecv vs lambda")
plt.legend([lineEcv], ["Ecv"])
plt.xlabel("Lambda")

