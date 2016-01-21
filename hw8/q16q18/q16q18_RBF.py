
# coding: utf-8

# In[1]:

# %matplotlib inline
import matplotlib.pyplot as plt
import math as math


# In[2]:

def sign(x):
    if x>=0:
        return 1
    else:
        return -1


# In[3]:

def E_01(y_predict,y):
    temp=[1/2*abs(a-b) for a,b in zip(y_predict,y)]
    return sum(temp)/len(y)


# In[4]:

def read_data(path,dimensions):
    x=[]
    y=[]
    with open(path) as file:
        for line in file:
            x.append([])
            for i,value in enumerate(line.split()):
                if i!=dimensions-1:
                    x[len(x)-1].append(float(value))
                else:
                    y.append(int(value))
        file.close()
    return (x,y)


# In[5]:

def RBF_predict(x_train,y_train,x_test,gamma):
    y_test_predict=[]
    for i in range(len(x_test)):
        predict=0
        for j in range(len(x_train)):
            temp = [a - b for a, b in zip(x_test[i], x_train[j])]
            predict+=y_train[j]*math.exp(-gamma*sum([ i*i for i in temp]))
        y_test_predict.append(sign(predict))
    return y_test_predict


# In[6]:

x_train, y_train = read_data("../hw8_train.dat",10)
x_test, y_test = read_data("../hw8_test.dat",10)


# In[7]:

gamma_list=list(range(-3,3))
gamma_list=[10**i for i in gamma_list]


# In[8]:

Ein_record=[]
Eout_record=[]
for gamma in gamma_list:
    y_train_predict=RBF_predict(x_train,y_train,x_train,gamma)
    Ein=E_01(y_train_predict,y_train)
    Ein_record.append(Ein)
    y_test_predict=RBF_predict(x_train,y_train,x_test,gamma)
    Eout=E_01(y_test_predict,y_test)
    Eout_record.append(Eout)


# In[9]:

print(Ein_record)


# In[10]:

print(Eout_record)


# In[11]:

plt.plot(gamma_list, Ein_record,"o")
plt.ylabel("Ein")
plt.xlabel("gamma")
plt.xscale("log")


# In[12]:

plt.plot(gamma_list, Eout_record,"o")
plt.ylabel("Eout")
plt.xlabel("gamma")
plt.xscale("log")

