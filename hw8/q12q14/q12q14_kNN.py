
# coding: utf-8

# In[1]:

# %matplotlib inline
import matplotlib.pyplot as plt


# In[2]:

def sign(x):
    if x>0:
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

def kNN_predict(x_train,y_train,x_test,num_k):
    y_test_predict=[]
    big_num=100000
    for i in range(len(x_test)):
        kNN_distance=[big_num]*num_k
        kNN_index=[0]*num_k
        distance_max=big_num
        distance_max_index=0
        for j in range(len(x_train)):
            temp = [a - b for a, b in zip(x_test[i], x_train[j])]
            distance=sum([ i*i for i in temp])
            if distance<distance_max:
                temp=kNN_distance.index(distance_max)
                kNN_distance[temp]=distance
                kNN_index[temp]=j
                distance_max=max(kNN_distance)
                distance_max_index=kNN_distance.index(distance_max)
        predict=0
        for k in range(len(kNN_index)):
            predict+=y_train[kNN_index[k]]
        y_test_predict.append(sign(predict))
    return y_test_predict


# In[6]:

x_train, y_train = read_data("../hw8_train.dat",10)
x_test, y_test = read_data("../hw8_test.dat",10)


# In[7]:

k_list=[1,3,5,7,9]


# In[8]:

Ein_record=[]
Eout_record=[]
for k in k_list:
    y_train_predict=kNN_predict(x_train,y_train,x_train,k)
    Ein=E_01(y_train_predict,y_train)
    Ein_record.append(Ein)
    y_test_predict=kNN_predict(x_train,y_train,x_test,k)
    Eout=E_01(y_test_predict,y_test)
    Eout_record.append(Eout)


# In[9]:

print(Ein_record)


# In[10]:

print(Eout_record)


# In[13]:

plt.plot(k_list,Ein_record,"o")
plt.ylabel("Ein")
plt.xlabel("k")


# In[14]:

plt.plot(k_list,Eout_record,"o")
plt.ylabel("Eout")
plt.xlabel("k")


# In[ ]:



