
# coding: utf-8

# In[10]:

get_ipython().magic(u'matplotlib inline')
import random as rand
import matplotlib.pyplot as plt


# In[11]:

times=5000
size=20


# In[12]:

def sign(input):
    if input>=0:
        return 1;
    else:
        return -1;


# In[13]:

errorRatesRecord=[]
for i in range(times):
    #Generate the data
    x=[]
    y=[]
    for j in range(size):
        x.append(rand.uniform(-1,1))
        flip=rand.uniform(0,1)
        if x[j]<0:
            if flip>0.2:
                y.append(-1)
            else:
                y.append(1)
        else:
            if flip>0.2:
                y.append(1)
            else:
                y.append(-1)
    #Run the decision stump algorithm
    #print(x)
    x_sorted=sorted(x)
    x_median=[]
    for j in range(size-1):
        x_median.append((x_sorted[j]+x_sorted[j+1])/2)
    bestErrorTimes=20;
    bestS=0;
    bestTheta=0;
    errorTimes=0
    count1=0;
    for j in range(size):
        if y[j]==1:
            count1+=1
    if count1>size-count1:
        bestS=1
        bestTheta=-2
        bestErrorTimes=size-count1
    else:
        bestS=1
        bestTheta=2
        bestErrorTImes=count1
    for j in range(size-1):
        theta=x_median[j]
        for k in range(2):
            if k==0:
                s=1
            else:
                s=-1
            errorTimes=0
            for n in range(size):
                if s*sign(x[n]-theta)!=y[n]:
                    errorTimes+=1
            if errorTimes<bestErrorTimes:
                bestErrorTimes=errorTimes
                bestS=s
                bestTheta=theta
    #print("Ein:",bestErrorTimes/size)
    errorRatesRecord.append(0.5+0.3*bestS*(abs(bestTheta)-1))


# In[14]:

print("Average Eout:",sum(errorRatesRecord)/times)


# In[15]:

plt.hist(errorRatesRecord)
plt.title("Eout distribution")
plt.xlabel("Eout")
plt.ylabel("Frequency")
plt.show()

