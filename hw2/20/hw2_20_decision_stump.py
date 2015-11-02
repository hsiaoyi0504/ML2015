
# coding: utf-8

# In[1]:

def sign(input):
    if input>=0:
        return 1;
    else:
        return -1;


# In[2]:

dimensions=9
size=100
size2=1000


# In[3]:

x=[]
x2=[]
for i in range(dimensions):
    x.append([])
    x2.append([])
y=[]
y2=[]


# In[4]:

#read training data
with open('../hw2_train.dat') as f:
    for line in f:
        for i,value in enumerate(line.split()):
            if i<9:
                x[i].append(float(value))
            else:
                y.append(int(value))
    f.close()


# In[5]:

bestErrorTimes=size
bestS=0
bestTheta=0
count1=0
bestDimension=0
for j in range(size):
    if y[j]==1:
        count1+=1
for i in range(dimensions):
    x_sorted=sorted(x[i])
    x_median=[]
    for j in range(size-1):
        x_median.append((x_sorted[j]+x_sorted[j+1])/2)
    if count1>size-count1:
        if size-count1<bestErrorTimes:
            bestS=1
            bestTheta=x_sorted[0]-1
            bestErrorTimes=size-count1
            bestDimension=i
    else:
        if count1<bestErrorTimes:
            bestS=1
            bestTheta=x_sorted[-1]+1
            bestErrorTimes=count1
            bestDimension=i
    for j in range(size-1):
        theta=x_median[j]
        for k in range(2):
            if k==0:
                s=1
            else:
                s=-1
            errorTimes=0
            for n in range(size):
                if s*sign(x[i][n]-theta)!=y[n]:
                    errorTimes+=1
            if errorTimes<bestErrorTimes:
                bestErrorTimes=errorTimes
                bestS=s
                bestTheta=theta
                bestDimension=i
print("Optimal:i:",bestDimension+1,"s:",bestS,"theta:",bestTheta,"Ein",bestErrorTimes/size)


# In[6]:

#read testing data
with open('../hw2_test.dat') as f:
    for line in f:
        for i,value in enumerate(line.split()):
            if i<9:
                x2[i].append(float(value));
            else:
                y2.append(int(value));
    f.close();


# In[7]:

testErrorTimes=0
for j in range(size2):
    if bestS*sign(x2[bestDimension][j]-bestTheta)!=y2[j]:
        testErrorTimes+=1
print("Etest:",testErrorTimes/size2)

