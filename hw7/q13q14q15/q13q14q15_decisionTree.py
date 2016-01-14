
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.feature = None
        self.theta = None
        self.constant = None
        self.size = None


# In[3]:

def read_data(path,dimensions):
    x=[]
    for i in range(dimensions-1):
        x.append([])
    y=[]
    with open(path) as file:
        for line in file:
            for i,value in enumerate(line.split()):
                if i!=dimensions-1:
                    x[i].append(float(value))
                else:
                    y.append(int(value))
        file.close()
    return (x,y)


# In[4]:

def sign(x):
    if x>=0:
        return 1
    else:
        return -1


# In[5]:

def impurity(y):
    mu=y.count(1)/len(y)
    return 2*mu*(1-mu)


# In[6]:

def isTerminate(x,y):
    if all(i == y[0] for i in y):
        return True
    else:
        for i in range(len(x)):
            if not all(j == x[i][0] for j in x[i]):
                return False
        return True


# In[7]:

def decisionTree_train(x,y):
    if isTerminate(x,y):
        if all(i == y[0] for i in y):
            root = Tree()
            root.constant = y[0]
            root.size=len(y)
        else:
            root = Tree()
            if y.count(1)>y.count(-1):
                root.constant = 1
            else:
                root.constant = -1
            root.size=len(y)
        return root
    else:
        x_candidates=[]
        for i in range(len(x)):
            x_candidates.append([])
        for i in range(len(x)):
            temp=sorted(x[i])
            for j in range(1,len(temp)):
                x_candidates[i].append((temp[j-1]+temp[j])/2)
        best_impurity=len(x[0])/2
        for i in range(len(x_candidates)):
            for j in range(len(x_candidates[0])):
                y1=[]
                y2=[]
                for k in range(len(x[0])):
                    if x[i][k]<x_candidates[i][j]:
                        y1.append(y[k])
                    else:
                        y2.append(y[k])
                total_impurity=len(y1)*impurity(y1)+len(y2)*impurity(y2)
                if total_impurity<best_impurity:
                    best_impurity=total_impurity
                    best_feature=i
                    best_candidate=j
        x1=[]
        x2=[]
        y1=[]
        y2=[]
        for i in range(len(x)):
            x1.append([])
            x2.append([])
        for k in range(len(x[0])):
            if x[best_feature][k]<x_candidates[best_feature][best_candidate]:
                for i in range(len(x)):
                    x1[i].append(x[i][k])
                y1.append(y[k])
            else:
                for i in range(len(x)):
                    x2[i].append(x[i][k])
                y2.append(y[k])
        root = Tree()
        root.left = decisionTree_train(x1,y1)
        root.right = decisionTree_train(x2,y2)
        root.feature = best_feature
        root.theta = x_candidates[best_feature][best_candidate]
        root.size=len(x[0])
        return root


# In[8]:

def decisionTree_predict(root,x):
    if root.constant!=None:
        return root.constant
    if x[root.feature]<root.theta:
        return decisionTree_predict(root.left,x)
    else:
        return decisionTree_predict(root.right,x)


# In[9]:

x,y = read_data("../hw7_train.dat",3)
tree=decisionTree_train(x,y)


# In[10]:

print("level 0:",tree.feature,tree.theta)
print("level 1:",
      tree.left.feature,
      tree.left.theta,
      
      tree.right.feature,
      tree.right.theta)
print("level 2:",
      tree.left.left.feature,
      tree.left.left.theta,
      
      tree.left.right.feature,
      tree.left.right.theta,
      
      tree.right.left.feature,
      tree.right.left.theta,
      "(",tree.right.left.constant,")",
      
      tree.right.right.feature,
      tree.right.right.theta,
      "(",tree.right.right.constant,")")


# In[11]:

print("level 3:",
      tree.left.left.left.feature,
      tree.left.left.left.theta,
      "(",tree.left.left.left.constant,")",
      
      tree.left.left.right.feature,
      tree.left.left.right.theta,
      "(",tree.left.left.right.constant,")",
      
      tree.left.right.left.feature,
      tree.left.right.left.theta,
      
      tree.left.right.right.feature,
      tree.left.right.right.theta)

print("level 4:",
      tree.left.right.left.left.feature,
      tree.left.right.left.left.theta,
      
      tree.left.right.left.right.feature,
      tree.left.right.left.right.theta,
      
      tree.left.right.right.left.feature,
      tree.left.right.right.left.theta,
      
      tree.left.right.right.right.feature,
      tree.left.right.right.right.theta,
      "(",tree.left.right.right.right.constant,")")

print("level 5:",
      tree.left.right.left.left.left.feature,
      tree.left.right.left.left.left.theta,
      "(",tree.left.right.left.left.left.constant,")",
      
      tree.left.right.left.left.right.feature,
      tree.left.right.left.left.right.theta,
      "(",tree.left.right.left.left.right.constant,")",
      
      tree.left.right.left.right.left.feature,
      tree.left.right.left.right.left.theta,
      "(",tree.left.right.left.right.left.constant,")",
      
      tree.left.right.left.right.right.feature,
      tree.left.right.left.right.right.theta,
      "(",tree.left.right.left.right.right.constant,")",
      
      tree.left.right.right.left.left.feature,
      tree.left.right.right.left.left.theta,
      "(",tree.left.right.right.left.left.constant,")",
      
      tree.left.right.right.left.right.feature,
      tree.left.right.right.left.right.theta,
      "(",tree.left.right.right.left.right.constant,")",
      )


# In[12]:

x=np.array(x)
Ein=0
for k in range(len(x[0])):
    if y[k]!=decisionTree_predict(tree,x[:,k]):
        Ein+=1
Ein/=len(x[0])
print("Ein:",Ein)


# In[13]:

x2,y2 = read_data("../hw7_test.dat",3)
x2=np.array(x2)
Eout=0
for k in range(len(x2[0])):
    if y2[k]!=decisionTree_predict(tree,x2[:,k]):
        Eout+=1
Eout/=len(x2[0])
print("Eout:",Eout)


# In[ ]:



