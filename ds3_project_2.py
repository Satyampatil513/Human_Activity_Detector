# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:05:12 2022

@author: jites
"""

#!/usr/bin/env python
# coding: utf-8

# ### the dataset contains 13 activities among which 4 activities are related to human falls 
#and 9 are normal human activities.
# 
# ### 3 classes: Fall, Active and Ascension-Descension
# 
# #### The fall activities are:
# #### FOL: Forward-lying. 
# #### FKL: Forward knee-lying. 
# #### SDL: Sideward-lying. 
# #### BSC: Back sitting chair. 
# 
# #### Active:
# #### STD: Standing. 
# #### WAL: Walking. 
# #### JOG: Jogging. 
# #### JUM: Jumping.
# 
# #### Ascension-Descension:
# #### STU: Stairs up. 
# #### STN: Stairs down. 
# #### SCH: Sit chair. 
# #### CSI: Car-step in. 
# #### CSO: Car-step out

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

# # train data

# In[2]:


train = pd.read_csv("Train.csv")
# print(train)


# In[3]:


# train.head()


# ## removing column 'fall' and renaming categories :
# ##  0= fall ;1= active ; 2=ascending descending

# In[4]:


train=train.drop('fall',axis=1)

fall=["FOL","FKL","SDL","BSC"]
active=["STD","WAL","JOG","JUM"]
asds=["STU","STN","SCH","CSI","CSO"]

act=[]

for i in range (len(train)):
    if train["label"][i] in fall:
        act.append(0)
    elif train["label"][i] in active:
        act.append(1)
    elif train["label"][i] in asds:
        act.append(2)
train["activity"]=act

# train.head()


# In[5]:

label_train=train["label"]
train=train.drop('label',axis=1)


xtrain=train.iloc[:,:-1]
ytrain=train.activity


# print(xtrain)
#print(ytrain)
# print(xtrain.shape)


# In[6]:


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
xtr=scaler.fit_transform(xtrain)


# # test data
# 

# In[7]:


test = pd.read_csv("Test.csv")


# In[8]:


# test.head()


# In[9]:


test=test.drop('fall',axis=1)

fall=["FOL","FKL","SDL","BSC"]
active=["STD","WAL","JOG","JUM"]
asds=["STU","STN","SCH","CSI","CSO"]

act=[]

for i in range (len(test)):
    if test["label"][i] in fall:
        act.append(0)
    elif test["label"][i] in active:
        act.append(1)
    elif test["label"][i] in asds:
        act.append(2)
test["activity"]=act

# test.head()


# In[10]:

label_test=test["label"]
test=test.drop('label',axis=1)
xtest=test.iloc[:,:-1]
ytest=test.activity
#print(xtest)
#print(ytest)
# print(xtest.shape)


# In[11]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
xts=scaler.fit_transform(xtest)


# In[12]:




# In[13]:


#df = pd.DataFrame(xtrain, columns=['PC1','PC2'])
# df['activity'] = train.get('activity')
# df.head()


# In[14]:

xtrain=xtrain.iloc[:,1:]
xtrain=xtrain.drop('post_lin_max',axis=1)
xtest=xtest.iloc[:,1:]
xtest=xtest.drop('post_lin_max',axis=1)
xtrain_1=xtrain
# from sklearn.decomposition import PCA
  
# pca = PCA(n_components = 5)
  
# xtrain = pca.fit_transform(xtrain)
# xtest = pca.transform(xtest)
  
# comp_variance = pca.explained_variance_ratio_

# df = pd.DataFrame(data = xtrain, 
#                   columns = ['PC1', 'PC2','PC3','pc4','pc5'])
# activity = pd.Series(train['activity'], name='activity')

# result_df = pd.concat([df, activity], axis=1)
# # result_df.head(5)


# In[15]:

'''
fig = plt.figure(figsize = (12,10))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('First Principal Component ', fontsize = 15)
ax.set_ylabel('Second Principal Component ', fontsize = 15)
#ax.set_zlabel('third Principal Component ', fontsize = 15)
ax.set_title('Principal Component Analysis', fontsize = 20)
'''



from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
"""
fig = plt.figure()
 
# syntax for 3-D projection
#ax = plt.axes(projection ='3d')
 
# defining axes
z = df['PC2']
x = df['PC1']
y = df['PC3']
ax.scatter(x, y, z)
 
# syntax for plotting
ax.set_title('3d Scatter plot geeks for geeks')
plt.show()






targets = [0, 1, 2]
colors = ['r', 'g', 'b']
for activity, color in zip(targets, colors):
    indicesToKeep = result_df['activity']==activity
    ax.scatter(result_df.loc[indicesToKeep, 'PC1'], 
               result_df.loc[indicesToKeep, 'PC2'], 
               #result_df.loc[indicesToKeep, 'PC3'],
               c = color, 
               s = 50)
ax.legend(targets)
ax.grid()

print('Variance of each component:', pca.explained_variance_ratio_)
print('\n Total Variance Explained:', round(sum(list(pca.explained_variance_ratio_))*100, 2))

"""
# ## knn classifier

# In[16]:



from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

k=7
classifier=KNeighborsClassifier(n_neighbors=k)
classifier.fit(xtrain_1,ytrain)
pred=classifier.predict(xtrain_1)
print(pred)
score1=accuracy_score(ytrain,pred)
print("accuracy : ",score1)
c=confusion_matrix(ytrain,pred)
print(c)

Pkl_Filename = "Model1.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(classifier, file)

# In[17]:


# res=classifier.predict([xtest[87]])
#print(type(res))
# if res[0]==0:
#     print("person was falling")
# elif res[0]==1:
#     print('person was active')
# elif res[0]==2:
#     print('person was ascending descending')

print("--------------------------------------------------------------------")
"""
# In[ ]:
l=[1,3,5]  #creating a list of the required knn values
n=[]
for i in l:
    knn=KNeighborsClassifier(n_neighbors=i) #classifing our data into each of the given knn values using the KNeighborsClassifier function 
    knn.fit(xtr,ytrain)  #fitting out testa nd tarin data
    knn_predict=knn.predict(xts)  #creating a pedicted data of the test data
    print(confusion_matrix(ytest,knn_predict))  #craeting confusion matrix using the original data and predicted data
    print(accuracy_score(ytest,knn_predict))  #finding accuracy of our predicted data by comparing it with the original data 
    n.append(accuracy_score(ytest,knn_predict))  #appending accuracy score of each knn value in a list
print(f"Highest accuracy is {max(n)}")  #printing the maximum accuracy of the three knn values
"""


# In[ ]:
#creating a function to perform bayes classification
fall1=[1,2,3,4]
active1=[5,6,7,8]
asds1=[9,10,11,12,13]



xtrain=pd.DataFrame(xtrain)
xtrain=xtrain.join(label_train)


xtest=pd.DataFrame(xtest)
xtest=xtest.join(label_test)



act1=[]

for i in range (len(xtest)):
    if xtest["label"][i]=="FOL":
        act1.append(1)
    elif xtest["label"][i]=="FKL":
        act1.append(2)
    elif xtest["label"][i]=="SDL":
        act1.append(3)
    elif xtest["label"][i]=="BSC":
        act1.append(4)
    elif xtest["label"][i]=="STD":
        act1.append(5)
    elif xtest["label"][i]=="WAL":
        act1.append(6)
    elif xtest["label"][i]=="JOG":
        act1.append(7)
    elif xtest["label"][i]=="JUM":
        act1.append(8)
    elif xtest["label"][i]=="STU":
        act1.append(9)
    elif xtest["label"][i]=="STN":
        act1.append(10)
    elif xtest["label"][i]=="SCH":
        act1.append(11)
    elif xtest["label"][i]=="CSI":
        act1.append(12)
    elif xtest["label"][i]=="CSO":
        act1.append(13)
xtest["label"]=act1


act2=[]

for i in range (len(xtrain)):
    if xtrain["label"][i]=="FOL":
        act2.append(1)
    elif xtrain["label"][i]=="FKL":
        act2.append(2)
    elif xtrain["label"][i]=="SDL":
        act2.append(3)
    elif xtrain["label"][i]=="BSC":
        act2.append(4)
    elif xtrain["label"][i]=="STD":
        act2.append(5)
    elif xtrain["label"][i]=="WAL":
        act2.append(6)
    elif xtrain["label"][i]=="JOG":
        act2.append(7)
    elif xtrain["label"][i]=="JUM":
        act2.append(8)
    elif xtrain["label"][i]=="STU":
        act2.append(9)
    elif xtrain["label"][i]=="STN":
        act2.append(10)
    elif xtrain["label"][i]=="SCH":
        act2.append(11)
    elif xtrain["label"][i]=="CSI":
        act2.append(12)
    elif xtrain["label"][i]=="CSO":
        act2.append(13)
xtrain["label"]=act2





# xtrain=xtrain.iloc[:,1:]
# xtest=xtest.iloc[:,1:]

xtrain_fall=xtrain[xtrain["label"].isin(fall1)]
ytrain_fall=xtrain_fall["label"]
xtest_fall=xtest[xtest["label"].isin(fall1)]
ytest_fall=xtest_fall["label"]


# In[ ]:


xtrain_active=xtrain[xtrain["label"].isin(active1)]
ytrain_active=xtrain_active["label"]
xtest_active=xtest[xtest["label"].isin(active1)]
ytest_active=xtest_active["label"]





xtrain_asds=xtrain[xtrain["label"].isin(asds1)]
ytrain_asds=xtrain_asds["label"]
xtest_asds=xtest[xtest["label"].isin(asds1)]
ytest_asds=xtest_asds["label"]


xtrain_fall=xtrain_fall.drop("label",axis=1)
xtrain_active=xtrain_active.drop("label",axis=1)
xtrain_asds=xtrain_asds.drop("label",axis=1)


xtest_fall=xtest_fall.drop("label",axis=1)
xtest_active=xtest_active.drop("label",axis=1)
xtest_asds=xtest_asds.drop("label",axis=1)



print("_____________________________________________________")
l=[1,3,5]  #creating a list of the required knn values
n=[]
for i in l:
    knn_fall=KNeighborsClassifier(n_neighbors=i) #classifing our data into each of the given knn values using the KNeighborsClassifier function 
    knn_fall.fit(xtrain_fall,ytrain_fall)  #fitting out testa nd tarin data
    knn_predict=knn_fall.predict(xtest_fall)  #creating a pedicted data of the test data
    #print(confusion_matrix(ytest_fall,knn_predict))  #craeting confusion matrix using the original data and predicted data
    print(accuracy_score(ytest_fall,knn_predict))  #finding accuracy of our predicted data by comparing it with the original data 
    n.append(accuracy_score(ytest_fall,knn_predict))  #appending accuracy score of each knn value in a list
print(f"Highest accuracy is {max(n)}")  #printing the maximum accuracy of the three knn values


# Number of trees in random forest
n_estimators = [200,400,600,800,1000]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [None,10,30,50,70]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 9, 12]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 3, 5, 7]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_random1 = RandomizedSearchCV(estimator = rf,
                                param_distributions = random_grid,
                                n_iter = 100, cv = 5,
                                verbose=2, 
                                random_state=42, 
                                n_jobs = -1
                              )
rf_random1.fit(xtrain_fall, ytrain_fall)
x=rf_random1.predict(xtest_fall)
print(accuracy_score(ytest_fall,x))
Pkl_Filename = "Model_fall.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf_random1, file)
print("_____________________________________________________")
l=[4,3,5]  #creating a list of the required knn values
n=[]
for i in l:
    knn_act=KNeighborsClassifier(n_neighbors=i) #classifing our data into each of the given knn values using the KNeighborsClassifier function 
    knn_act.fit(xtrain_active,ytrain_active)  #fitting out testa nd tarin data
    knn_predict=knn_act.predict(xtest_active)  #creating a pedicted data of the test data
    #print(confusion_matrix(ytest_fall,knn_predict))  #craeting confusion matrix using the original data and predicted data
    print(accuracy_score(ytest_active,knn_predict))  #finding accuracy of our predicted data by comparing it with the original data 
    n.append(accuracy_score(ytest_active,knn_predict))  #appending accuracy score of each knn value in a list
print(f"Highest accuracy is {max(n)}")  #printing the maximum accuracy of the three knn values


#Number of trees in random forest
n_estimators = [200,400,600,800,1000]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [None,10,30,50,70]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 9, 12]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 3, 5, 7]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

rf = RandomForestClassifier()
rf_random2 = RandomizedSearchCV(estimator = rf,
                                param_distributions = random_grid,
                                n_iter = 100, cv = 5,
                                verbose=2, 
                                random_state=42, 
                                n_jobs = -1
                              )
rf_random2.fit(xtrain_active,ytrain_active)
x=rf_random2.predict(xtest_active)
print(accuracy_score(ytest_active,x))

Pkl_Filename = "Model_act.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf_random2, file)


print("_____________________________________________________")
l=[1,3,5]  #creating a list of the required knn values
n=[]
for i in l:
    knn_asds=KNeighborsClassifier(n_neighbors=i) #classifing our data into each of the given knn values using the KNeighborsClassifier function 
    knn_asds.fit(xtrain_asds,ytrain_asds)  #fitting out testa nd tarin data
    knn_predict=knn_asds.predict(xtest_asds)  #creating a pedicted data of the test data
    #print(confusion_matrix(ytest_fall,knn_predict))  #craeting confusion matrix using the original data and predicted data
    print(accuracy_score(ytest_asds,knn_predict))  #finding accuracy of our predicted data by comparing it with the original data 
    n.append(accuracy_score(ytest_asds,knn_predict))  #appending accuracy score of each knn value in a list
print(f"Highest accuracy is {max(n)}")  #printing the maximum accuracy of the three knn values


#Number of trees in random forest
n_estimators = [200,400,600,800,1000]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [None,10,30,50,70]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 9, 12]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 3, 5, 7]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_random3 = RandomizedSearchCV(estimator = rf,
                                param_distributions = random_grid,
                                n_iter = 100, cv = 5,
                                verbose=2, 
                                random_state=42, 
                                n_jobs = -1
                              )
rf_random3.fit(xtrain_asds, ytrain_asds)
x=rf_random3.predict(xtest_asds)
print(accuracy_score(ytest_asds,x))

Pkl_Filename = "Model_asds.pkl"  



with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(rf_random3, file)





