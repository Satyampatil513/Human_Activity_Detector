##Actual Code
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv(r"Train.csv")
# print(train)
#removed fall
train=train.drop('fall',axis=1)
fall=["FOL","FKL","SDL","BSC"]
active=["STD","WAL","JOG","JUM"]
asds=["STU","STN","SCH","CSI","CSO"]
#divided in 3 categories
act=[]

for i in range (len(train)):
    if train["label"][i] in fall:
        act.append(0)
    elif train["label"][i] in active:
        act.append(1)
    elif train["label"][i] in asds:
        act.append(2)
train["activity"]=act
xtrain=train.iloc[:,:-1]
xtrain=xtrain.drop('label',axis=1)
xtrain=xtrain.drop('Unnamed: 0',axis=1)
print(xtrain.columns)
ytrain=train.activity

#training model
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
xtr=scaler.fit_transform(xtrain)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#knn classifier
k=10
classifier=KNeighborsClassifier(n_neighbors=k)
classifier.fit(xtr,ytrain)
list = [2,2,2,2,2,2,2,2,2]
list = np.array(list)
list= list.reshape(1,-1)
pred=classifier.predict(list)
print(pred)