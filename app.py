from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#Connectivity
app = Flask(__name__)

@app.route('/')
def index():
       return render_template('index.html')


# @app.route('/submit',methods = ['POST', 'GET'])



##Actual Code


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

@app.route('/', methods=['POST'])
def getvalue():
    acc_max=float(request.form['acc_max']) 
    gyro_max=float(request.form['gyro_max'])
    acc_kurtosis=float(request.form['acc_kurtosis'])
    gyro_kurtosis=float(request.form['gyro_kurtosis'])
    lin_max=float(request.form['gyro_kurtosis'])
    acc_skewness=float(request.form['acc_skewness'])
    gyro_skewness=float(request.form['gyro_skewness'])
    post_gyro_max=float(request.form['post_gyro_max'])
    post_lin_max=float(request.form['post_lin_max'])
    list=[acc_max,gyro_max,acc_kurtosis,gyro_kurtosis,lin_max,acc_skewness,gyro_skewness,post_gyro_max,post_lin_max]
    list= np.array(list)
    list= list.reshape(1,-1)
    pred=classifier.predict(list)
    print(pred)
    return render_template('pass.html', p=pred) 
    

if __name__ == '__main__':
       app.run()
