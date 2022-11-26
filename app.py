from flask import Flask,render_template,request
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)

#Connectivity
app = Flask(__name__)
# pikachu=pd.read_csv('Test.csv')

# testing=pikachu
# testing=testing.drop('fall',axis=1)
# testing=testing.drop('label',axis=1)

# testing=testing.iloc[:,1:]

@app.route('/')
def index():
       return render_template('index.html')


# @app.route('/submit',methods = ['POST', 'GET'])


with open('Model1.pkl', 'rb') as file:  
    Model1 = pickle.load(file)
with open('random_forest.pkl', 'rb') as file:  
    rf = pickle.load(file)
with open('Model_act.pkl', 'rb') as file:  
    M2 = pickle.load(file)
with open('Model_asds.pkl', 'rb') as file:  
    M3 = pickle.load(file)
test=pd.read_csv('Train.csv')
test=test.drop('label',axis=1)
test=test.drop('fall',axis=1)
test=test.drop('gyro_max',axis=1)
test=test.iloc[:,1:]
x=rf.predict(test)
print(x)



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
    # post_lin_max=float(request.form['post_lin_max'])
    list=[acc_max,gyro_max,acc_kurtosis,gyro_kurtosis,lin_max,acc_skewness,gyro_skewness,post_gyro_max]
    list= np.array(list)
    list= list.reshape(1,-1)
    
    

    
     
    
    
    
    
    
    
    
    
    
    
    # abc = pca.fit_transform(testing1)
    pred=Model1.predict(list)
    print(list)
    print(pred[-1])
    if pred[-1] == 0 :
        pred="fall"
        q=rf.predict(list)
        q=q[-1]
        if (q==1) : q="FOL"
        elif (q==2) : q="FKL"
        elif (q==3) : q="SDL"
        else : q="BSC"
        
    elif pred[-1] == 1:
        pred="active"
        q=M2.predict(list)
        q=q[-1]
        if (q==5) : q="STANDING"
        elif (q==6) : q="WALKING"
        elif (q==7) : q="JOGGING"
        else : q="JUMPING"
    else :
        q=M3.predict(list)
        q=q[-1]
        if (q==1) : q="STU"
        elif (q==2) : q="STN"
        elif (q==3) : q="SCH"
        elif (q==3) : q="CHI"
        else : q="CSO"
        pred="asc-dsc"
    
    return render_template('pass.html', p=pred,q=q) 
    

if __name__ == '__main__':
       app.run()
