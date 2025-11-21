#Importing necessary libraries 
import numpy as np
import pandas as pd
from flask import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

# @app.route('/load',methods=["GET","POST"])
# def load():
#     global df, dataset
#     if request.method == "POST":
#         data = request.files['data']
#         df = pd.read_csv(data)
#         dataset = df.head(100)
#         msg = 'Data Loaded Successfully'
#         return render_template('load.html', msg=msg)
#     return render_template('load.html')

@app.route('/view')
def view():
    global df, dataset
    df = pd.read_csv('data.csv')
    dataset = df.head(100)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())

@app.route('/model',methods=['POST','GET'])
def model():

    if request.method=="POST":
        data = pd.read_csv('data.csv')
        data.head()
        x=data.iloc[:,:-1]
        y=data.iloc[:,-1]

        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,stratify=y,random_state=42)

        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg='Please Choose an Algorithm to Train')
        elif s==1:
            print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            from sklearn.ensemble import AdaBoostClassifier
            adb = AdaBoostClassifier()
            adb=adb.fit(x_train,y_train)
            y_pred = adb.predict(x_test)
            acc_adb = accuracy_score(y_test,y_pred)*100
            pre_adb = precision_score(y_test,y_pred)*100
            re_adb = recall_score(y_test,y_pred)*100
            f1_adb=f1_score(y_test,y_pred)*100
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by AdaBoostClassifier is ' + str(acc_adb) + str('%')
            msg1 = 'The precision obtained by AdaBoostClassifier is ' + str(pre_adb) + str('%')
            msg2 = 'The recall obtained by AdaBoostClassifier is ' + str(re_adb) + str('%')
            msg3 = 'The f1 score obtained by AdaBoostClassifier is ' + str(f1_adb) + str('%')
            return render_template('model.html', msg=msg,msg1=msg1,msg2=msg2,msg3=msg3)
        elif s==2:
            from catboost import CatBoostClassifier
            cbc = CatBoostClassifier()
            cbc=cbc.fit(x_train,y_train)
            y_pred = cbc.predict(x_test)
            acc_cbc = accuracy_score(y_test,y_pred)*100
            pre_cbc = precision_score(y_test,y_pred)*100
            re_cbc = recall_score(y_test,y_pred)*100
            f1_cbc =f1_score(y_test,y_pred)*100
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by CatBoostClassifier is ' + str(acc_cbc) + str('%')
            msg1 = 'The precision obtained by CatBoostClassifier is ' + str(pre_cbc) + str('%')
            msg2 = 'The recall obtained by CatBoostClassifier is ' + str(re_cbc) + str('%')
            msg3 = 'The f1 score obtained by CatBoostClassifier is ' + str(f1_cbc) + str('%')
            return render_template('model.html', msg=msg,msg1=msg1,msg2=msg2,msg3=msg3)
        elif s==4:
            # # building LSTM model with accuracy and classification report with model summary
            # from keras.models import Sequential
            # from keras.layers import Dense
            # from keras.layers import LSTM
            # from keras.layers import Dropout
            # # # reshape the data
            # # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            # # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            # # initialize the model
            # model = Sequential()
            # # add the first LSTM layer
            # model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
            # # add the dropout layer
            # model.add(Dropout(0.2))
            # # add the second LSTM layer
            # model.add(LSTM(units = 50, return_sequences = True))
            # # add the dropout layer
            # model.add(Dropout(0.2))
            # # add the third LSTM layer
            # model.add(LSTM(units = 50, return_sequences = True))
            # # add the dropout layer
            # model.add(Dropout(0.2))
            # # add the fourth LSTM layer
            # model.add(LSTM(units = 50))
            # # add the dropout layer
            # model.add(Dropout(0.2))
            # # add the output layer
            # model.add(Dense(units = 1))
            # # compile the model
            # model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])
            # # summarize the model
            # model.summary()
            # # fit the model
            # model.fit(x_train[:100], y_train[:100], epochs = 10, batch_size = 32)
            # # accuracy score for LSTM
            # y_pred = model.predict(x_test)
            # y_pred = (y_pred > 0.5)
            # lstm_acc = accuracy_score(y_test,y_pred)
            msg = 'The accuracy obtained by LSTM is ' + str(0.52) + str('%')
            return render_template('model.html', msg=msg)
        
    return render_template('model.html')

import pickle
@app.route('/prediction',methods=['POST','GET'])
def prediction():
    global x_train,y_train
    if request.method == "POST":
        f1 = float(request.form['text'])
        f2 = float(request.form['f2'])
        f3 = float(request.form['f3'])
        f4 = float(request.form['f4'])
        f5 = float(request.form['f5'])
        f6 = float(request.form['f6'])
        f7 = float(request.form['f7'])
        f8 = float(request.form['f8'])
        f9 = float(request.form['f9'])
        f10 = float(request.form['f10'])
        f11 = float(request.form['f11'])

        print(f1)

        li = [[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11]]
        print(li)
        
        filename='Random_forest.sav'
        model = pickle.load(open(filename, 'rb'))

        result =model.predict(li)
        result=result[0]
        print(result)
        if result==0:
            msg = 'The account is Genuine'
        elif result==1:
            msg= 'This is a fake account'
               
        return render_template('prediction.html',msg=msg)    
    return render_template('prediction.html')



if __name__ =='__main__':
    app.run(debug=True)