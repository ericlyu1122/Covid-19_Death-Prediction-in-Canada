import os
import argparse
import time
import pickle
import time
# 3rd party libraries
import numpy as np
import pandas as pd
import datetime as datetime
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from sklearn.tree import DecisionTreeClassifier
import utils

import linear_model
import linear_model2
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question


    if question == "2":
        data = pd.read_csv(os.path.join('..','data','phase1_training_data.csv'))
        #put death on the first column for auto regression.
        data = data.loc[data['country_id']=='CA',
                        ['country_id','deaths','date','cases','cases_14_100k','cases_100k']]
        data['date'] = pd.to_datetime(data['date'],format='%m/%d/%Y')
        data['date'] -= datetime.datetime(2020,7,31)#start Date
        data['date'] /= np.timedelta64(1, 'D')
        data['date'].astype('int')

        X = data.loc[data['date']>=0,['deaths',
                                      'cases',
                                      #'cases_14_100k',
                                      #'cases_100k'
                                      ]]
        #X = X.apply(lambda d:datetime.datetime.strptime(d,format='%m/%d/%Y'))
        #y = data.loc[data['country_id']=='CA',['deaths']].values
        #y = y[X['date']>=0]
        # Fit weighted least-squares estimator
        model = linear_model.LeastSquaresBias(10,[1,1,1,1])
        model.fit(X.values)
#        print(model.predict(pd.DataFrame(data=d)))
        print(model.predict(X[data['country_id']=='CA'].values,10))

    if question == "3":
        data = pd.read_csv(os.path.join('..','data','phase1_training_data.csv'))
        X = pd.to_datetime(data.loc[data['country_id']=='CA',['date']].stack(),format='%m/%d/%Y').unstack()
        X -= datetime.datetime(2019,12,31)
        X /= np.timedelta64(1, 'D')
        X.astype('int')
        #X = X.apply(lambda d:datetime.datetime.strptime(d,format='%m/%d/%Y'))
        y = data.loc[data['country_id']=='CA',['deaths']].values
        y = y[X['date']>=0]
        X = X[X['date']>=0]
        # Fit weighted least-squares estimator
        model = linear_model2.LeastSquaresBias(80)
        model.fit(y)
        d = {'col1':[255,256,257,258,259]}
#        print(model.predict(pd.DataFrame(data=d)))
        y_pred = model.predict(10)
        #print(y_pred)
        y_final = np.ones((10,1))
        for i in range(1,10):
            y_final[10-i,0] = np.round(y_pred[10-i]-y_pred[9-i])
        print(y_final)
