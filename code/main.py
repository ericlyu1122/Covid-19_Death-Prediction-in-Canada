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
import sklearn.metrics
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question


    if question == "2":
        # Data set
        data = pd.read_csv(os.path.join('..','data','phase2_training_data.csv'))
        # put death on the first column for auto regression.

        data['date'] = pd.to_datetime(data['date'],format='%m/%d/%Y')

        ''' Choose start Date '''
        startdate = datetime.datetime(2020,7,15)
        print('startdate : ', startdate)
        data['date'] -= startdate#start Date


        data['date'] /= np.timedelta64(1, 'D')
        data.replace(np.nan, -1, inplace=True)
        data.replace(np.inf, -1, inplace=True)
        data['date'].astype('int')
        data=data[data['date']>=0]


        ''' Features to choose from '''
        X = data.loc[:,['country_id', 'deaths',
                                      #12'cases',
                                      'cases_14_100k',
                                      'cases_100k'
                                      ]]

        ''' Choose from Countries for training '''
        #X = X[(X['country_id']=='CA')|(X['country_id']=='SE')]

        ''' Choose K '''
        K = 35
        mintest = 1000
        ans= np.array([9504,9530,9541,9557,9585,9585,9585,9627,9654,9664,9699])
        #for k in range(K):
            # Fit weighted least-squares estimator
        model = linear_model.MultiFeaturesAutoRegressor(K)
        model.fit(X)
        #    currtest = np.sqrt(sklearn.metrics.mean_squared_error(model.predict(X[X['country_id']=='CA'],11), ans))
        #    print(k)
        #    if currtest<=mintest:
        #        mintest = currtest
        #        print(mintest)
        r = model.predict(X[X['country_id']=='CA'],5)
        #print(np.sqrt(sklearn.metrics.mean_squared_error(r, ans)))
        print(r)
