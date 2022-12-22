import math
import mainNewOhneText
import os
import datetime

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

from time import time
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from colorama import Fore, Back, Style
from http.server import BaseHTTPRequestHandler, HTTPServer
from sklearn.metrics import PrecisionRecallDisplay

import matplotlib.pyplot as plt
# check if running in main

def do_one_model(model, X_train, y_train, X_test, y_test, name, folder):
    print(name + ': start training')
    start_fit = time()  # Get start time
    model.fit(X_train, y_train)
    end_fit = time()  # Get end time
    time_taken_fit = end_fit - start_fit  # Calculate training time
    print(name + ': training finished in ' + str(time_taken_fit) + ' seconds')
    print(name + ': start testing')
    start_test = time()  # Get start time
    results = mainNewOhneText.test_model(model, X_test, y_test, X_train, y_train, name, folder)
    end_test = time()  # Get end time
    time_taken_test = end_test - start_test  # Calculate training time
    results['time_train'] = time_taken_fit
    results['time_test'] = time_taken_test
    print(name + ': finished testing')

    if hasattr(model, 'feature_importances_'):
        plt = mainNewOhneText.visualize_feature_importance(model, X_train, name, folder)
        # plt.savefig(folder + '/' + name + '_feature_importance.png')
        # plt.close()
    return results

if __name__ == '__main__':
    # load features_raw
    features_raw = pd.read_csv('preProcessed/features_raw.csv')

    # load price_raw
    price_raw = pd.read_csv('preProcessed/price_raw.csv')

    # split data into training and testing
    X_train, X_test, y_train, y_test = mainNewOhneText.splitData(price_raw, features_raw)

    # create models
    models = []
    models.append(('RandomForestClassifier', RandomForestClassifier()))
    models.append(('GaussianNB', GaussianNB()))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
    models.append(('KNeighborsClassifier', KNeighborsClassifier()))
    models.append(('LogisticRegression', LogisticRegression()))
    models.append(('MLPClassifier', MLPClassifier()))
    models.append(('AdaBoostClassifier', AdaBoostClassifier()))
    models.append(('SVC', SVC()))

    # create output folder if not exists
    if not os.path.exists('output'):
        os.makedirs('output')

    # create folder in output folder with current time and date
    # get current hour, minute and second
    now = datetime.datetime.now()

    folder_name = 'output/' + str(now.year) + '-' + str(now.month) + '-' + str(now.day) + ' ' + str(now.hour) + ':' + str(now.minute) + ':' + str(now.second)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # create df with for results
    overall_results = pd.DataFrame(columns=['name', 'acc_test', 'acc_train', 'fbeta_test', 'fbeta_train'])

    # do one model
    for name, model in models:
        results = do_one_model(model, X_train, y_train, X_test, y_test, name, folder_name)
        # append results to overall_results
        overall_results = overall_results.append(results, ignore_index=True)

    # save overall_results to csv
    overall_results.to_csv(folder_name + '/overall_results.csv', index=False)

    # final print
    print('finished')


    # do_one_model(AdaBoostClassifier(), X_train, y_train, X_test, y_test, 'AdaBoostClassifier')
