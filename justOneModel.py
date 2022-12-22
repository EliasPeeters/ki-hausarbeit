
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

if __name__ == '__main__':
    # load features_raw
    features_raw = pd.read_csv('preProcessed/features_raw.csv')

    features_raw.drop('guenstig', axis=1, inplace=True)
    features_raw.drop('oepnv', axis=1, inplace=True)
    features_raw.drop('geschaeft', axis=1, inplace=True)
    features_raw.drop('distanz', axis=1, inplace=True)
    features_raw.drop('teuer', axis=1, inplace=True)
    features_raw.drop('arbeit', axis=1, inplace=True)
    features_raw.drop('ausbildung', axis=1, inplace=True)
    features_raw.drop('studium', axis=1, inplace=True)
    features_raw.drop('ausstattung', axis=1, inplace=True)

    # features_raw.drop('text_length', axis=1, inplace=True)
    # features_raw.drop('capital_words', axis=1, inplace=True)
    # features_raw.drop('word_cnt', axis=1, inplace=True)
    # features_raw.drop('errorPercentage', axis=1, inplace=True)

    # load price_raw
    price_raw = pd.read_csv('preProcessed/price_raw.csv')

    # split data into training and testing
    X_train, X_test, y_train, y_test = mainNewOhneText.splitData(price_raw, features_raw)

    model = LogisticRegression()
    start_fit = time()  # Get start time
    model.fit(X_train, y_train)
    end_fit = time()  # Get end time
    time_taken_fit = end_fit - start_fit  # Calculate training time
    print('training finished in ' + str(time_taken_fit) + ' seconds')

    mainNewOhneText.test_model(model, X_test, y_test, X_train, y_train, 'LogisticRegression', 'images')
    print('finished testing')

    col_names = []
    # summarize feature importance
    for col in X_train.columns:
        col_names.append(col)

    values = model.coef_[0]

    plt.figure(figsize=(9, 3))
    # plt.subplot(131)
    plt.barh(col_names, values)
    plt.suptitle('LogisticRegression')
    plt.savefig("images/logisticRegression_feature_importance.png")
    plt.show()
    plt.close()

    print(len(model.coef_[0]))