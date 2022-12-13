import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from time import time
from sklearn.svm import SVC
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)



def read_data():
    # Use a breakpoint in the code line below to debug your script.
    file_name = 'data.csv'
    print("Reading data from file: " + file_name)
    data = pd.read_csv(file_name)
    return data

def storePriceSeperate(data):
    data.loc[(data.price > 19), 'price'] = 24
    data.loc[(data.price <= 19), 'price'] = 0

    data.loc[(data.price == 24), 'price'] = 1
    # data.loc[(data.price == '0'), 'price'] = '0'

    price_raw = data['price']
    features_raw = data.drop('price', axis=1)
    print(price_raw)
    return price_raw, features_raw

def removeUnusedColumns(data):
    data = data.drop('id', axis=1)
    data = data.drop('nebenkosten', axis=1)
    data = data.drop('kaltmiete', axis=1)
    # data = data.drop('location_postalCode', axis=1)
    data = data.drop('location_cityName', axis=1)
    data = data.drop('location_street', axis=1)
    # data = data.drop('description', axis=1)

    output = pd.Series([y for x in data['description'].values.flatten() for y in x.split()]).value_counts()
    data = pd.get_dummies(data)
    # data = data.drop('description', axis=1)
    return data

def custom_hot_encoding(data):
    for i in range(len(data)):
        words_as_array = data.loc[i, 'description'].split()
        for word in words_as_array:
            if 'word_' + word not in data:
                # new_df = pd.Series(0)
                data['word_' + word] = 0
                data.loc[i, 'word_' + word] = 1
            else:
                data.loc[i, 'word_' + word] += 1
        print(i)
    return data



def splitData(price_raw, features_raw):
    from sklearn.model_selection import train_test_split

    # Split the 'features' and 'income' data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_raw,
                                                        price_raw,
                                                        test_size=0.2,
                                                        random_state=42)

    # Show the results of the split
    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = DecisionTreeClassifier(random_state=10)
    start = time()  # Get start time
    learner = model.fit(X_train, y_train.astype('int').values.ravel())
    end = time()

    predictions_train = learner.predict(X_train[:300].astype('int'))

    print(end-start)
    return model

def test_model(model, X_test, y_test):
    results = {}
    predictions_test = model.predict(X_test)

    y_train_helper = y_train.to_numpy()
    y_test_helper = y_test.to_numpy()

    results['acc_test'] = accuracy_score(y_test_helper, predictions_test)
    print(results)

if __name__ == '__main__':
    data = read_data()
    price_raw, features_raw = storePriceSeperate(data)
    features_raw = custom_hot_encoding(features_raw)
    # print(features_raw)
    features_raw = removeUnusedColumns(features_raw)
    features_raw.to_csv(index=False)
    # X_train, X_test, y_train, y_test = splitData(price_raw, features_raw)
    # model = train_model(X_train, y_train)
    # test_model(model, X_test, y_test)


