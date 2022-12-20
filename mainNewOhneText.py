import math

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
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score
from colorama import Fore, Back, Style
from http.server import BaseHTTPRequestHandler, HTTPServer
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def read_data():
    # Use a breakpoint in the code line below to debug your script.
    file_name = 'Result_8.csv'
    # file_name = 'dataNormalized.csv'
    # file_name = 'allNormalizedDifference.csv'
    print(Fore.GREEN + 'Reading data from file: ' + file_name)

    data = pd.read_csv(file_name)

    return data


def storePriceSeperate(data):
    data.loc[(data.price > 19.3478), 'price'] = 24
    data.loc[(data.price <= 19.3478), 'price'] = 0

    data.loc[(data.price == 24), 'price'] = 1

    # data.loc[(data.price > 0), 'price'] = 1
    # data.loc[(data.price < 0), 'price'] = 0

    price_raw = data['price']
    features_raw = data.drop('price', axis=1)
    return price_raw, features_raw


def removeUnusedColumns(data):
    data = data.drop('id', axis=1)
    data = data.drop('nebenkosten', axis=1)
    data = data.drop('kaltmiete', axis=1)
    data = data.drop('location_cityName', axis=1)
    data = data.drop('location_street', axis=1)

    data = data.drop('description', axis=1)

    data = data.drop('location_postalCode', axis=1)
    data = data.drop('roomSize', axis=1)
    # data = data.drop('errorPercentage', axis=1)

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


def preProcessData(data):
    # add new column if word is in description
    data['description'] = data['description'].str.lower()

    word_list = ["balkon", "trockner", "garage", "jura", "garten", "wein", "vegan", "jura", "bier"]
    for word in word_list:
        data['word_' + word] = data['description'].str.contains(word).astype(int)

    # add description length
    data['text_length'] = round(data['description'].str.len() / 100)

    # round errorPercentage to two digits
    data['errorPercentage'] = round(data['errorPercentage'], 2)

    # round plz to three digits
    data['location_postalCode'] = round(data['location_postalCode'] / 100, 0)

    # count capital words in description
    data['capital_words'] = data['description'].str.findall(r'[A-Z]{2,}').str.len()

    teuer_words = ["teuer", "kostenträchti", "viel geld kosten", "hochpreisig", "im oberen preissegment",
                   "kostenaufwändig", "kostenaufwendig", "kostenintensiv", "preisintensiv", "deier", "gepfeffert",
                   "gesalzen", "happig", "ins geld gehen", "ins geld reißen", "Loch in die kasse reißen",
                   "loch ins portmonee reißen", "richtig geld kosten", "saftig", "sich gewaschen haben", "stolz",
                   "teurer spaß", "teures vergnügen"]
    data['teuer'] = data['description'].str.contains('|'.join(teuer_words)).astype(int)

    munich_words = ["Allach", "Altstadt", "Am Hart", "Am Moosfeld", "Am Riesenfeld", "Au", "Aubing", "Berg am Laim", "Bogenhausen", "Daglfing", "Denning", "Englschalking", "Fasangarten", "Feldmoching", "Forstenried", "Freiham", "Freimann", "Fürstenried", "Giesing (Obergiesing)", "Giesing (Untergiesing)", "Hadern", "Haidhausen", "Harlaching", "Hasenbergl", "Holzapfelkreuth (Ostteil)", "Holzapfelkreuth (Westteil)", "Isarvorstadt", "Johanneskirchen", "Laim", "Langwied", "Lehel", "Lochhausen", "Ludwigsvorstadt", "Maxvorstadt", "Milbertshofen", "Moosach", "Neuhausen", "Nymphenburg", "Oberföhring", "Obermenzing", "Pasing", "Perlach", "Ramersdorf", "Riem", "Schwabing (Ostteil)", "Schwabing (Westteil)", "Schwanthalerhöhe", "Sendling (Obersendling)", "Sendling (Unter- und Mittersendling)", "Sendling (Westteil)", "Solln", "Steinhausen", "Thalkirchen", "Trudering", "Untermenzing", "Zamdorf"]
    data['munich'] = data['description'].str.contains('|'.join(munich_words)).astype(int)

    # get age of user from description when he wrote number + "Jahre"
    # data['age'] = data['description'].str.findall(r'(\d+)Jahre').str[0].astype(int)




    # data['word_balkon'] = data['description'].str.contains('balkon').astype(int)
    # data['word_trockner'] = data['description'].str.contains('trockner').astype(int)
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
    # model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    # model = DecisionTreeClassifier(random_state=42)
    model = SVR(verbose=True)
    # model = GaussianNB()
    # model = KNeighborsClassifier()
    # model = LogisticRegression(random_state=42, n_jobs=-1)
    # model = MLPClassifier()
    # model = AdaBoostClassifier(random_state=42, n_estimators=100)


    start = time()  # Get start time
    learner = model.fit(X_train, y_train.astype('int').values.ravel())
    end = time()

    predictions_train = learner.predict(X_train[:300].astype('int'))

    print(end - start)
    return model


def test_model(model, X_test, y_test, X_train, y_train):
    results = {}
    predictions_test = model.predict(X_test)
    predictions_train = model.predict(X_train)

    # y_train_helper = y_train.to_numpy()
    y_test_helper = y_test.to_numpy()
    y_train_helper = y_train.to_numpy()

    results['acc_test'] = accuracy_score(y_test_helper, predictions_test)
    results['acc_train'] = accuracy_score(y_train_helper, predictions_train)
    results['fbeta_test'] = fbeta_score(y_test_helper, predictions_test, beta=0.5)
    results['fbeta_train'] = fbeta_score(y_train_helper, predictions_train, beta=0.5)

    display = PrecisionRecallDisplay.from_estimator(
        model, X_test, y_test_helper, name="LinearSVC"
    )
    _ = display.ax_.set_title("2-class Precision-Recall curve")

    print(results)


def visualize_feature_importance(model, data_for_names):
    importance = model.feature_importances_
    col_names = []
    # summarize feature importance
    for col in data_for_names.columns:
        col_names.append(col)
        # print(col)

    # plot feature importance
    # pyplot.bar([x for x in range(len(importance))], importance)
    # pyplot.show()

    names = ['group_a', 'group_b', 'group_c']
    values = [1, 10, 100]

    names = col_names
    values = importance

    plt.figure(figsize=(9, 3))
    # plt.subplot(131)
    plt.barh(names, values)
    plt.suptitle('Categorical Plotting')
    plt.show()


if __name__ == '__main__':
    data = read_data()
    print(Style.RESET_ALL + 'read data')

    # remove nan values
    data = data.dropna()

    price_raw, features_raw = storePriceSeperate(data)
    print('store price seperate')
    features_raw = preProcessData(features_raw)
    print('pre process data')

    features_raw = removeUnusedColumns(features_raw)
    print('remove unused columns')

    # features_raw.to_csv(index=False)
    X_train, X_test, y_train, y_test = splitData(price_raw, features_raw)
    model = train_model(X_train, y_train)
    print('finished training')

    test_model(model, X_test, y_test, X_train, y_train)
    print('finished testing')

    visualize_feature_importance(model, X_train)
    print('finished visualizing')

    # HTTPServer((hostName, serverPort), HandleRequests).serve_forever()
    # print("Server started http://%s:%s" % (hostName, serverPort))

    running = True
    while running:
        descriptionInput = input('Enter description')
        if descriptionInput == 'exit':
            running = False
            break
        d = {'description': [descriptionInput]}
        df = pd.DataFrame(data=d)
        df = preProcessData(df)
        df = df.drop('description', axis=1)
        output = model.predict(df)
        # print(df)
        print(Fore.RED + str(output[0]))
        print(Style.RESET_ALL)
