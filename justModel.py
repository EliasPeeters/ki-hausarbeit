import pandas as pd
import mainNewOhneText
from sklearn.model_selection import train_test_split

# check if running in main

if __name__ == '__main__':
    # load features_raw
    features_raw = pd.read_csv('preProcessed/features_raw.csv')

    # load price_raw
    price_raw = pd.read_csv('preProcessed/price_raw.csv')

    # split data into training and testing
    X_train, X_test, y_train, y_test = mainNewOhneText.splitData(price_raw, features_raw)

    print('start training')
    model = mainNewOhneText.train_model(X_train, y_train)
    print('finished training')

    mainNewOhneText.test_model(model, X_test, y_test, X_train, y_train)
    print('finished testing')

    mainNewOhneText.visualize_feature_importance(model, X_train)
    print('finished visualizing')

