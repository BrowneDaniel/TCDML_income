import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math


def preprocess(df, train):
    df["Low Population"] = 0
    for j in range(len(df)):
        if df["Size of City"][j] < 3000:
            df["Low Population"][j] = 1

    if train:
        df = df[np.isfinite(df["Age"])]
        df = df[np.isfinite(df["Year of Record"])]
    df = df.drop(["Size of City"], 1)
    df["Income in EUR"] = df["Income in EUR"].apply(np.log)
    return df


columns = ["Instance", "Year of Record", "Gender",
           "Age", "Country", "Size of City",
           "Profession", "University Degree", "Wears Glasses",
           "Hair Color", "Body Height [cm]", "Income in EUR"]

training_data = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")
test_data = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")
training_data = training_data.loc[:, columns]
test_data = test_data.loc[:, columns]

training_data = preprocess(training_data, True)
test_data = preprocess(test_data, False)
training_data = pd.get_dummies(training_data)
test_data = pd.get_dummies(test_data)

training_data.fillna(training_data.mean(), axis=0, inplace=True)
test_data.fillna(test_data.mean(), axis=0, inplace=True)
overlap_columns = [x for x in set(training_data.columns).intersection(test_data.columns)]
training_data = training_data[overlap_columns]
test_data = test_data[overlap_columns]

Y = training_data["Income in EUR"]
X = training_data.drop(["Income in EUR"], 1)
test_data = test_data.drop("Income in EUR", 1)

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regression = LinearRegression()
regression.fit(X, Y) # X_train, Y_train

#y_pred = regression.predict(X_test)
y_pred = regression.predict(test_data)
y_pred = np.exp(y_pred)
submissionframe = pd.read_csv("tcd ml 2019-20 income prediction submission file.csv")
for x in range(len(y_pred)):
    submissionframe["Income"][x] = y_pred[x]
submissionframe.to_csv("tcd ml 2019-20 income prediction submission file.csv", index=False)
#print("Root Mean squared error: " + str(math.sqrt(mean_squared_error(Y_test, y_pred))))
