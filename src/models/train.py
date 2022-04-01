#    Copyright 2022 Reuben Owen-Williams

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys
import os
sys.path.append(os.getcwd())
from src.data.dbLogin import configprivate
import mysql.connector
from mysql.connector import Error
from ordered_set import OrderedSet
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor

"""
Trains XGBoost, Scikit Decision Tree, and Scikit Linear Regression models on the "uk_gov_data_dense_preproc" data, and saves the models in "results\models\trained".
"""

def create_database_connection(host_name, user_name, user_password, database):
    """
    Returns a connection to the database "vehicles" in the local MySQL server.
    """

    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=database
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")

    return connection

def main():
    """
    Trains XGBoost, Scikit Decision Tree, and Scikit Linear Regression models on the "uk_gov_data_dense_preproc" data, and saves the models in "results\models\trained".
    """
    
    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")

    # Read the "uk_gov_data_dense_preproc" from the "vehicles" database using pandas.
    govData = pd.read_sql("SELECT * FROM uk_gov_data_dense_preproc", connection)

    # Creating array for training use.
    # Manufacturer is excluded from the features, the idea being that the design ethos and balance in priority should be reflected in the other features... DECIDE IF KEEPING MANUFACTURER
    ctsFeatures = ["engine_size_cm3", "power_ps"]
    cateFeatures = ["manufacturer", "transmission", "transmission_type", "fuel", "powertrain"]
    features = ctsFeatures + cateFeatures
    target = "co2_emissions_gPERkm"
    
    # Define arrays: features (X_ordinal for ordinal encoding of categorical features, X_onehot for one hot encoding), and target (y).
    X_ordinal = govData[features].copy()
    X_onehot = govData[features].copy()
    y = govData[target]

    # Ordinal encoding of categorical features.
    cateDict = {}
    for category in cateFeatures:
        value = 0
        cateDict[category] = dict()
        for option in OrderedSet(X_ordinal[category]):
            cateDict[category][option] = value
            value += 1
        X_ordinal[category] = X_ordinal[category].map(cateDict[category])

    # One hot encoding of categorical features - remembering that one identifier column is removed per category so that columns are independent.
    cateOneHot = {}
    for category in cateFeatures:
        value = 0
        cateOneHot[category] = dict()  # Dictionary not used here, just for input validation later if required
        for option in OrderedSet(X_onehot[category]):
            cateOneHot[category][option] = value
            value += 1
        X_onehot = pd.get_dummies(data=X_onehot, drop_first=True, columns = [category])

    # Define training and testing arrays.
    testSize = 0.2
    trainX_ordinal, testX_ordinal, trainy, testy = train_test_split(X_ordinal, y, test_size = testSize, random_state=1)
    trainX_onehot, testX_onehot, trainy, testy = train_test_split(X_onehot, y, test_size = testSize, random_state=1)

    print("")
    print("*********************************************************************************************************************************************************************")
    print("MODEL TRAINING REPORT")
    print("=====================================================================================================================================================================")

    print("Models Summary")
    print("")
    print("""I will train and compare the accuracies of the following 4 models, without tuning hyperparameters at this stage:
        (1) XGB with Ordinal Encoding of categorical features
        (2) XGB with One-Hot Encoding of categorical features
        (3) Scikit Decision Tree - Ordinal
        (4) Scikit Linear Regression - One-Hot""")
    print("")
    print(r"Before model training, I shuffle the data and then split into training (80% of data) and testing (20% of data) sets, using a fixed seed for reproducibility.")
    print("In trainTestAnalyse.py, I compare the distributions of the training and testing sets to assess how well they represent the whole data.")
    print("""In training each model on the training set, I will provide the following hyperparameters as applicable...
        objective = reg:squarederror, learning_rate = 0.1, max_depth = 10, n_estimators = 50""")
    print("The testing set will act as a hold-out to emulate real-world data, and will be used as a final assessment for each model once trained.")
    print("")
    print("""The observed accuracies (R^2, Coefficient of Determination) of each model are as follows:
        Model                                      Train Accuracy     Test Accuracy
        ---------------------------------------------------------------------------
        (1) XGB - Ordinal                          0.962262           0.945486
        (2) XGB - One-Hot                          0.959849           0.950118
        (3) Scikit Decision Tree - Ordinal         0.937138           0.932501
        (4) Scikit Linear Regression - One-Hot     0.886677           0.885848""")
    print("")
    print("The two XGB models seem to perform approximately equally, with One-Hot encoding providing a slight edge over Ordinal encoding.")
    print("""I pursue tuning the hyperparameters for both types of XGB model in trainTunedOrdinal.py and trainTunedOnehot.py using Trieste's Bayesian Optimisation with a
        Gaussian Process (GPFlow), and discover that Ordinal encoding provides a very slight edge over One-Hot encoding.""")
    print("")
    print("Also of note is that the test accuracies are very similar to the training accuracies (only slightly lower), indicating very little over-fitting.")
    print("=====================================================================================================================================================================")

    print("(1)")
    print("XGB - Ordinal")
    print("")

    # Define hyperparameters and fit XGB with Ordinal encoding, and calculate predictions on training and testing data.
    xgb_ordinal = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.1, max_depth = 10, n_estimators = 50, seed=1)
    xgb_ordinal.fit(trainX_ordinal,trainy)
    trainPreds = xgb_ordinal.predict(trainX_ordinal)
    testPreds = xgb_ordinal.predict(testX_ordinal)

    # Calculate RMSE of XGB on training and testing data.
    print("Train RMSE: %f" % (np.sqrt(mean_squared_error(trainy, trainPreds))))
    print("Test RMSE: %f" % (np.sqrt(mean_squared_error(testy, testPreds))))
    print("Train Accuracy: %f" % (r2_score(trainy, trainPreds)))
    print("Test Accuracy: %f" % (r2_score(testy, testPreds)))
    # Train RMSE: 11.563080
    # Test RMSE: 12.315796
    # Train Accuracy: 0.955496
    # Test Accuracy: 0.950915

    xgb_ordinal.save_model("./results/models/trained/XGB_ord.json")
    print("=====================================================================================================================================================================")

    print("(2)")
    print("XGB - One Hot")
    print("")

    # Define hyperparameters and fit XGB with One-Hot encoding, and calculate predictions on training and testing data.
    xgb_onehot = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.1, max_depth = 10, n_estimators = 50, seed=1)
    xgb_onehot.fit(trainX_onehot,trainy)
    trainPreds = xgb_onehot.predict(trainX_onehot)
    testPreds = xgb_onehot.predict(testX_onehot)

    # Calculate RMSE of XGB on training and testing data.
    print("Train RMSE: %f" % (np.sqrt(mean_squared_error(trainy, trainPreds))))
    print("Test RMSE: %f" % (np.sqrt(mean_squared_error(testy, testPreds))))
    print("Train Accuracy: %f" % (r2_score(trainy, trainPreds)))
    print("Test Accuracy: %f" % (r2_score(testy, testPreds)))
    # Train RMSE: 17.515802
    # Test RMSE: 17.962747
    # Train Accuracy: 0.897879
    # Test Accuracy: 0.895584

    xgb_onehot.save_model("./results/models/trained/XGB_onehot.json")
    print("=====================================================================================================================================================================")

    print("(3)")
    print("Scikit Decision Tree - Ordinal")
    print("")

    # Define hyperparameters and fit Scikit's decision tree with Ordinal encoding, and calculate predictions on training and testing data.
    decTree = DecisionTreeRegressor(max_depth = 10)
    decTree = decTree.fit(trainX_ordinal,trainy)
    trainPreds = decTree.predict(trainX_ordinal)
    testPreds = decTree.predict(testX_ordinal)

    # Calculate RMSE of decision tree on training and testing data.
    print("Train RMSE: %f" % (np.sqrt(mean_squared_error(trainy, trainPreds))))
    print("Test RMSE: %f" % (np.sqrt(mean_squared_error(testy, testPreds))))
    print("Train Accuracy: %f" % (r2_score(trainy, trainPreds)))
    print("Test Accuracy: %f" % (r2_score(testy, testPreds)))
    # Train RMSE: 13.742548
    # Test RMSE: 14.442383
    # Train Accuracy: 0.937138
    # Test Accuracy: 0.932501

    dump(decTree, "./results/models/trained/DT_ord.joblib")
    print("=====================================================================================================================================================================")

    print("(4)")
    print("Linear Regression - One Hot")
    print("")

    # Define hyperparameters and fit Scikit's linear regression with One-Hot encoding (necessary), and calculate predictions on training and testing data.
    regr = linear_model.LinearRegression()
    regr.fit(trainX_onehot, trainy)

    trainPreds = regr.predict(trainX_onehot)
    testPreds = regr.predict(testX_onehot)

    # Calculate RMSE oflinear regression on training and testing data.
    print("Train RMSE: %f" % (np.sqrt(mean_squared_error(trainy, trainPreds))))
    print("Test RMSE: %f" % (np.sqrt(mean_squared_error(testy, testPreds))))
    print("Train Accuracy: %f" % (r2_score(trainy, trainPreds)))
    print("Test Accuracy: %f" % (r2_score(testy, testPreds)))
    # Train RMSE: 18.451528
    # Test RMSE: 18.781536
    # Train Accuracy: 0.886677
    # Test Accuracy: 0.885848

    dump(regr, "./results/models/trained/Linreg_onehot.joblib")
    print("*********************************************************************************************************************************************************************")    

main()