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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from hyperopt import hp, fmin, tpe
import tensorflow as tf
from trieste import bayesian_optimizer, space
from trieste.objectives.utils import mk_observer
from trieste.models.gpflow import GaussianProcessRegression
import gpflow
import tensorflow_probability as tfp
from time import time


r"""
Trains XGBoost models with hyper-parameter tuning using Random Search, Grid Search, HYPEROPT, and Trieste. Saves the models in "results\models\tuned".
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
    r"""
    Trains XGBoost models with hyper-parameter tuning using Random Search, Grid Search, HYPEROPT, and Trieste. Saves the models in "results\models\tuned".
    """
    
    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")

    # Read the "uk_gov_data_dense_preproc" from the "vehicles" database using pandas.
    govData = pd.read_sql("SELECT * FROM uk_gov_data_dense_preproc", connection)

    # Creating array for training use.
    ctsFeatures = ["engine_size_cm3", "power_ps"]
    cateFeatures = ["manufacturer", "transmission", "transmission_type", "fuel", "powertrain"]
    features = ctsFeatures + cateFeatures
    target = "co2_emissions_gPERkm"
    
    # Define arrays: features (X_ordinal for ordinal encoding of categorical features), and target (y).
    X_ordinal = govData[features].copy()
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

    # Define training and testing arrays. Test sets are a hold-out to check models against after CV.
    testSize = 0.2
    trainX_ordinal, testX_ordinal, trainy, testy = train_test_split(X_ordinal, y, test_size = testSize, random_state=1)

    print("")
    print("*********************************************************************************************************************************************************************")
    print("XGB ORDINAL ENCODING HYPERPARAMETER TUNING REPORT")
    print("=====================================================================================================================================================================")

    print("Hyperparameter Tuning Approaches")
    print("")
    print("""I will train and compare the accuracies of XGB models obtained using 6 methods of hyperparameter tuning:
        (1) Grid Search
        (2) Random Search
        (3) HYPEROPT Bayesian Optimisation with Tree Parzen Estimation (TPE), using the same hyperparameter ranges as Random Search
        (4) HYPEROPT Bayesian Optimisation with TPE, tuning more hyperparameters for greater performance
        (5) Trieste Bayesian Optimisation with Gaussian Processes (GPFlow) and Expected Improvement (EI), using the same hyperparameter ranges as Random Search
        (6) Trieste Bayesian Optimisation with GPFlow and EI, tuning more hyperparameters for greater performance""")
    print("")
    print(r"Before model training, I shuffle the data and then split into training (80% of data) and testing (20% of data) sets, using a fixed seed for reproducibility.")
    print("In trainTestAnalyse.py, I compare the distributions of the training and testing sets to assess how well they represent the whole data.")
    print("Each method aims to minimise a k-fold cross-validation score on the training data in order to select optimal hyperparameters.")
    print("The testing set will act as a hold-out to emulate real-world data, and will be used as a final assessment for each model once hyperparameters are found.")
    print("")
    print("""The observed accuracies (R^2, Coefficient of Determination) of each method are as follows:
        Method                            Train Accuracy     Test Accuracy     Tuning Duration (s)
        ------------------------------------------------------------------------------------------
        (1) Grid Search                   0.961497           0.953565          6223
        (2) Random Search                 0.963479           0.953614          5406
        (3) HYPEROPT                      0.962829           0.954932          557
        (4) HYPEROPT more hyperparams     0.964786           0.956611          2283
        (5) Trieste                       0.964402           0.956253          786
        (6) Trieste more hyperparams      0.963918           0.956700          3120""")
    print("")
    print(r"""All the models perform very similarly. The test accuracies are very similar to the training accuracies (only slightly lower), indicating very little over-fitting.
        The greatest test accuracy is 95.67%, using hyperparameters found with Trieste and GPFlow.
        The lowest test accuracy is 95.36%, obtained using Grid Search.""")
    print("")
    print("""It is also worth reviewing the tuning durations. In this setup, Grid Search takes the longest to run and yields the lowest accuracy. Random Search provides a small
        speed improvement along with a tiny accuracy increase. Both Hyperopt and Trieste offer substantial speed increases as well as improvements to the model accuracies.""")
    print("")
    print("I review the model fitted using method (6) in modelReview.py.")
    print("")

    proceed = input("Please type (y) and then press Enter if you wish to proceed with tuning and training the models (could take a while):")
    if proceed=="y":
        print("Starting tuning...")
    else:
        print("Exiting...")
        print("*********************************************************************************************************************************************************************")
        raise SystemExit(0)

    print("=====================================================================================================================================================================")

    print("(1)")
    print("XGB - Grid Search")
    print("")

    # Hyper-parameters for Grid Search. Total number of combinations is 8748.
    params_grid = {
    "min_child_weight": [1, 5, 10],
    "alpha": [0, 0.5, 2],
    "gamma": [0.5, 1, 2, 3],
    "subsample": [0.4, 0.7, 1.0],
    "colsample_bytree": [0.4, 0.7, 1.0],
    "max_depth": [3, 5, 10],
    "learning_rate": [0.02, 0.1, 0.5],
    "n_estimators": [100, 300, 600]
    }

    # Configure Grid Search.
    folds = 3
    xgb_ordinal_grid_search = xgb.XGBRegressor(objective="reg:squarederror", verbosity=0, nthread=-1, seed=1)
    grid_search = GridSearchCV(xgb_ordinal_grid_search, param_grid=params_grid, scoring='neg_root_mean_squared_error', n_jobs=4, cv=folds, verbose=0)

    # Time duration of Grid Search.
    t1 = time()
    grid_search.fit(trainX_ordinal, trainy)
    t2 = time()
    print("Time elapsed in training: %f" % (t2-t1))
    # Time elapsed in training: 6223.165973901749

    params_grid_best = grid_search.best_params_
    print("Best random search hyperparameters: ")
    print(params_grid_best)
    # Best grid search hyperparameters:
    # {'alpha': 0, 'colsample_bytree': 0.7, 'gamma': 0.5, 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 10, 'n_estimators': 600, 'subsample': 1.0}

    # Create XGB using grid search's best found hyperparameters.
    xgb_ordinal_grid = xgb.XGBRegressor(**params_grid_best, objective="reg:squarederror", verbosity=0, nthread=-1, seed=1)
    xgb_ordinal_grid.fit(trainX_ordinal, trainy)
    trainPreds = xgb_ordinal_grid.predict(trainX_ordinal)
    testPreds = xgb_ordinal_grid.predict(testX_ordinal)
    print("Train RMSE: %f" % (np.sqrt(mean_squared_error(trainy, trainPreds))))
    print("Test RMSE: %f" % (np.sqrt(mean_squared_error(testy, testPreds))))
    print("Train Accuracy: %f" % (r2_score(trainy, trainPreds)))
    print("Test Accuracy: %f" % (r2_score(testy, testPreds)))
    # Train RMSE: 10.755259
    # Test RMSE: 11.978820
    # Train Accuracy: 0.961497
    # Test Accuracy: 0.953565

    xgb_ordinal_grid.save_model("./results/models/trained/XGB_ord_grid_search.json")
    results = pd.DataFrame(grid_search.cv_results_)
    results.to_csv("./results/models/tuned/XGB_ord_grid_search_results.csv", index=False)
    print("=====================================================================================================================================================================")

    print("(2)")
    print("XGB - Random Search")
    print("")

    # Hyperparameters for Random Search. Will randomly sample a subset of a more refined hyperparameter space than Grid Search.
    params_random = {
        "min_child_weight": [1, 5, 10],
        "alpha": [0, 0.5, 1, 1.5, 2, 5],
        "gamma": [0.5, 1, 1.5, 2, 5],
        "subsample": [0.4, 0.6, 0.8, 1.0],
        "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5],
        "n_estimators": [100, 300, 600]
        }
    
    # Configure Random Search. Use number of hyperparameter combinations equal to Grid Search for fairness of comparison.
    folds = 3
    param_comb = 8748
    xgb_ordinal_random_search = xgb.XGBRegressor(objective="reg:squarederror", verbosity=0, nthread=-1, seed=1)
    random_search = RandomizedSearchCV(xgb_ordinal_random_search, param_distributions=params_random, n_iter=param_comb,
                                       scoring='neg_root_mean_squared_error', n_jobs=4, cv=folds, verbose=0, random_state=1)

    # Time duration of random search.
    t3 = time()
    random_search.fit(trainX_ordinal, trainy)
    t4 = time()
    print("Time elapsed in training: %f" % (t4-t3))
    # Time elapsed in training: 5406.3555743694305

    params_random_best = random_search.best_params_
    print("Best random search hyperparameters: ")
    print(params_random_best)
    # Best random search hyperparameters:
    # {'subsample': 1.0, 'n_estimators': 600, 'min_child_weight': 10, 'max_depth': 4, 'learning_rate': 0.2, 'gamma': 0.5, 'colsample_bytree': 1.0, 'alpha': 1.5}
    # Best random search RMSE: 12.654229460570724

    # Create XGB using random search's best found hyperparameters.
    xgb_ordinal_random = xgb.XGBRegressor(**params_random_best, objective="reg:squarederror", verbosity=0, nthread=-1, seed=1)
    xgb_ordinal_random.fit(trainX_ordinal, trainy)
    trainPreds = xgb_ordinal_random.predict(trainX_ordinal)
    testPreds = xgb_ordinal_random.predict(testX_ordinal)
    print("Train RMSE: %f" % (np.sqrt(mean_squared_error(trainy, trainPreds))))
    print("Test RMSE: %f" % (np.sqrt(mean_squared_error(testy, testPreds))))
    print("Train Accuracy: %f" % (r2_score(trainy, trainPreds)))
    print("Test Accuracy: %f" % (r2_score(testy, testPreds)))
    # Train RMSE: 10.474755
    # Test RMSE: 11.972508
    # Train Accuracy: 0.963479
    # Test Accuracy: 0.953614

    xgb_ordinal_random.save_model("./results/models/trained/XGB_ord_random_search.json")
    results = pd.DataFrame(random_search.cv_results_)
    results.to_csv("./results/models/tuned/XGB_ord_random_search_results.csv", index=False)
    print("=====================================================================================================================================================================")

    print("(3)")
    print("XGB - HYPEROPT + TPE Bayesian Optimisation, Same Hyperparameters as Grid/Random Search")
    print("")

    # Hyperparameters for HYPEROPT Bayesian Optimisation (with TPE).
    # Use same hyperparameters and ranges as for grid/random search for fairness of comparison.
    params_hyperopt={
        "max_depth": hp.choice("max_depth", np.arange(3, 7+1, dtype=int)),
        "gamma": hp.uniform("gamma", 0.5, 5),
        "reg_alpha": hp.uniform("reg_alpha", 0, 5),
        "subsample": hp.uniform("subsample", 0.4, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.4, 1),
        "min_child_weight": hp.choice("min_child_weight", np.arange(1, 10+1, dtype=int)),
        "learning_rate":  hp.uniform ("learning_rate", 0.01, 0.5), 
        "n_estimators": hp.choice("n_estimators", np.arange(100, 600+1, dtype=int))
    }
    
    def objective_hyperopt(space):
        """
        Objective function, taking set of hyperparameters and returning mean validation accuracy of k-fold cross-validation.
        """
        
        xgb_ordinal_hyperopt = xgb.XGBRegressor(**space, objective="reg:squarederror", verbosity=0, nthread=-1, seed=1)
        xgb_ordinal_hyperopt.fit(trainX_ordinal, trainy)
        cv_score = cross_val_score(xgb_ordinal_hyperopt, trainX_ordinal, trainy, scoring="neg_root_mean_squared_error", cv=10)
        return (-sum(cv_score)/len(cv_score))
    
    # Time duration of HYPEROPT. Give HYPEROPT only 100 evaluations (compared to grid/random search's 8748) to demonstrate the superior performance due to Bayesian Optimisation.
    rstate = np.random.default_rng(1)   # Reproducibility.
    t5 = time()
    # return_argmin=False is essential for obtaining the correct best hyperparameters: https://github.com/hyperopt/hyperopt/issues/530
    params_hyperopt_best = fmin(fn = objective_hyperopt, space = params_hyperopt, algo = tpe.suggest, max_evals = 100, rstate=rstate, return_argmin=False)
    t6 = time()
    print("Time elapsed in training: %f" % (t6-t5))
    # 100% ... 100/100 [09:17<00:00,  5.57s/trial, best loss: 12.183767804749547]
    # Time elapsed in training: 557.220675
    
    # Create XGB using HYPEROPT's best found hyperparameters.
    xgb_ordinal_hyperopt_best = xgb.XGBRegressor(**params_hyperopt_best, objective="reg:squarederror", verbosity=0, nthread=-1, seed=1)
    xgb_ordinal_hyperopt_best.fit(trainX_ordinal, trainy)
    trainPreds = xgb_ordinal_hyperopt_best.predict(trainX_ordinal)
    testPreds = xgb_ordinal_hyperopt_best.predict(testX_ordinal)

    print("Best HYPEROPT hyperparameters: ")
    print(params_hyperopt_best)
    # {'colsample_bytree': 0.6597818584110863, 'gamma': 1.7700638017993198, 'learning_rate': 0.16348928243486646, 'max_depth': 6,
    # 'min_child_weight': 9, 'n_estimators': 407, 'reg_alpha': 0.1535235589281927, 'subsample': 0.5063560817838623}

    print("Train RMSE: %f" % (np.sqrt(mean_squared_error(trainy, trainPreds))))
    print("Test RMSE: %f" % (np.sqrt(mean_squared_error(testy, testPreds))))
    print("Train Accuracy: %f" % (r2_score(trainy, trainPreds)))
    print("Test Accuracy: %f" % (r2_score(testy, testPreds)))
    # Train RMSE: 10.567621
    # Test RMSE: 11.801092
    # Train Accuracy: 0.962829
    # Test Accuracy: 0.954932

    xgb_ordinal_hyperopt_best.save_model("./results/models/trained/XGB_ord_hyperopt.json")
    print("=====================================================================================================================================================================")

    print("(4)")
    print("XGB - HYPEROPT + TPE Bayesian Optimisation, More Hyperparameters")
    print("")

    # Hyper-parameters for HYPEROPT Bayesian Optimisation (with TPE).
    params_hyperopt_more={
        "max_depth": hp.choice("max_depth", np.arange(3, 18+1, dtype=int)),
        "gamma": hp.uniform("gamma", 0.05, 20),
        "reg_alpha": hp.uniform("reg_alpha", 0, 20),
        "reg_lambda": hp.uniform("reg_lambda", 0, 20),
        "subsample": hp.uniform("subsample", 0.1, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.1, 1),
        "colsample_bylevel": hp.uniform("colsample_bylevel", 0.1, 1),
        "colsample_bynode": hp.uniform("colsample_bynode", 0.1, 1),
        "min_child_weight": hp.choice("min_child_weight", np.arange(0, 10+1, dtype=int)),
        "learning_rate":  hp.uniform ("learning_rate", 0.005, 1), 
        "n_estimators": hp.choice("n_estimators", np.arange(10, 1000+1, dtype=int))
    }
    
    def objective_hyperopt_more(space):
        """
        Objective function, taking set of hyperparameters and returning mean validation accuracy of k-fold cross-validation.
        """
        
        xgb_ordinal_hyperopt_more = xgb.XGBRegressor(**space, objective="reg:squarederror", verbosity=0, nthread=-1, seed=1)
        xgb_ordinal_hyperopt_more.fit(trainX_ordinal, trainy)
        cv_score = cross_val_score(xgb_ordinal_hyperopt_more, trainX_ordinal, trainy, scoring="neg_root_mean_squared_error", cv=10)
        return (-sum(cv_score)/len(cv_score))
    
    # Time duration of HYPEROPT.
    rstate = np.random.default_rng(1)   # Reproducibility.
    t5 = time()
    # return_argmin=False is essential for obtaining the correct best hyperparameters: https://github.com/hyperopt/hyperopt/issues/530
    params_hyperopt_more_best = fmin(fn = objective_hyperopt_more, space = params_hyperopt_more, algo = tpe.suggest, max_evals = 300, rstate=rstate, return_argmin=False)
    t6 = time()
    print("Time elapsed in training: %f" % (t6-t5))
    # 100% ... 300/300 [38:03<00:00,  7.61s/trial, best loss: 12.154993299522264]
    # Time elapsed in training: 2283.569938

    # Create XGB using HYPEROPTS's best found hyperparameters.
    xgb_ordinal_hyperopt_more_best = xgb.XGBRegressor(**params_hyperopt_more_best, objective="reg:squarederror", verbosity=0, nthread=-1, seed=1)
    xgb_ordinal_hyperopt_more_best.fit(trainX_ordinal, trainy)
    trainPreds = xgb_ordinal_hyperopt_more_best.predict(trainX_ordinal)
    testPreds = xgb_ordinal_hyperopt_more_best.predict(testX_ordinal)

    print("Best HYPEROPT hyperparameters: ")
    print(params_hyperopt_more_best)
    # Best HYPEROPT hyperparameters:
    # {'colsample_bylevel': 0.7959522883412514, 'colsample_bynode': 0.5887365734644787, 'colsample_bytree': 0.5206615408214966, 'gamma': 4.2522350513885865,
    # 'learning_rate': 0.24702299052479343, 'max_depth': 18, 'min_child_weight': 5, 'n_estimators': 854, 'reg_alpha': 5.139231471392408,
    # 'reg_lambda': 9.537270700292027, 'subsample': 0.8959962452852309}

    print("Train RMSE: %f" % (np.sqrt(mean_squared_error(trainy, trainPreds))))
    print("Test RMSE: %f" % (np.sqrt(mean_squared_error(testy, testPreds))))
    print("Train Accuracy: %f" % (r2_score(trainy, trainPreds)))
    print("Test Accuracy: %f" % (r2_score(testy, testPreds)))
    # Train RMSE: 10.285699
    # Test RMSE: 11.579282
    # Train Accuracy: 0.964786
    # Test Accuracy: 0.956611

    xgb_ordinal_hyperopt_more_best.save_model("./results/models/trained/XGB_ord_hyperopt_more.json")
    print("=====================================================================================================================================================================")

    print("(5)")
    print("XGB - Trieste + Gaussian Process (GPFlow) Bayesian Optimisation, Same Hyperparameters as Grid/Random Search")
    print("")

    def objective_trieste(space):
        """
        Objective function, taking set of hyperparameters and returning mean validation accuracy of k-fold cross-validation.
        """
        
        cv_scores = []
        for i in range(0,space.numpy().shape[0]):
            # Hyper-parameters for Trieste Bayesian Optimisation (with GP).
            # Use same hyperparameters and ranges as for grid/random search for fairness of comparison.
            xgb_ordinal_trieste = xgb.XGBRegressor(
                **{"gamma": space.numpy()[i][0],
                   "reg_alpha": space.numpy()[i][1],
                   "subsample": space.numpy()[i][2],
                   "colsample_bytree": space.numpy()[i][3],
                   "learning_rate": space.numpy()[i][4],
                   "max_depth": space.numpy()[i][5].astype(int),
                   "min_child_weight": space.numpy()[i][6].astype(int),
                   "n_estimators": space.numpy()[i][7].astype(int),
                  },
                objective="reg:squarederror", verbosity=0, nthread=-1, seed=1)
            xgb_ordinal_trieste.fit(trainX_ordinal, trainy)
            cv_k_scores = cross_val_score(xgb_ordinal_trieste, trainX_ordinal, trainy, scoring="neg_root_mean_squared_error", cv=10)
            cv_scores.append( [-sum(cv_k_scores)/len(cv_k_scores)] )
        return tf.convert_to_tensor(cv_scores, dtype=tf.float64, dtype_hint=None, name=None)
    
    # Time duration of Trieste. Give Trieste only 100 evaluations (compared to grid/random search's 8748) to demonstrate the superior performance due to Bayesian Optimisation.
    t5 = time()
    observer = mk_observer(objective=objective_trieste)

    # Continuous search space/box for [gamma, reg_alpha, subsample, colsample_bytree, learning_rate]
    # Discrete hyperparameters are max_depth, min_child_weight, n_estimators.
    search_space = space.TaggedProductSearchSpace(
        [space.Box([0.5, 0, 0.4, 0.4, 0.01], [5, 5, 1, 1, 0.5]),
         space.DiscreteSearchSpace(tf.constant(np.arange(3,7+1, dtype=float).reshape(-1,1))),
         space.DiscreteSearchSpace(tf.constant(np.arange(1,10+1, dtype=float).reshape(-1,1))),
         space.DiscreteSearchSpace(tf.constant(np.arange(100,600+1, dtype=float).reshape(-1,1)))]
    )

    # Sample initial objective function points.
    num_initial_points = 20
    initial_query_points = search_space.sample(num_initial_points)
    initial_data = observer(initial_query_points)

    # Construct GPFlow model.
    def build_model(data):
        variance = tf.math.reduce_variance(data.observations)
        # Using lengthscales approximately 20% of max possible hyperparameter values appears reasonable.
        kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[1, 1, 0.2, 0.2, 0.1, 1, 2, 100])
        prior_scale = tf.cast(1.0, dtype=tf.float64)
        kernel.variance.prior = tfp.distributions.LogNormal(
            tf.cast(-2.0, dtype=tf.float64), prior_scale
        )
        kernel.lengthscales.prior = tfp.distributions.LogNormal(
            tf.math.log(kernel.lengthscales), prior_scale
        )
        gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
        gpflow.set_trainable(gpr.likelihood, False)
        return GaussianProcessRegression(gpr, num_kernel_samples=100)
    model = build_model(initial_data)

    # Define Trieste optimizer and optimize hyperparameters.
    trieste_bo = bayesian_optimizer.BayesianOptimizer(observer=observer, search_space=search_space)
    num_steps = 80
    result = trieste_bo.optimize(num_steps, initial_data, model)
    dataset = result.try_get_final_dataset()

    # Retrieve best hyperparameters.
    query_points = dataset.query_points.numpy()
    observations = dataset.observations.numpy()
    arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))
    best_point = query_points[arg_min_idx, :]
    params_trieste_best = {"gamma": best_point[0],
                           "reg_alpha": best_point[1],
                           "subsample": best_point[2],
                           "colsample_bytree": best_point[3],
                           "learning_rate": best_point[4],
                           "max_depth": best_point[5].astype(int),
                           "min_child_weight": best_point[6].astype(int),
                           "n_estimators": best_point[7].astype(int),
                          }

    t6 = time()
    print("Time elapsed in training: %f" % (t6-t5))
    # Time elapsed in training: 786.772372

    # Create XGB using Trieste's best found hyperparameters.
    xgb_ordinal_trieste_best = xgb.XGBRegressor(**params_trieste_best, objective="reg:squarederror", verbosity=0, nthread=-1, seed=1)
    xgb_ordinal_trieste_best.fit(trainX_ordinal, trainy)
    trainPreds = xgb_ordinal_trieste_best.predict(trainX_ordinal)
    testPreds = xgb_ordinal_trieste_best.predict(testX_ordinal)

    print("Best Trieste hyperparameters: ")
    print(params_trieste_best)
    # Best Trieste hyperparameters:
    # {'gamma': 0.5, 'reg_alpha': 5.0, 'subsample': 1.0, 'colsample_bytree': 0.4, 'learning_rate': 0.5,
    # 'max_depth': 7, 'min_child_weight': 10, 'n_estimators': 581}

    print("Train RMSE: %f" % (np.sqrt(mean_squared_error(trainy, trainPreds))))
    print("Test RMSE: %f" % (np.sqrt(mean_squared_error(testy, testPreds))))
    print("Train Accuracy: %f" % (r2_score(trainy, trainPreds)))
    print("Test Accuracy: %f" % (r2_score(testy, testPreds)))
    # Train RMSE: 10.341544
    # Test RMSE: 11.626880
    # Train Accuracy: 0.964402
    # Test Accuracy: 0.956253

    xgb_ordinal_trieste_best.save_model("./results/models/trained/XGB_ord_trieste.json")
    print("=====================================================================================================================================================================")

    print("(6)")
    print("XGB - Trieste + Gaussian Process (GPFlow) Bayesian Optimisation, More Hyperparameters")
    print("")

    def objective_trieste_more(space):
        """
        Objective function, taking set of hyperparameters and returning mean validation accuracy of k-fold cross-validation.
        """

        cv_scores = []
        for i in range(0,space.numpy().shape[0]):
            # Hyper-parameters for Trieste Bayesian Optimisation (with GP).
            xgb_ordinal_trieste_more = xgb.XGBRegressor(
                **{"gamma": space.numpy()[i][0],
                   "reg_alpha": space.numpy()[i][1],
                   "reg_lambda": space.numpy()[i][2],
                   "subsample": space.numpy()[i][3],
                   "colsample_bytree": space.numpy()[i][4],
                   "colsample_bylevel": space.numpy()[i][5],
                   "colsample_bynode": space.numpy()[i][6],
                   "learning_rate": space.numpy()[i][7],
                   "max_depth": space.numpy()[i][8].astype(int),
                   "min_child_weight": space.numpy()[i][9].astype(int),
                   "n_estimators": space.numpy()[i][10].astype(int),
                  },
                objective="reg:squarederror", verbosity=0, nthread=-1, seed=1)
            xgb_ordinal_trieste_more.fit(trainX_ordinal, trainy)
            cv_k_scores = cross_val_score(xgb_ordinal_trieste_more, trainX_ordinal, trainy, scoring="neg_root_mean_squared_error", cv=10)
            cv_scores.append( [-sum(cv_k_scores)/len(cv_k_scores)] )
        return tf.convert_to_tensor(cv_scores, dtype=tf.float64, dtype_hint=None, name=None)
    
    # Time duration of Trieste.
    t5 = time()
    observer = mk_observer(objective=objective_trieste_more)

    # Continuous search space/box for [gamma, reg_alpha, reg_lambda, subsample, colsample_bytree, colsample_bylevel, colsample_bynode, learning_rate]
    # Discrete hyperparameters are max_depth, min_child_weight, n_estimators.
    search_space = space.TaggedProductSearchSpace(
        [space.Box([0.05, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.005], [20, 20, 20, 1, 1, 1, 1, 1]),
         space.DiscreteSearchSpace(tf.constant(np.arange(3,18+1, dtype=float).reshape(-1,1))),
         space.DiscreteSearchSpace(tf.constant(np.arange(0,10+1, dtype=float).reshape(-1,1))),
         space.DiscreteSearchSpace(tf.constant(np.arange(10,1000+1, dtype=float).reshape(-1,1)))]
    )

    # Sample initial objective function points.
    num_initial_points = 20
    initial_query_points = search_space.sample(num_initial_points)
    initial_data = observer(initial_query_points)

    # Construct GPFlow model.
    def build_model_more(data):
        variance = tf.math.reduce_variance(data.observations)
        # Using lengthscales approximately 20% of max possible hyperparameter values appears reasonable.
        kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[4, 4, 4, 0.2, 0.2, 0.2, 0.2, 0.2, 3, 2, 200])
        prior_scale = tf.cast(1.0, dtype=tf.float64)
        kernel.variance.prior = tfp.distributions.LogNormal(
            tf.cast(-2.0, dtype=tf.float64), prior_scale
        )
        kernel.lengthscales.prior = tfp.distributions.LogNormal(
            tf.math.log(kernel.lengthscales), prior_scale
        )
        gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
        gpflow.set_trainable(gpr.likelihood, False)
        return GaussianProcessRegression(gpr, num_kernel_samples=100)
    model = build_model_more(initial_data)

    # Define Trieste optimizer and optimize hyperparameters.
    trieste_bo = bayesian_optimizer.BayesianOptimizer(observer=observer, search_space=search_space)
    num_steps = 280
    result = trieste_bo.optimize(num_steps, initial_data, model)
    dataset = result.try_get_final_dataset()

    # Retrieve best hyperparameters.
    query_points = dataset.query_points.numpy()
    observations = dataset.observations.numpy()
    arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))
    best_point_more = query_points[arg_min_idx, :]
    params_trieste_more_best = {"gamma": best_point_more[0],
                                "reg_alpha": best_point_more[1],
                                "reg_lambda": best_point_more[2],
                                "subsample": best_point_more[3],
                                "colsample_bytree": best_point_more[4],
                                "colsample_bylevel": best_point_more[5],
                                "colsample_bynode": best_point_more[6],
                                "learning_rate": best_point_more[7],
                                "max_depth": best_point_more[8].astype(int),
                                "min_child_weight": best_point_more[9].astype(int),
                                "n_estimators": best_point_more[10].astype(int),
                          }

    t6 = time()
    print("Time elapsed in training: %f" % (t6-t5))
    # Time elapsed in training: 3120.666984

    # Create XGB using Trieste's best found hyperparameters.
    xgb_ordinal_trieste_more_best = xgb.XGBRegressor(**params_trieste_more_best, objective="reg:squarederror", verbosity=0, nthread=-1, seed=1)
    xgb_ordinal_trieste_more_best.fit(trainX_ordinal, trainy)
    trainPreds = xgb_ordinal_trieste_more_best.predict(trainX_ordinal)
    testPreds = xgb_ordinal_trieste_more_best.predict(testX_ordinal)

    print("Best Trieste hyperparameters: ")
    print(params_trieste_more_best)
    # Best Trieste hyperparameters:
    # {'gamma': 1.385229919602636, 'reg_alpha': 17.33424059817828, 'reg_lambda': 18.28910938536196, 'subsample': 0.885860728296831,
    # 'colsample_bytree': 0.980718423322313, 'colsample_bylevel': 0.9448228957956979, 'colsample_bynode': 0.2456709938043513,
    # 'learning_rate': 0.24339192476833982, 'max_depth': 16, 'min_child_weight': 6, 'n_estimators': 884}

    print("Train RMSE: %f" % (np.sqrt(mean_squared_error(trainy, trainPreds))))
    print("Test RMSE: %f" % (np.sqrt(mean_squared_error(testy, testPreds))))
    print("Train Accuracy: %f" % (r2_score(trainy, trainPreds)))
    print("Test Accuracy: %f" % (r2_score(testy, testPreds)))
    # Train RMSE: 10.411563
    # Test RMSE: 11.567392
    # Train Accuracy: 0.963918
    # Test Accuracy: 0.956700

    xgb_ordinal_trieste_more_best.save_model("./results/models/trained/XGB_ord_trieste_more.json")
    print("*********************************************************************************************************************************************************************")    

main()