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
import tensorflow as tf
from trieste import bayesian_optimizer, space
from trieste.objectives.utils import mk_observer
from trieste.models.gpflow import GaussianProcessRegression
import gpflow
import tensorflow_probability as tfp
from time import time


r"""
Trains XGBoost model with One-Hot encoding and hyperparameter tuning using Trieste. Saves the model in "results\models\tuned".
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
    Trains XGBoost model with One-Hot encoding and hyperparameter tuning using Trieste. Saves the model in "results\models\tuned".
    """
    
    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")

    # Read the "uk_gov_data_dense_preproc" from the "vehicles" database using pandas.
    govData = pd.read_sql("SELECT * FROM uk_gov_data_dense_preproc", connection)

    # Creating array for training use.
    ctsFeatures = ["engine_size_cm3", "power_ps"]
    cateFeatures = ["manufacturer", "transmission", "transmission_type", "fuel", "powertrain"]
    features = ctsFeatures + cateFeatures
    target = "co2_emissions_gPERkm"
    
    # Define arrays: features (X_onehot for one hot encoding), and target (y).
    X_onehot = govData[features].copy()
    y = govData[target]

    # One hot encoding of categorical features - remembering that one identifier column is removed per category so that columns are independent.
    cateOneHot = {}
    for category in cateFeatures:
        value = 0   #----------------- Dictionary not used here, just for input validation
        cateOneHot[category] = dict()
        for option in OrderedSet(X_onehot[category]):
            cateOneHot[category][option] = value
            value += 1
        X_onehot = pd.get_dummies(data=X_onehot, drop_first=True, columns = [category])        

    # Define training and testing arrays. Test sets are a hold-out to check models against after CV.
    testSize = 0.2
    trainX_onehot, testX_onehot, trainy, testy = train_test_split(X_onehot, y, test_size = testSize, random_state=1)

    print("")
    print("*********************************************************************************************************************************************************************")
    print("XGB ONE-HOT ENCODING HYPERPARAMETER TUNING REPORT")
    print("=====================================================================================================================================================================")

    print("Hyperparameter Tuning Approach")
    print("")
    print("In tuning hyperparameters for an XGB model using One-Hot encoding, I use Trieste Bayesian optimization with Gaussian Processes (GPFlow) and Expected Improvement (EI).")
    print("I will then compare the model accuracy produce with this method with (5) in trainTunedOrdinal.py.")
    print("The procurement of train/test data and the use of cross-validation is the same as in trainTunedOrdinal.py.")
    print("I allow 100 evaluations and use the same hyperparameter ranges as (5) (so the only difference is the encoding) for fair comparison.")
    print("")
    print("""The observed accuracies (R^2, Coefficient of Determination) of the method here and of (5) in trainTunedOrdinal.py are:
        Method                    Train Accuracy     Test Accuracy     Tuning Duration (s)
        ----------------------------------------------------------------------------------
        Trieste - One-Hot         0.963519           0.955957          1812
        (5) Trieste - Ordinal     0.964402           0.956253          786""")

    print("")
    print("The models perform approximately equally, with Ordinal encoding (95.63%) having a marginally higher accuracy than One-Hot encoding (95.60%) in this case.")
    print("")
    print("""Considering XGB documentation states there is not full support for categorical variables, just for partition or One-Hot encoded variables, this result might be unexpected:
        https://xgboost.readthedocs.io/en/latest/tutorials/categorical.html""")
    print("""However, the cardinality of the categorical variables is not that high (greatest number of options is 42 for "transmission"), so perhaps XGB was still able to sufficiently
        split the encoded categories to identify patterns, possibly explaining why the Ordinal model still performed well.""")
    print("")
    print("""Also worth noting is that tuning hyperparameters took much longer for the One-Hot model than for the Ordinal method, at 1812s vs 485s.
        This is likely due to the One-Hot training data becoming a much larger, sparse data set, and as a result model evaluations (obtaining cross-validation scores) taking much
        longer. As the two models otherwise perform approximately equally, the speed of the Ordinal model tuning indicates Ordinal encoding is superior in this context.""")
    print("")

    proceed = input("Please type (y) and then press Enter if you wish to proceed with training models (could take a while):")
    if proceed=="y":
        print("Starting tuning...")
    else:
        print("Exiting...")
        print("*********************************************************************************************************************************************************************")
        raise SystemExit(0)

    print("=====================================================================================================================================================================")

    print("XGB - Trieste + Gaussian Process (GPFlow) Bayesian Optimisation, Same Hyperparameters as Grid/Random Search")
    print("")

    def objective_trieste(space):
        cv_scores = []
        for i in range(0,space.numpy().shape[0]):
            # Hyper-parameters for Trieste Bayesian Optimisation (with GP).
            # Use same hyperparameters and ranges as for grid/random search for fairness of comparison.
            xgb_onehot_trieste = xgb.XGBRegressor(
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
            xgb_onehot_trieste.fit(trainX_onehot, trainy)
            cv_k_scores = cross_val_score(xgb_onehot_trieste, trainX_onehot, trainy, scoring="neg_root_mean_squared_error", cv=10)
            cv_scores.append( [-sum(cv_k_scores)/len(cv_k_scores)] )
        return tf.convert_to_tensor(cv_scores, dtype=tf.float64, dtype_hint=None, name=None)
    

    # Time duration of Trieste.  Give Trieste 100 evaluations for comparison with (5) (Ordinal encoding).
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
    # Time elapsed in training: 1812.665458

    # Create XGB using Trieste's best found hyperparameters.
    xgb_onehot_trieste_best = xgb.XGBRegressor(**params_trieste_best, objective="reg:squarederror", verbosity=0, nthread=-1, seed=1)
    xgb_onehot_trieste_best.fit(trainX_onehot, trainy)
    trainPreds = xgb_onehot_trieste_best.predict(trainX_onehot)
    testPreds = xgb_onehot_trieste_best.predict(testX_onehot)

    print("Best Trieste hyperparameters: ")
    print(params_trieste_best)
    # Best Trieste hyperparameters:
    # {'gamma': 0.5, 'reg_alpha': 5.0, 'subsample': 1.0, 'colsample_bytree': 0.5325027559054646, 'learning_rate': 0.16038858405659404,
    # 'max_depth': 6, 'min_child_weight': 9, 'n_estimators': 597}

    print("Train RMSE: %f" % (np.sqrt(mean_squared_error(trainy, trainPreds))))
    print("Test RMSE: %f" % (np.sqrt(mean_squared_error(testy, testPreds))))
    print("Train Accuracy: %f" % (r2_score(trainy, trainPreds)))
    print("Test Accuracy: %f" % (r2_score(testy, testPreds)))
    # Train RMSE: 10.469041
    # Test RMSE: 11.666168
    # Train Accuracy: 0.963519
    # Test Accuracy: 0.955957

    xgb_onehot_trieste_best.save_model("./results/models/trained/XGB_onehot_trieste.json")
    print("*********************************************************************************************************************************************************************")    

main()