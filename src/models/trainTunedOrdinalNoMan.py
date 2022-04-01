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
Trains XGBoost model with hyperparameter tuning using Trieste, excluding "manufacturer". Saves the model in "results\models\tuned".
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
    Trains XGBoost model with hyperparameter tuning using Trieste, excluding "manufacturer". Saves the model in "results\models\tuned".
    """
    
    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")

    # Read the "uk_gov_data_dense_preproc" from the "vehicles" database using pandas.
    govData = pd.read_sql("SELECT * FROM uk_gov_data_dense_preproc", connection)

    # Creating array for training use.
    # Manufacturer is excluded from the features to create a model that can predict emissions even for unseen manufacturers.
    ctsFeatures = ["engine_size_cm3", "power_ps"]
    cateFeatures = ["transmission", "transmission_type", "fuel", "powertrain"]
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
    print("XGB ORDINAL ENCODING, NO MANUFACTURER, HYPERPARAMETER TUNING REPORT")
    print("=====================================================================================================================================================================")

    print("Hyperparameter Tuning Approach")
    print("")
    print("""In tuning hyperparameters for this XGB model not selecting "manufacturer" as a feature, I use Trieste Bayesian Optimisation with GPFlow and Expected Improvement (EI).""")
    print("I will then compare the model accuracy produce with this method with (6) in trainTunedOrdinal.py.")
    print("The procurement of train/test data and the use of cross-validation is the same as in trainTunedOrdinal.py.")
    print("""I allow 300 evaluations and use the same hyperparameter ranges as (6) (so the only difference is the selection of "manufacturer") for fair comparison.""")
    print("")
    print("""The observed accuracies (R^2, Coefficient of Determination) of the model here and of (6) in trainTunedOrdinal.py are:
        Model                          Train Accuracy     Test Accuracy     Tuning Duration (s)
        ---------------------------------------------------------------------------------------
        Trieste - No Manufacturer      0.961333           0.954035          3222
        (6) Trieste - Manufacturer     0.963918           0.956700          3120""")

    print("")
    print("""The models perform approximately equally. The model utilising "manufacturer" has a slightly higher accuracy (95.67%) than the model excluding this feature
        ("NoMan" model) (95.40%).""")
    print("")
    print("""Therefore, the exclusion of "manufacturer" as a feature in this context has little impact on model accuracy, and it would be valid to consider using the NoMan model over
        the model with the greater number of features, as the NoMan model has the benefit of requiring less data per example.""")
    print("")
    print("""The speed of tuning the hyperparameters is equivalent between the two models (3222s vs 3120s), which is not too unexpected.""")
    print("")
    print("I review the NoMan model in modelReview.py.")

    print("")

    proceed = input("Please type (y) and then press Enter if you wish to proceed with training models (could take a while):")
    if proceed=="y":
        print("Starting tuning...")
    else:
        print("Exiting...")
        print("*********************************************************************************************************************************************************************")
        raise SystemExit(0)

    print("=====================================================================================================================================================================")

    print("XGB - Trieste + Gaussian Process (GPFlow) Bayesian Optimisation, More Hyperparameters")
    print("")

    def objective_trieste_more(space):
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
    # Time elapsed in training: 3222.682545

    # Create XGB using Trieste's best found hyperparameters.
    xgb_ordinal_trieste_more_best = xgb.XGBRegressor(**params_trieste_more_best, objective="reg:squarederror", verbosity=0, nthread=-1, seed=1)
    xgb_ordinal_trieste_more_best.fit(trainX_ordinal, trainy)
    trainPreds = xgb_ordinal_trieste_more_best.predict(trainX_ordinal)
    testPreds = xgb_ordinal_trieste_more_best.predict(testX_ordinal)

    print("Best Trieste hyperparameters: ")
    print(params_trieste_more_best)
    # Best Trieste hyperparameters:
    # {'gamma': 6.54233039316413, 'reg_alpha': 0.09785897598015403, 'reg_lambda': 17.650008930890767, 'subsample': 0.9859317355445424,
    # 'colsample_bytree': 0.38894794572124625, 'colsample_bylevel': 0.9854400126973125, 'colsample_bynode': 0.6346448251882812,
    # 'learning_rate': 0.2474972354880873, 'max_depth': 17, 'min_child_weight': 8, 'n_estimators': 991}

    print("Train RMSE: %f" % (np.sqrt(mean_squared_error(trainy, trainPreds))))
    print("Test RMSE: %f" % (np.sqrt(mean_squared_error(testy, testPreds))))
    print("Train Accuracy: %f" % (r2_score(trainy, trainPreds)))
    print("Test Accuracy: %f" % (r2_score(testy, testPreds)))
    # Train RMSE: 10.778138
    # Test RMSE: 11.918023
    # Train Accuracy: 0.961333
    # Test Accuracy: 0.954035

    xgb_ordinal_trieste_more_best.save_model("./results/models/tuned/XGB_ord_trieste_more_noman.json")
    print("*********************************************************************************************************************************************************************")    

main()