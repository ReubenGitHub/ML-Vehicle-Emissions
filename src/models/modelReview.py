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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

r"""
Compares XGB Ordinal encoding (Trieste-tuned) model and equivalent NoMan model (excludes "manufacturer" from features) and saves visualisations in "results\models\tuned_vis".
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

def save_line_graph(traindf, testdf, var, rmse, modelName):
    r"""
    Saves histograms in 'results\models\train_test_vis'.
    """

    graphName = var + " Prediction Error Distribution for " + modelName
    
    plt.plot( np.arange(0,traindf.shape[0])*100/traindf.shape[0], traindf, color="#03bffe", label="train error")
    plt.plot( np.arange(0,testdf.shape[0])*100/testdf.shape[0], testdf, color="#ff845b", label="test error")
    plt.plot( np.arange(0,testdf.shape[0])*100/testdf.shape[0], [rmse]*testdf.shape[0], "--", color="#ff845b", label="test RMSE")
    plt.xlabel("percentile of train/test data set")
    plt.ylabel(var + " error = abs(true - prediction)")
    plt.title(graphName)
    plt.legend(loc="best")

    fileName = graphName + ".png"
    plt.savefig('./results/models/tuned_vis/' + fileName, bbox_inches='tight')
    plt.clf()

def main():
    r"""
    Compares XGB Ordinal encoding (Trieste-tuned) model and equivalent NoMan model (excludes "manufacturer" from features) and saves visualisations in "results\models\tuned_vis".
    """
    
    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")

    # Read the "uk_gov_data_dense_preproc" from the "vehicles" database using pandas.
    govData = pd.read_sql("SELECT * FROM uk_gov_data_dense_preproc", connection)

    # Creating arrays for predictions/analysis.
    ctsFeatures = ["engine_size_cm3", "power_ps"]
    cateFeatures = ["manufacturer", "transmission", "transmission_type", "fuel", "powertrain"]
    cateFeaturesNoMan = ["transmission", "transmission_type", "fuel", "powertrain"]
    features = ctsFeatures + cateFeatures
    featuresNoMan = ctsFeatures + cateFeaturesNoMan
    target = "co2_emissions_gPERkm"
    
    # Define arrays: features (X_ordinal for ordinal encoding of categorical features), and target (y).
    X_ordinal = govData[features].copy()
    X_ordinal_noman = govData[featuresNoMan].copy()
    y = govData[target]

    # Ordinal encoding of categorical features for Full model.
    cateDict = {}
    for category in cateFeatures:
        value = 0
        cateDict[category] = dict()
        for option in OrderedSet(X_ordinal[category]):
            cateDict[category][option] = value
            value += 1
        X_ordinal[category] = X_ordinal[category].map(cateDict[category])
    
    # Ordinal encoding of categorical features for NoMan model.
    cateDictNoMan = {}
    for category in cateFeaturesNoMan:
        value = 0
        cateDictNoMan[category] = dict()
        for option in OrderedSet(X_ordinal_noman[category]):
            cateDictNoMan[category][option] = value
            value += 1
        X_ordinal_noman[category] = X_ordinal_noman[category].map(cateDictNoMan[category])

    # Define training and testing arrays.
    testSize = 0.2
    trainX_ordinal, testX_ordinal, trainy, testy = train_test_split(X_ordinal, y, test_size = testSize, random_state=1)
    trainX, testX, trainy, testy = train_test_split(govData, y, test_size = testSize, random_state=1)
    trainX_ordinal_noman, testX_ordinal_noman, trainy_noman, testy_noman = train_test_split(X_ordinal_noman, y, test_size = testSize, random_state=1)

    # Load in XGB models.
    xgb_ordinal = xgb.XGBRegressor()
    xgb_ordinal.load_model("./results/models/tuned/XGB_ord_trieste_more.json")
    xgb_ordinal_noman = xgb.XGBRegressor()
    xgb_ordinal_noman.load_model("./results/models/tuned/XGB_ord_trieste_more_noman.json")

    # Evaluate the train/test errors and save Prediction Error Distribution graphs.
    trainPreds = xgb_ordinal.predict(trainX_ordinal)
    testPreds = xgb_ordinal.predict(testX_ordinal)
    trainPredsNoMan = xgb_ordinal_noman.predict(trainX_ordinal_noman)
    testPredsNoMan = xgb_ordinal_noman.predict(testX_ordinal_noman)
    trainErrors = abs(trainy - trainPreds).sort_values(0, ascending=True)
    testErrors = abs(testy - testPreds).sort_values(0, ascending=True)
    trainErrorsNoMan = abs(trainy_noman - trainPredsNoMan).sort_values(0, ascending=True)
    testErrorsNoMan = abs(testy_noman - testPredsNoMan).sort_values(0, ascending=True)
    save_line_graph(trainErrors.values, testErrors.values, "co2_emissions_gPERkm", np.sqrt(mean_squared_error(testy, testPreds)), "Full Model" )
    save_line_graph(trainErrorsNoMan.values, testErrorsNoMan.values, "co2_emissions_gPERkm", np.sqrt(mean_squared_error(testy_noman, testPredsNoMan)), "NoMan Model" )
    
    # List test data by descending prediction error.
    test_worst_indices = testErrors.sort_values(0, ascending=False).index[:]
    testErrors = testErrors.rename("error")
    testPreds = pd.DataFrame(testPreds, columns=["prediction"]).set_index(testX.index)
    testX = pd.concat([testX, testPreds, testErrors], axis=1)
    worst_examples = testX.filter(items=test_worst_indices, axis="index" )

    # Save feature importance charts.
    xgb.plot_importance(xgb_ordinal,color=(3/255, 191/255, 254/255), title="Feature Importance (Gain) for Full Model", importance_type="gain", grid=False, show_values=False)
    plt.savefig('./results/models/tuned_vis/Feature Importance Full Model', bbox_inches='tight')
    plt.clf()
    xgb.plot_importance(xgb_ordinal_noman,color=(3/255, 191/255, 254/255), title="Feature Importance (Gain) for NoMan Model", importance_type="gain", grid=False, show_values=False)
    plt.savefig('./results/models/tuned_vis/Feature Importance NoMan Model', bbox_inches='tight')
    plt.clf()

    print("")
    print("*********************************************************************************************************************************************************************")
    print("XGB MODELS REPORT")
    print("=====================================================================================================================================================================")
    print(r"""Visualisations of the models are saved in the "results\models\tuned_vis" folder.""")
    print("")
    print("""I have created several models predicting CO2 emissions of vehicles using a variety of methods and tuning techniques, and I will now review the two models which
        appear to be the best candidates: the two XGB models with Ordinal encoding and with hyperparameters tuned using Trieste (300 evaludations).""")
    print("""One model utilises the features "manufacturer", "transmission", "transmission_type", "engine_size_cm3", "fuel", "powertrain", and "power_ps" (the "Full" model).""")
    print("""The other utilises the same features except excludes "manufacturer" (the "NoMan" model).""")
    print("")
    print("""The observed accuracies (R^2, Coefficient of Determination) of these models are:
        Model           Train Accuracy     Test Accuracy     Test RMSE     Tuning Duration (s)
        --------------------------------------------------------------------------------------
        Full model      0.963918           0.956700          11.567392     3120
        NoMan model     0.961333           0.954035          11.918023     3222""")
    print("")
    print("""As discussed in trainTunedOrdinalNoMan.py, the models perform approximately equally. As the NoMan model requires less data per example (no information about the manufactuer
        is required) than the Full model, one may consider using the NoMan model over the Full model to obtain similar performance.""")
    print("")
    print("""The Prediction Error Distribution graphs visualise the errors of the models across each percentile of the train and test data sets.
        The graphs shows that a small proportion (~20%) of the train/test sets have prediction errors larger than the test RMSEs (11.6 g/km for Full, 11.9 g/km for NoMan).
        The graphs also shows that the prediction error grows rapidly at the higher percentiles of the data sets, due to significant prediction errors on a small proportion of
        the data.""")
    print("")
    print("The vehicles in the test set with the 10 largest errors as predicted by the Full model are:""")
    print(worst_examples.head(10))
    #       car_id   manufacturer                                model                      description transmission  ...                           powertrain  power_ps co2_emissions_gPERkm  prediction      error
    # 1332    1333           FORD  Tourneo Custom Model Year Post 2021                      2.0 EcoBlue           M6  ...     Internal Combustion Engine (ICE)       105                153.0  240.061478  87.061478
    # 1344    1345           FORD  Tourneo Custom Model Year Post 2021               2.0 EcoBlue (mHEV)           M6  ...  Mild Hybrid Electric Vehicle (MHEV)       130                145.0  218.706024  73.706024
    # 1347    1348           FORD  Tourneo Custom Model Year Post 2021  2.0 EcoBlue - 4.19 FDR - (mHEV)           M6  ...  Mild Hybrid Electric Vehicle (MHEV)       185                265.0  200.960464  64.039536
    # 4224    4225  MERCEDES-BENZ                     G-Class MY 201.5                         AMG G 63           A9  ...     Internal Combustion Engine (ICE)       585                377.0  313.194244  63.805756
    # 3963    3964  MERCEDES-BENZ         GLE Estate Model Year 2021.5                 GLE 300 d 4MATIC           A9  ...     Internal Combustion Engine (ICE)       245                218.0  164.095993  53.904007
    # 2652    2653     LAND ROVER                          Velar, 20MY     2.0 i4 Petrol 300PS AWD Auto       A8-AWD  ...     Internal Combustion Engine (ICE)       300                215.0  263.069885  48.069885
    # 4273    4274  MERCEDES-BENZ              GLC SUV Model Year 2022             AMG GLC 63 S 4MATIC+           A9  ...     Internal Combustion Engine (ICE)       510                300.0  253.034454  46.965546
    # 3507    3508  MERCEDES-BENZ                GLC Model Year 2020.5             AMG GLC 63 S 4MATIC+           A9  ...     Internal Combustion Engine (ICE)       510                298.0  253.034454  44.965546
    # 4307    4308  MERCEDES-BENZ                                  GLS                 GLS 400 d 4MATIC           A9  ...     Internal Combustion Engine (ICE)       330                251.0  206.231155  44.768845
    # 1135    1136           FORD        Focus Model Year Post 2021.75                      2.0 EcoBlue           A8  ...     Internal Combustion Engine (ICE)       150                124.0  165.160431  41.160431
    print("")
    print("""Validating the data for these examples by searching online indicates there may be some leftover errors in the data:
        From the documentation, it is not clear that the emissions values for the Ford Tourneo vehicles are correct; they should possibly all be around 200 g/km, closer to the model prediction:
            https://www.ford.co.uk/content/dam/guxeu/uk/documents/feature-pdfs/FT-New_Tourneo_Custom.pdf
        The Mercedes GLE Estate vehicle emissions should possibly be lower, between 172-192 g/km, closer to the model prediction:
            https://www.mercedes-benz.com/en/vehicles/wltp/wltp-fuel-consumption-and-emission-values/""")
    print("The other vehicles appear to be outliers or exceptions to the trends the model has identified. Less accurate predictions for vehicles that don't fit the general distribution are not unexpected.")
    print("")
    print("The Feature Importance charts visualise the value of each feature to each model, in terms of the improvements in accuracy attained to the branches it is on.")
    print("""Both charts show that "fuel" and "powertrain" are valuable features. This is not unexpected given the clustering seen in the "VS engine_size_cm3" and "VS power_ps" graphs,
        and the distinct distributions seen in the "VS fuel" and "VS powertrain" graphs.
        It appears the importances of "fuel" and "powertrain" may be inversely related to each other. This may be because they contain similar information, e.g.
        "fuel" = "electricity" <=> "powertrain" = "...(BEV)", so where a model relies heavily on one of these features, it may not need to use the other.""")
    print("""The NoMan model values "transmission_type" very highly. While different transmission types do have distinct distributions ("VS transmission_type" graphs), I may
        have initially expected the models to gain more from "powertrain" as in the Full model, as this contains similar information but at a more granular level. However, the
        NoMan model may be extracting relevant trends out of "transmission_type" and "fuel", as indicated by the high value of "fuel", and "powertrain" may no longer be so useful.""")
    print(""""engine_size_cm3" has moderate value to each model, as does "power_ps" in the Full model. This is not unexpected considering the correlations investigated
        in dataPreprocessAnalyse.py""")
    print("""The Full model identifies "manufacturer" as being of low importance, providing further evidence that we can expect the NoMan model to still perform well even though
        it does not use "manufacturer".""")
    print("")
    print(r"""One possible use for models such as the ones created here could be data (entry) validation. Once a model has been trained on data in a data set, both existing data
        and future insertions could be checked against the model's prediction: a flag could be raised if, for a given record, the recorded emissions are beyond some tolerance of
        the model's predicted emissions based on the other field values. The database manager could then double-check the inserted example for errors, such as typos or inaccurate
        data, and either amend the data or confirm its validity.

        This application would see the model acting as a passive error identifier, and may help to catch and prevent vitally incorrect data entering a system.
        For example, in this government vehicles data set, these models combined with automated validation might raise flags for the examples above. As it appears that some of
        these examples may have inaccurate data, this may help in spotting and correcting the errors.""")
    print("")
    print("""Next steps or improvements upon this work may include:
        Further development of models which require even fewer features than the NoMan model. The feature importance charts indicate that there may be some interchangeability
        between the importance of various features, such as "powertrain", "fuel", and "transmission_type". Therefore, investigating the data with a Factor Analysis or Multiple
        Correspondence Analysis (MCA) may reveal hidden stuctures in the data and help reduce dimensionality. A model might then be built requiring fewer features.
        Further cleansing and validating of the data prior to training the model may produce models with higher accuracy. Now that models have been produced, it may be possible
        to identify and correct remaining errors in the data following a process similar to that described above. A new model could then be trained, possibly attaining greater
        accuracy.""")
    print("*********************************************************************************************************************************************************************")    

main()