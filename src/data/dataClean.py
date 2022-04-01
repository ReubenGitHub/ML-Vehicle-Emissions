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

import mysql.connector
from mysql.connector import Error
from dbLogin import configprivate
import pandas as pd
import numpy as np

"""
Manipulates the "uk_gov_data" table to produce both sparse ("uk_gov_data_sparse") and dense ("uk_gov_data_dense") tables to address the points identified in dataInitialiseAnalyse.py.
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

def create_table(connection, query):
    """
    Creates a table in the "vehicles" database in the local MySQL server.
    """

    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Table created successfully")
    except Error as err:
        print(f"Error: '{err}'")

def insert_table(connection, query, df):
    """
    Performs queries, e.g. INSERT, in the "vehicles" database.
    """
    
    cursor = connection.cursor()
    try:
        for i in range(0, df.shape[0]):
            cursor.execute(query, tuple(df.iloc[i].values.flatten().tolist()))
        connection.commit()
        print("Table edited successfully")
    except Error as err:
        print(f"Error: '{err}'")

def main():
    """
    Manipulates the "uk_gov_data" table to produce both sparse ("uk_gov_data_sparse") and dense ("uk_gov_data_dense") tables to address the points identified in dataInitialiseAnalyse.py.
    """
    
    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")

    # Read the UK gov data from the "vehicles" database using pandas. Convert "car_id" from int64 (a numpy type) to float as MySQL cannot convert:
        # https://stackoverflow.com/questions/56731036/interfaceerror-failed-executing-the-operation-python-type-numpy-int64-cannot-b
    govData = pd.read_sql("SELECT * FROM uk_gov_data", connection)
    govData = govData.astype(dtype = {"car_id": float}, copy=True)

    # Create the table "uk_gov_data_sparse".
    create_govtablesparse_query = """
        USE vehicles;
        CREATE TABLE uk_gov_data_sparse LIKE uk_gov_data;
    """
    create_table(connection, create_govtablesparse_query)

    # (4) Replace "Electric - Not Applicable" in "transmission_type" with "Automatic" when "fuel" = "Petrol".
    govData.loc[(govData["fuel"] == "Petrol")&(govData["transmission_type"] == "Electric - Not Applicable"),"transmission_type"] = "Automatic"
    # (7) Replace "powertrain" and "co2_emission_gPERkm" when "model" = "Evoque, 20MY" and "powertrain" = "Hybrid Electric Vehicle (HEV)".
    indices = govData[ (govData["powertrain"]=="Hybrid Electric Vehicle (HEV)") & (govData["model"]=="Evoque, 20MY")  ].index
    govData.loc[indices,"powertrain"] = "Plug-in Hybrid Electric Vehicle (PHEV)"
    govData.loc[indices[0],"co2_emissions_gPERkm"] = 32
    govData.loc[indices[1],"co2_emissions_gPERkm"] = 38
    # (7) Replace "co2_emissions_gPERkm" with "22" when "description" = "RAV4 Design 2.5 Plug-in Hybrid".
    govData.loc[govData["description"] == "RAV4 Design 2.5 Plug-in Hybrid","co2_emissions_gPERkm"] = 22

    # Populate the (relatively speaking) sparse table "uk_gov_data_sparse".
    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")
    govDataSparseImport = govData.replace({np.nan: None}, inplace=False)
    query = """INSERT INTO uk_gov_data_sparse VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    insert_table(connection, query, govDataSparseImport)
    # Save this cleaned sparse data as a csv to "data\intermediate".
    govDataSparseImport.to_csv('./data/intermediate/uk_gov_data_sparse.csv', index=False, encoding="ISO-8859-1")

    # (1) Now to create the dense data set, replace nulls in "transmission" with "Auto".
    govData["transmission"].replace({np.nan: "Auto"}, inplace=True)
    # (2) Replace nulls in "engine_size_cm3" with 0.
    govData["engine_size_cm3"].replace({np.nan: 0}, inplace=True)
    # (3) Replace nulls in "power_ps" with 0.
    govData["power_ps"].replace({np.nan: 0}, inplace=True)

    # Create the table "uk_gov_data_dense".
    create_govtabledense_query = """
        USE vehicles;
        CREATE TABLE uk_gov_data_dense LIKE uk_gov_data;
    """
    create_table(connection, create_govtabledense_query)

    # Populate the dense table "uk_gov_data_dense".
    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")
    govDataDenseImport = govData
    query = """INSERT INTO uk_gov_data_dense VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    insert_table(connection, query, govDataDenseImport)
    # Save this cleaned dense data as a csv to "data\intermediate".
    govDataDenseImport.to_csv('./data/intermediate/uk_gov_data_dense.csv', index=False, encoding="ISO-8859-1")

main()