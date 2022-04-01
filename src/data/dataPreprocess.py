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
Manipulates the "uk_gov_data_sparse" and "uk_gov_data_dense" table to produce "...preproc" tables, while addressing the outliers identified in dataCleanAnalyse.py.
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
    Manipulates the "uk_gov_data_sparse" and "uk_gov_data_dense" table to produce "...preproc" tables, while addressing the outliers identified in dataCleanAnalyse.py.
    """
    
    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")

    # Read the UK gov data from the "vehicles" database using pandas. Convert "car_id" from int64 (a numpy type) to float as MySQL cannot convert:
        # https://stackoverflow.com/questions/56731036/interfaceerror-failed-executing-the-operation-python-type-numpy-int64-cannot-b
    govDataSparse = pd.read_sql("SELECT * FROM uk_gov_data_sparse", connection)
    govDataSparse = govDataSparse.astype(dtype = {"car_id": float}, copy=True)

    # Create the table "uk_gov_data_sparse_preproc".
    create_govtablesparse_query = """
        USE vehicles;
        CREATE TABLE uk_gov_data_sparse_preproc LIKE uk_gov_data_sparse;
    """
    create_table(connection, create_govtablesparse_query)

    # (1) Replace "model" = "G-Class MY 201.5" (ICE outlier) emissions with "377.0".
    govDataSparse.loc[(govDataSparse["powertrain"]=="Internal Combustion Engine (ICE)")&(govDataSparse["co2_emissions_gPERkm"]<90), "co2_emissions_gPERkm"] = 377
    # (3) Replace "model" = "Galaxy Model Year Post 2021" (HEV outlier) emissions with "148.0".
    govDataSparse.loc[(govDataSparse["model"]=="Galaxy Model Year Post 2021")&(govDataSparse["description"]=="2.5 Duratec (FHEV)")&(govDataSparse["co2_emissions_gPERkm"]>300), "co2_emissions_gPERkm"] = 148

    # Populate the sparse preprocessed table "uk_gov_data_sparse_preproc".
    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")
    govDataSparseImport = govDataSparse.replace({np.nan: None}, inplace=False)
    query = """INSERT INTO uk_gov_data_sparse_preproc VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    insert_table(connection, query, govDataSparseImport)
    # Save this preprocessed sparse data as a csv to "data\processed".
    govDataSparseImport.to_csv('./data/processed/uk_gov_data_sparse_preproc.csv', index=False, encoding="ISO-8859-1")

    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")
    # Read the UK gov data from the "vehicles" database using pandas. Convert "car_id" from int64 (a numpy type) to float as MySQL cannot convert:
        # https://stackoverflow.com/questions/56731036/interfaceerror-failed-executing-the-operation-python-type-numpy-int64-cannot-b
    govDataDense = pd.read_sql("SELECT * FROM uk_gov_data_dense", connection)
    govDataDense = govDataDense.astype(dtype = {"car_id": float, "engine_size_cm3": float, "power_ps": float}, copy=True)

    # Create the table "uk_gov_data_dense_preproc".
    create_govtabledense_query = """
        USE vehicles;
        CREATE TABLE uk_gov_data_dense_preproc LIKE uk_gov_data_dense;
    """
    create_table(connection, create_govtabledense_query)

    # (1) Replace "model" = "G-Class MY 201.5" (ICE outlier) emissions with "377.0".
    govDataDense.loc[(govDataDense["powertrain"]=="Internal Combustion Engine (ICE)")&(govDataDense["co2_emissions_gPERkm"]<90), "co2_emissions_gPERkm"] = 377
    # (3) Replace "model" = "Galaxy Model Year Post 2021" (HEV outlier) emissions with "148.0".
    govDataDense.loc[(govDataDense["model"]=="Galaxy Model Year Post 2021")&(govDataDense["description"]=="2.5 Duratec (FHEV)")&(govDataDense["co2_emissions_gPERkm"]>300), "co2_emissions_gPERkm"] = 148

    # Populate the dense table "uk_gov_data_dense_preproc".
    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")
    govDataDenseImport = govDataDense
    query = """INSERT INTO uk_gov_data_dense_preproc VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    insert_table(connection, query, govDataDenseImport)
    # Save this cleaned dense data as a csv to "data\processed".
    govDataDenseImport.to_csv('./data/processed/uk_gov_data_dense_preproc.csv', index=False, encoding="ISO-8859-1")

main()