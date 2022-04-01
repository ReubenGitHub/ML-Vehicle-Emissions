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
Connects to the local MySQL server and creates a "vehicles" database. Then creates table "uk_gov_data" and populates it with the vehicles data in "Euro_6_latest.csv".
"""

def create_server_connection(host_name, user_name, user_password):
    """
    Returns a connection to the local MySQL server.
    """

    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password
        )
        print("MySQL Server connection successful")
    except Error as err:
        print(f"Error: '{err}'")

    return connection

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

def create_database(connection, query):
    """
    Creates a database called "vehicles" in the local MySQL server.
    """

    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Database created successfully")
    except Error as err:
        print(f"Error: '{err}'")

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
            #cursor.execute(query, tuple([i+1] + df.iloc[i].values.flatten().tolist()))
            cursor.execute(query, tuple(df.iloc[i].values.flatten().tolist()))
        connection.commit()
        print("Table edited successfully")
    except Error as err:
        print(f"Error: '{err}'")

def main():
    """
    Connects to the local MySQL server and creates a "vehicles" database. Then creates table "uk_gov_data" and populates it with the vehicles data in "Euro_6_latest.csv".
    """
    
    connection = create_server_connection("localhost", configprivate.username, configprivate.password)

    create_database_query = """
        CREATE DATABASE vehicles
    """
    create_database(connection, create_database_query)

    create_govtable_query = """
        USE vehicles;
        CREATE TABLE uk_gov_data (
            car_id INT PRIMARY KEY,
            manufacturer VARCHAR(20) NOT NULL,
            model VARCHAR(50) NOT NULL,
            description VARCHAR(100) NOT NULL,
            transmission VARCHAR(20),
            transmission_type VARCHAR(30),
            engine_size_cm3 INT,
            fuel VARCHAR(20),
            powertrain VARCHAR(100),
            power_ps INT,
            co2_emissions_gPERkm FLOAT
            );
    """
    create_table(connection, create_govtable_query)

    connection.close
    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")

    # Read the UK gov data from the source csv using pandas. Using the chardet module reveals the encoding is "ISO-8859-1".
    # govData = pd.read_csv('./data/raw/Euro_6_latest.csv', encoding="ISO-8859-1", keep_default_na=False, na_values="")
    govData = pd.read_csv('./data/raw/Euro_6_latest.csv', encoding="ISO-8859-1")
    fields = ["Manufacturer", "Model", "Description", "Transmission", "Manual or Automatic", "Engine Capacity", "Fuel Type", "Powertrain", "Engine Power (PS)", "WLTP CO2", "WLTP CO2 Weighted"]
    govData = govData[fields]

    # Plug-in Hybrid Electric Vehicles' (PHEVs') CO2 emissions are tested differently from other powertrain types under WLTP, so their scores come from the "WLTP CO2 Weighted" column.
        # https://carfueldata.vehicle-certification-agency.gov.uk/search-by-low-emissions.aspx
        # https://heycar.co.uk/guides/what-is-wltp
        # Further info: https://carfueldata.vehicle-certification-agency.gov.uk/additional/2021/2021%20Booklet.pdf
    for i in range(0, govData.shape[0]):
        if govData.iloc[i]["Powertrain"] == "Plug-in Hybrid Electric Vehicle (PHEV)":
            govData.loc[i,"WLTP CO2"] = govData.iloc[i]["WLTP CO2 Weighted"]
    govData.drop(columns="WLTP CO2 Weighted", inplace=True)

    # Replace numpy nan's with None to avoid error "Unknown column 'NaN'": https://github.com/mysqljs/mysql/issues/2403
    govDataImport = govData.replace({np.nan: None}, inplace=False)
    # Create "car_id" column.
    govDataImport.insert(0, "car_id", np.arange(1, govDataImport.shape[0] + 1))
    govDataImport = govDataImport.astype(dtype = {"car_id": float}, copy=True)
    # Insert the data into the SQL database.
    query = """INSERT INTO uk_gov_data VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    insert_table(connection, query, govDataImport)

    # Save this read raw data as a csv to "data\raw".
    govDataImport.to_csv('./data/raw/uk_gov_data.csv', index=False, encoding="ISO-8859-1")

main()