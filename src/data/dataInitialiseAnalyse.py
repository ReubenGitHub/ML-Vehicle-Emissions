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
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

"""
Checks for missing values and errors in the "uk_gov_data" table in the "vehicles" database.
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
    Checks for missing values and errors in the "uk_gov_data" table in the "vehicles" database.
    """
    
    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")

    # Read the UK gov data from the "vehicles" database using pandas.
    govData = pd.read_sql("SELECT * FROM uk_gov_data", connection)

    # Save heatmap of the missing values in the table.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lightcmap = cm.colors.ListedColormap( [(3/255, 191/255, 254/255), (255/255, 132/255, 91/255)] )
    ax = sns.heatmap(govData.isnull(),cbar=False,cmap=lightcmap)
    plt.savefig('./data/raw_vis/heatmap', bbox_inches='tight')

    print("")
    print("*********************************************************************************************************************************************************************")
    print("INITIAL DATA CHECK REPORT")
    print("=====================================================================================================================================================================")
    print("The dataframe info shows the number of rows and datatypes are as expected, and reveals that there are null values in some fields.")
    print("")
    print(govData.info())
        # RangeIndex: 6756 entries, 0 to 6755
        # Data columns (total 11 columns):
        # #   Column                Non-Null Count  Dtype
        # ---  ------                --------------  -----
        # 0   car_id                6756 non-null   int64
        # 1   manufacturer          6756 non-null   object
        # 2   model                 6756 non-null   object
        # 3   description           6756 non-null   object
        # 4   transmission          6609 non-null   object
        # 5   transmission_type     6756 non-null   object
        # 6   engine_size_cm3       6755 non-null   float64
        # 7   fuel                  6756 non-null   object
        # 8   powertrain            6756 non-null   object
        # 9   power_ps              6712 non-null   float64
        # 10  co2_emissions_gPERkm  6756 non-null   float64
    print("=====================================================================================================================================================================")

    print("""Clearer null values by field. Most are in "transmission", and some are in "power_ps" and "engine_size_cm3".""")
    print("")
    print(govData.isnull().sum())
        # car_id                    0
        # manufacturer              0
        # model                     0
        # description               0
        # transmission            147
        # transmission_type         0
        # engine_size_cm3           1
        # fuel                      0
        # powertrain                0
        # power_ps                 44
        # co2_emissions_gPERkm      0
        # dtype: int64
    print("=====================================================================================================================================================================")

    print("Reviewing unique values present in each field:")
    print(r"""Points of Interest (POI) are nulls in (1) "transmission", (2) "engine_size_cm3", and (3) "power_ps". Nulls are visualised in a heatmap in "data\raw_vis".""")
    print("""Also, (4) "transmission_type" has "Electric - Not Applicable" values.""")
    print("""Further, (5) "engine_size_cm3", (6) "power_ps", and (7) "co2_emissions_gPERkm" have "0" values.""")
    print("")
    for col in govData.columns:
        print('{} : {}'.format(col,govData[col].unique()))
    print("=====================================================================================================================================================================")

    print("(1)")
    print("""Print rows where "transmission" = null. Seems to occur only if "fuel" = Electricity.""")
    print("")
    print(govData[govData["transmission"].isnull()])
    print("")
    print("""Print unique values of "fuel" when "transmission" = null. This reveals that "transmission" = null => "fuel" = "Electricity".""")
    print("")
    print(govData[govData["transmission"].isnull()]["fuel"].value_counts(dropna = False))
        # Electricity    147
    print("")
    print("""Print unique values of "co2_emissions_gPERkm" when "fuel" = "Electricity". This reveals that "fuel" = "Electricity" => "co2_emissions_gPERkm" = "0.0".""")
    print("")
    print(govData[govData["fuel"]=="Electricity"]["co2_emissions_gPERkm"].value_counts(dropna = False))
        # 0.0    180
    print("")
    print("Deleting these rows could lead to an under-representation of electric vehicles in the data, so it would be good to replace these nulls if possible.")
    print("""A key characteristic of these rows is that all the vehicles are electric, and this implies "co2_emissions_gPERkm" = "0.0".""")
    print("""Therefore, a replacement "transmission" value should ideally imply "fuel" = "Electricity".""")
    print("""Print unique values of "fuel" when "transmission" = "Auto". This reveals that "transmission" = "Auto" => "fuel" = "Electricity".""")
    print("")
    print(govData[govData["transmission"]=="Auto"]["fuel"].value_counts(dropna = False))
        # Electricity    6
    print("")
    print("""Therefore, I will replace all "transmission" nulls with "Auto" in generating a dense data set in dataClean.py, resolving POI (1).""")
    print("=====================================================================================================================================================================")

    print("(2)")
    print("""Print "fuel" where "engine_size_cm3" = null. Occurs only if "fuel" = "Electricity".""")
    print("")
    print(govData[govData["engine_size_cm3"].isnull()])
        #       car_id manufacturer    model description transmission  ... engine_size_cm3         fuel                                         powertrain power_ps  co2_emissions_gPERkm
        # 1703    1704      HYUNDAI  IONIQ 5  SE Connect         None  ...             NaN  Electricity  Battery Electric Vehicle (BEV) / Pure Electric...    170.0                   0.0
    print("")
    print("""Print unique values of "engine_size_cm3" when "fuel" = "Electricity". This reveals that "fuel" = "Electricity" => "engine_size_cm3" = "0.0".""")
    print("")
    print(govData[govData["fuel"]=="Electricity"]["engine_size_cm3"].value_counts())
        # 0.0    179
    print("")
    print("""Therefore, I will replace the "engine_size_cm3" null with "0.0" in generating a dense data set in dataClean.py, resolving POI (2).""")
    print("=====================================================================================================================================================================")

    print("(3)")
    print("""Print rows where "power_ps" = null. Occurs only if "fuel" contains "Electricity", and only if "co2_emissions_gPERkm" = "0.0".""")
    print("")
    print(govData[govData["power_ps"].isnull()])
    print("")
    print("""Print unique values of "co2_emissions_gPERkm" when "power_ps" = "0.0". Reveals "power_ps" = "0.0" => "co2_emissions_gPERkm" = "0.0".""")
    print("")
    print(govData[govData["power_ps"]==0]["co2_emissions_gPERkm"].value_counts())
        # 0.0    82
    print("")
    print("""Since the emissions are 0.0 for all the "power_ps" nulls, I will replace these nulls with "0.0" in generating a dense data set in dataClean.py, resolving POI (3).""")
    print("=====================================================================================================================================================================")

    print("(4)")
    print("""Print unique values of "fuel" when "transmission_type" = "Electric - Not Applicable".""")
    print("")
    print(govData[govData["transmission_type"]=="Electric - Not Applicable"]["fuel"].value_counts())
        # Electricity    102
        # Petrol           2
    print("")
    print("""The presence of petrol-fuelled cars where "transmission_type" = "Electric - Not Applicable" is unexpected. There are only 2, so should be quick to validate.""")
    print("Print manufacturer, model, and description for the petrol vehicles.")
    print("")
    print(govData[ (govData["transmission_type"]=="Electric - Not Applicable") & (govData["fuel"]=="Petrol")][ ["manufacturer", "model", "description"]])
        # 287  CHRYSLER JEEP  Renegade MY21  NIGHT EAGLE 1.0 T3 120hp MT FWD
        # 828        FERRARI             F8                          Tributo
    print("")
    print("""Actual transmission type for both is "Automatic":
        https://www.cars-data.com/en/jeep-renegade-1-3t-4wd-limited-specs/81427/tech
        https://www.cars-data.com/en/ferrari-f8-tributo-specs/104709/tech""")
    print("""Therefore, I will only replace the two instances of "Electric - Not Applicable" where "fuel" = "Petrol" in dataClean.py, resolving POI (4).""")
    print("=====================================================================================================================================================================")

    print("(5)")
    print("""Print unique values of "fuel" when "engine_size_cm3" = "0.0". Reveals "engine_size_cm3" = "0.0" => "fuel" = "Electricity".""")
    print("")
    print(govData[govData["engine_size_cm3"]==0]["fuel"].value_counts())
        # Electricity    179
    print("")
    print("""Print unique values of "engine_size_cm3" when "fuel" = "Electricity". Reveals "fuel" = "Electricity" => "engine_size_cm3" = "0.0".""")
    print("")
    print(govData[govData["fuel"]=="Electricity"]["engine_size_cm3"].value_counts())
        # 0.0    179
    print("")
    print("""Therefore, "fuel" = "Electricity" if and only if "engine_size_cm3" = "0.0", which is to be expected as electric engines do not function with air displacement, resolving POI (5).""")
    print("=====================================================================================================================================================================")

    print("(6)")
    print("""Print unique values of "fuel" when "power_ps" = "0.0". Reveals "power_ps" = "0.0" => "fuel" = "Electricity".""")
    print("")
    print(govData[govData["power_ps"]==0]["fuel"].value_counts())
        # Electricity    82
    print("")
    print("""This could be because (horse)power in BEVs is not directly comparable to power in other (combustion) vehicles, so has been left as 0.0:
        https://auto.howstuffworks.com/how-does-horsepower-figure-into-electric-cars.htm""")
    print("""Furthermore, we see in (7) that "fuel" = "Electricity" => "co2_emissions_gPERkm" = "0.0".""")
    print("""For unseen samples we would expect that if "power_ps" = "0.0", then the vehicle would be electric and hence emissions would be 0.""")
    print("""Therefore, these values are consistent with what we expect to see in the data, resolving POI (6).""")
    print("=====================================================================================================================================================================")

    print("(7)")
    print("""Print unique values of "powertrain" when "co2_emissions_gPERkm" = 0.0.""")
    print("")
    print(govData[govData["co2_emissions_gPERkm"]==0]["powertrain"].value_counts(dropna = False))
        # Battery Electric Vehicle (BEV) / Pure Electric Vehicle / Electric Vehicle (EV)    180
        # Hybrid Electric Vehicle (HEV)                                                       2
        # Plug-in Hybrid Electric Vehicle (PHEV)                                              1
    print("")
    print("""The CO2 emissions for pure electric vehicles should be nil, so we would expect the 180 "co2_emissions_gPERkm" = "0.0"'s when "powertrain" = "...(EV)".""")
    print("""Printing the unique values of "co2_emissions_gPERkm" when "powertrain" = "...(EV)" reveals "powertrain" = "...(EV)" => "co2_emissions_gPERkm" = "0.0".""")
    print("")
    print(govData[govData["powertrain"]=="Battery Electric Vehicle (BEV) / Pure Electric Vehicle / Electric Vehicle (EV)"]["co2_emissions_gPERkm"].value_counts(dropna = False))
    print("")
    print("""The 3 other "co2_emissions_gPERkm" = "0.0"'s are for hybrid electric cars and are not expected.""")
    print("""The emission values on the live database imply the HEV vehicles (both Land Rover Evoques) are actually PHEVs:""")
    print("""The emission values on the live database imply the PHEV vehicle (Toyota RAV4) actually has emissions of 22g/km:""")
    print("""   Live database: https://carfueldata.vehicle-certification-agency.gov.uk/search-by-low-emissions.aspx""")
    print("")
    print("""Therefore, I will change the "powertrain" of the 2 HEV's to "...(PHEV)" and their "co2_emissions_gPERkm" to "32" and "38",
    and I will change the "co2_emissions_gPERkm" of the PHEV to "22",  resolving POI (7).""")
    print("*********************************************************************************************************************************************************************")

main()