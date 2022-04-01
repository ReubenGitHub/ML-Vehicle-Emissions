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
import pandas as pd
import numpy as np
from ordered_set import OrderedSet
import matplotlib.pyplot as plt
from matplotlib import cm

"""
Create visualisations from the "uk_gov_data_dense_preproc" table in the "vehicles" database. Visualisations are saved in 'data\processed_vis'.
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

def save_cts_graph(df, indep, dep, grouping):
    """
    Saves scatter graphs (for continuous fields) in 'data\processed_vis'.
    """

    graphName = dep + "  VS  " + indep
    
    # Define no_labels number of colours for different labels in the "grouping" field, and plot points in a scatter with colour based on their label.
    no_labels = len(set(df[grouping]))
    lightcmap = cm.get_cmap('rainbow', no_labels)
    for label in OrderedSet(df[grouping]):
        col_index = OrderedSet(df[grouping]).index(label)
        label_data = df[df[grouping]==label]
        plt.scatter(label_data[indep], label_data[dep], marker='x', color=lightcmap(col_index), alpha=0.3, label=("..."+label[-7:] if len(label)>9 else label))
    # Correct the x-axis limits for engine size
    if indep=="engine_size_cm3":
        plt.xlim((-50,7050))
    plt.xlabel(indep)
    plt.ylabel(dep)
    plt.title(graphName)
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")

    fileName = graphName + ".png"
    plt.savefig('./data/processed_vis/' + fileName, bbox_inches='tight')
    plt.clf()

def save_disc_graph(df, indep, dep, label_angles):
    """
    Saves violin plots (for categorical fields) in 'data\processed_vis'.
    """

    graphName = dep + "  VS  " + indep
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot violin plot of result distributions for each option in the categorical field.
    options = list(OrderedSet(df[indep])) # List of possible values for the field.
    no_options = len(options)
    pos = list(range(0,no_options)) # Integer positions to plot the violins over.
    data_to_plot = []
    for i in range(0,no_options): # Put list of result values for each option into a 2-d list.
        is_option = df[indep] == options[i]
        data_to_plot.append(df[dep][is_option].values.flatten().tolist())
    bp = plt.boxplot(data_to_plot, positions=pos, widths=0.8, showfliers=False, notch=False, patch_artist=False,
                     boxprops=dict(linewidth=1, color="#7c7c7c"),
                     capprops=dict(linewidth=1, color="#7c7c7c"),
                     whiskerprops=dict(linewidth=1, color="#7c7c7c"),
                     medianprops=dict(color="#7c7c7c"))
    vp = plt.violinplot(data_to_plot, positions=pos, widths=0.8, showextrema=False, showmeans=False, showmedians=False)
    
    # Change x-axis integers to labels for each option.
    ax.set_xticks(pos)
    if label_angles>0:
        options_labels = ["..."+x[-7:] if len(x)>9 else x for x in options]
        ax.set_xticklabels(options_labels, rotation = label_angles) # Truncate to end of labels if vertical labels.
    else:
        ax.set_xticklabels(options, rotation = label_angles)

    # Plot individual result values as a scatter plot for each option, overlaying the violins.
    marker_size = 400/(no_options+20) # Dynamic marker size. Decreases number of options in the field increases to save width.
    for i in range(0,no_options):
        is_option = df[indep] == options[i]
        ys = df[dep][is_option].values.flatten().tolist()
        # Add jitter to the plot by using a normal distribution (with fixed seed) centred on the option position for the x values.
        xmean = np.array([i] * df[dep][is_option].shape[0])
        np.random.seed(0)
        xs = np.random.normal(xmean, 0.05, size=len(ys)).clip(xmean - 0.2, xmean + 0.2)
        ax = plt.scatter(xs, ys, marker='x', color='#03bffe', alpha=0.3, s=marker_size, label=options[i] )

    plt.xlabel(indep)
    plt.ylabel(dep)
    plt.title(graphName)

    fileName = graphName + ".png"
    plt.savefig('./data/processed_vis/' + fileName, bbox_inches='tight')
    plt.clf()

def main():
    """
    Create visualisations from the "uk_gov_data_dense_preproc" table in the "vehicles" database. Visualisations are saved in 'data\processed_vis'.
    """
    
    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")

    # Read the UK gov data from the "vehicles" database using pandas.
    govData = pd.read_sql("SELECT * FROM uk_gov_data_dense_preproc", connection)

    # Save graphs for co2 emissions vs continuous independent fields, with separate labelling for differnt powertrains.
    for indep in ["engine_size_cm3", "power_ps"]:
        save_cts_graph(govData, indep, "co2_emissions_gPERkm", "powertrain")

    #Save graphs for co2 emissions vs categorical independent fields.
    for indep in ["manufacturer", "transmission", "transmission_type", "fuel", "powertrain"]:
        if indep == "transmission_type":
            label_angles = 0
        else:
            label_angles = 90
        save_disc_graph(govData, indep, "co2_emissions_gPERkm", label_angles)

    print("")
    print("*********************************************************************************************************************************************************************")
    print("PREPROCESSED DATA VISUALISATION REPORT")
    print("=====================================================================================================================================================================")
    print(r"""Visualisations of the "uk_gov_data_dense_preproc" data are saved in the "data\processed_vis" folder.""")
    print("")
    print("I investigate the distributions and correlations in the preprocessed vehicle emissions data.")
    print("")
    print("The features I have selected for modelling are all fields in the data other than 'model' and 'description'.")
    print("Any new real-world vehicle for use in prediction may contain unseen values for these two features, so including them is not suitable for supervised learning (without NLP).")
    print("""All other fields contain values that may help predict CO2 emissions, so I have selected them as features. I also train a model without "manufacturer" in trainTunedOrdinalNoMan.py.""")
    print("")
    print("""The primary findings upon reviewing the preprocessed data are as follows:
        (1) The median CO2 emissions for ICE, LPG, MHEV, and HEV are fairly comparable (all between 137-164). PHEVs have low median emissions of 34. BEVs have 0 emissions.
        (2) CO2 emissions and engine size are strongly positively correlated for ICEs and MHEVs, weakly positively correlated for PHEVs and HEVs, and there is no correlation for BEVs and LPGs.
        (3) CO2 emissions and power are strongly positively correlated for ICEs and MHEVs, positively correlated for HEVs, weakly positively correlated for PHEVs, and there is little/no correlation for BEVs and LPGs.""")
    print("""Some other insights I spotted but won't investigate in detail are:
        There is little difference in median emissions between vehicles using petrol and vehicles using diesel, regardless of whether the powertrain is ICE, MHEV, or PHEV.
        The manufacturer with the greatest median emissions is Rolls Royce, at over 350.""")
    print("=====================================================================================================================================================================")

    print("(1)")
    print("The median CO2 emissions for ICE, LPG, MHEV, and HEV are fairly comparable (all between 137-164). PHEVs have low median emissions of 34. BEVs have 0 emissions.")
    print("The median CO2 emissions for each type of powertrain are show below:")
    print("")
    print("The median CO2 emissions across all the data is: "
          + str( round(govData["co2_emissions_gPERkm"].median(),5) )  )
    print("The median CO2 emissions for ICEs is: .......... "
          + str( round(govData[govData["powertrain"] == "Internal Combustion Engine (ICE)"]["co2_emissions_gPERkm"].median(),5) )  )
    print("The median CO2 emissions for PHEVs is: ......... "
          + str( round(govData[govData["powertrain"] == "Plug-in Hybrid Electric Vehicle (PHEV)"]["co2_emissions_gPERkm"].median(),5) )  )
    print("The median CO2 emissions for BEVs is: .......... "
          + str( round(govData[govData["powertrain"] == "Battery Electric Vehicle (BEV) / Pure Electric Vehicle / Electric Vehicle (EV)"]["co2_emissions_gPERkm"].median(),5) )  )
    print("The median CO2 emissions for LPGs is: .......... "
          + str( round(govData[govData["powertrain"] == "Liquified Petroleum Gas (LPG)"]["co2_emissions_gPERkm"].median(),5) )  )
    print("The median CO2 emissions for MHEVs is: ......... "
          + str( round(govData[govData["powertrain"] == "Mild Hybrid Electric Vehicle (MHEV)"]["co2_emissions_gPERkm"].median(),5) )  )
    print("The median CO2 emissions for HEVs is: .......... "
          + str( round(govData[govData["powertrain"] == "Hybrid Electric Vehicle (HEV)"]["co2_emissions_gPERkm"].median(),5) )  )
    print("")
    print("""The median CO2 emissions for ICEs, MHEVs, and HEVs are the highest, all between 151-164. Vehicles with these powertrains utilise the greatest proportion of emission-producing
        fuels in the data set, so it is not unexpected that their median emissions are highest.""")
    print("""The median emissions for LPGs are slightly lower, at 137. Liquid petroleum gas produces less CO2 emissions than cars using petrol or diesel fuel, so this is not unexpected:
        https://www.drivelpg.co.uk/about-autogas/environmental-benefits/""")
    print("""The median emissions for PHEVs are the lowest of vehicles which utilise some level of emission-producing fuels. Plug-in electric vehicles utilise less emission-producing fuel than
        all types of powertrain other than BEVs, since their batteries can be plugged in and charged without necessesity of combustion. Therefore, PHEVs having the
        lowest median emissions other than BEVs is not surprising.""")
    print("""The median emissions for BEVs is 0, since BEVs are fully electric-powered, producing no emissions.""")
    print("=====================================================================================================================================================================")

    print("(2)")
    print("CO2 emissions and engine size are strongly positively correlated for ICEs and MHEVs, weakly positively correlated for PHEVs and HEVs, and there is no correlation for BEVs and LPGs.")
    print("The correlations between CO2 emissions and engine size are shown below:")
    print("")
    print("The correlation between CO2 emissions and engine size across all the data is: "
          + str( round(govData[["co2_emissions_gPERkm", "engine_size_cm3"]].corr().iloc[0,1],5) )  )
    print("The correlation between CO2 emissions and engine size for ICEs is: .......... "
          + str( round(govData[govData["powertrain"] == "Internal Combustion Engine (ICE)"][["co2_emissions_gPERkm", "engine_size_cm3"]].corr().iloc[0,1],5) )  )
    print("The correlation between CO2 emissions and engine size for PHEVs is: ......... "
          + str( round(govData[govData["powertrain"] == "Plug-in Hybrid Electric Vehicle (PHEV)"][["co2_emissions_gPERkm", "engine_size_cm3"]].corr().iloc[0,1],5) )  )
    print("The correlation between CO2 emissions and engine size for BEVs is: .......... "
          + str( round(govData[govData["powertrain"] == "Battery Electric Vehicle (BEV) / Pure Electric Vehicle / Electric Vehicle (EV)"][["co2_emissions_gPERkm", "engine_size_cm3"]].corr().iloc[0,1],5) )  )
    print("The correlation between CO2 emissions and engine size for LPGs is: .......... "
          + str( round(govData[govData["powertrain"] == "Liquified Petroleum Gas (LPG)"][["co2_emissions_gPERkm", "engine_size_cm3"]].corr().iloc[0,1],5) )  )
    print("The correlation between CO2 emissions and engine size for MHEVs is: ......... "
          + str( round(govData[govData["powertrain"] == "Mild Hybrid Electric Vehicle (MHEV)"][["co2_emissions_gPERkm", "engine_size_cm3"]].corr().iloc[0,1],5) )  )
    print("The correlation between CO2 emissions and engine size for HEVs is: .......... "
          + str( round(govData[govData["powertrain"] == "Hybrid Electric Vehicle (HEV)"][["co2_emissions_gPERkm", "engine_size_cm3"]].corr().iloc[0,1],5) )  )
    print("")
    print("""The correlations for ICEs and MHEVs are above 0.8, showing strong positive correlation. Vehicles with these powertrains utilise the greatest proportion of emission-producing
        fuels in the data set, so it is not unexpected that their engine sizes show the highest correlation with CO2 emissions.""")
    print("""The correlation HEVs is below 0.4, showing weak positive correlation. The data only holds information for HEVs in a relatively small range of engine sizes (~1000-3400cm3, less than
        half the range for ICEs), and the variation of emissions at each engine size can be quite large, particularly around 2000cm3, so the observered correlation on this data is not strong.
        Furthermore, hybrid electric vehicles make use of regenerative breaking to charge their batteries, and also switch off their combustion engines while stopped, which might have more of an impact
        in reducing emissions the largr the engine size, potentially explaining why the HEV correlation is lower than its whole combustion and mild hybrid counterparts.""")
    print("""The correlation for PHEVs is below 0.4, showing weak positive correlation. Plug-in electric vehicles utilise less emission-producing fuel than their hybrid counterparts,
        since their batteries can be plugged in and charged without necessesity of combustion. Therefore, the size of the combustion engine should have low relation to the emissions produced if the vehicle
        is using its  plug-charged battery, so it is not unexpected that their engine sizes have reduced correlations with CO2 emissions.
        The correlation excluding the 2 Ferrari SF90 outliers is """
        + str( round(govData[ (govData["powertrain"] == "Plug-in Hybrid Electric Vehicle (PHEV)") & (govData["model"] != "SF90")][["co2_emissions_gPERkm", "engine_size_cm3"]].corr().iloc[0,1],5) )
        + """, which reveals there is little/no correlation excluding the outlier supercars. Vehicles might drive the wheels using their combustion
        engine when focusing on performance over efficiency, so this is not unexpected.""")
    print("""The correlations for BEVs and LPGs are not defined:
        BEVs are fully electric-powered, so produce no emissions, and their engines do not displace any air, so their engine sizes are 0. Therefore, all points lie at (0,0) and the correlation is undefined.
        LPGs in this data all have the same engine size of 999cm3 (and are all made by Dacia), so all points lie on the vertical line (999,y) and correlation is undefined.""")
    print("=====================================================================================================================================================================")

    print("(3)")
    print("CO2 emissions and power are strongly positively correlated for ICEs and MHEVs, positively correlated for HEVs, weakly positively correlated for PHEVs, and there is little/no correlation for BEVs and LPGs.")
    print("The correlations between CO2 emissions and power are shown below:")
    print("")
    print("The correlation between CO2 emissions and power across all the data is: "
          + str( round(govData[["co2_emissions_gPERkm", "power_ps"]].corr().iloc[0,1],5) )  )
    print("The correlation between CO2 emissions and power for ICEs is: .......... "
          + str( round(govData[govData["powertrain"] == "Internal Combustion Engine (ICE)"][["co2_emissions_gPERkm", "power_ps"]].corr().iloc[0,1],5) )  )
    print("The correlation between CO2 emissions and power for PHEVs is: ......... "
          + str( round(govData[govData["powertrain"] == "Plug-in Hybrid Electric Vehicle (PHEV)"][["co2_emissions_gPERkm", "power_ps"]].corr().iloc[0,1],5) )  )
    print("The correlation between CO2 emissions and power for BEVs is: .......... "
          + str( round(govData[govData["powertrain"] == "Battery Electric Vehicle (BEV) / Pure Electric Vehicle / Electric Vehicle (EV)"][["co2_emissions_gPERkm", "power_ps"]].corr().iloc[0,1],5) )  )
    print("The correlation between CO2 emissions and power for LPGs is: .......... "
          + str( round(govData[govData["powertrain"] == "Liquified Petroleum Gas (LPG)"][["co2_emissions_gPERkm", "power_ps"]].corr().iloc[0,1],5) )  )
    print("The correlation between CO2 emissions and power for MHEVs is: ......... "
          + str( round(govData[govData["powertrain"] == "Mild Hybrid Electric Vehicle (MHEV)"][["co2_emissions_gPERkm", "power_ps"]].corr().iloc[0,1],5) )  )
    print("The correlation between CO2 emissions and power for HEVs is: .......... "
          + str( round(govData[govData["powertrain"] == "Hybrid Electric Vehicle (HEV)"][["co2_emissions_gPERkm", "power_ps"]].corr().iloc[0,1],5) )  )
    print("")
    print("""The correlations for ICEs and MHEVs are above 0.8, showing strong positive correlation. Vehicles with these powertrains utilise the greatest proportion of emission-producing
        fuels in the data set, so it is not unexpected that their powers show the highest correlation with CO2 emissions.""")
    print("""The correlation for HEVs is above 0.7, showing positive correlation. Hybrid electric vehicles are powered by batteries which are charged by a combustion engine and regenerative breaking.
        Therefore, emission-producing fuels are still the main source of energy for the vehicle, and this may explain why we see a similar (albeit weaker) correlation to ICEs and MHEVs.""")
    print("""The correlation for PHEVs is just above 0.4, showing weak positive correlation. Plug-in electric vehicles utilise less emission-producing fuel than their hybrid counterparts,
        since their batteries can be plugged in and charged without the use of combustion, so it is not unexpected that their powers have reduced correlations with CO2 emissions.
        The correlation excluding the 2 Ferrari SF90 outliers is """
        + str( round(govData[ (govData["powertrain"] == "Plug-in Hybrid Electric Vehicle (PHEV)") & (govData["model"] != "SF90")][["co2_emissions_gPERkm", "power_ps"]].corr().iloc[0,1],5) )
        + """, which reveals there is very weak correlation excluding the outlier supercars. Vehicles might drive the wheels
        using their combustion engine when focusing on performance over efficiency, so this is not unexpected.""")
    print("""There is little/no correlation for LPGs. LPGs in this data (all made by Dacia) all have powers of either 91 and 100 and emissions between 120-150, i.e. they all perform very similarly.""")
    print("""The correlation for BEVs in not defined. BEVs are fully electric-powered, so produce no emissions; all points lie on the horizontal line (x,0) and the correlation is undefined.""")
    print("*********************************************************************************************************************************************************************")

main()