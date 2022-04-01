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
Create visualisations from the "uk_gov_data_dense" table in the "vehicles" database. Visualisations are saved in 'data\intermediate_vis'.
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
    Saves scatter graphs (for continuous fields) in 'data\intermediate_vis'.
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
    plt.savefig('./data/intermediate_vis/' + fileName, bbox_inches='tight')
    plt.clf()

def save_disc_graph(df, indep, dep, label_angles):
    """
    Saves violin plots (for categorical fields) in 'data\intermediate_vis'.
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
    plt.savefig('./data/intermediate_vis/' + fileName, bbox_inches='tight')
    plt.clf()

def main():
    """
    Create visualisations from the "uk_gov_data_dense" table in the "vehicles" database. Visualisations are saved in 'data\intermediate_vis'.
    """
    
    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")

    # Read the UK gov data from the "vehicles" database using pandas.
    govData = pd.read_sql("SELECT * FROM uk_gov_data_dense", connection)

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
    print("CLEANSED DATA VISUALISATION REPORT")
    print("=====================================================================================================================================================================")
    print(r"""Visualisations of the "uk_gov_data_dense" data are saved in the "data\intermediate_vis" folder.""")
    print("")
    print("""I investigate potential outliers and errors in the cleansed vehicle emissions data, by reviewing the fields I will utilise as features in modelling later (i.e.
        all fields in 'uk_gov_data_dense' except for 'model' and 'description').""")
    print("")
    print("""The "...VS engine_size_cm3" and "...VS power_ps" graphs seem to indicate potential outliers:
        (1) One ICE vehicle (purple) with approximately 80 emissions, 4000 engine size, and 580 power.
        (2) Two PHEV vehicles (blue) with approximately 160 emissions, 4000 engine size, and 770 power.
        (3) One HEV vehicle (red) with approximately 310 emissions, 2500 engine size, and 190 power.""")
    print("=====================================================================================================================================================================")

    print("(1)")
    print("The ICE outlier is:")
    print("")
    print(govData[(govData["powertrain"]=="Internal Combustion Engine (ICE)")&(govData["co2_emissions_gPERkm"]<90)])
    print("")
    print("""Reviewing online resources and comparing with another "G-Class MY 201.5" entry in the table suggests these emissions are too low for this vehicle, especially for an SUV:
        https://www.cars-data.com/en/mercedes-benz-g-63-amg-specs/82510/tech
        https://www.nextgreencar.com/view-car/67506/mercedes-benz-g-class-g-63-amg-4matic-auto-petrol-semi-automatic-7-speed/""")
    print("")
    print(govData[ (govData["description"]=="AMG G 63")&(govData["co2_emissions_gPERkm"]>90) ])
    print("")
    print("""Therefore, I will assume there was an error in entering the data and I will replace these emissions with "377.0" in dataPreprocess.py, resolving outlier (1).""")
    print("=====================================================================================================================================================================")

    print("(2)")
    print("The PHEV outliers are:")
    print("")
    print(govData[ (govData["powertrain"]=="Plug-in Hybrid Electric Vehicle (PHEV)")&(govData["co2_emissions_gPERkm"]>150)])
    print("")
    print("""Reviewing online resources suggests the data for these vehicles are accurate (the Fiorano appears to be a high-performance version of the Stradale):
        https://www.cars-data.com/en/ferrari-sf90-stradale-specs/104720/tech
        https://www.roadandtrack.com/new-cars/future-cars/a27626676/ferrari-sf90-stradale-hybrid-hypercar-power-specs-photos/""")
    print("")
    print("Therefore, I will leave these vehicles as are in the data as they are valid, resolving outliers (2).")
    print("=====================================================================================================================================================================")

    print("(3)")
    print("The HEV outlier is:")
    print("")
    print(govData[ (govData["powertrain"]=="Hybrid Electric Vehicle (HEV)")&(govData["co2_emissions_gPERkm"]>300)])
    print("")
    print("""Reviewing online resources and comparing with another "Galaxy Model Year Post 2021, 2.5 Duratec (FHEV)" entry in the table suggests these emissions are too high for this vehicle:
        https://www.whatcar.com/ford/galaxy/mpv/25-fhev-190-titanium-5dr-cvt/96967
        https://www.motorparks.co.uk/technical-data/ford/galaxy/2.5-fhev-190-titanium-5dr-cvt""")
    print("")
    print(govData[ (govData["model"]=="Galaxy Model Year Post 2021")&(govData["description"]=="2.5 Duratec (FHEV)")&(govData["co2_emissions_gPERkm"]<300)])
    print("")
    print("""Therefore, I will assume there was an error in entering the data and I will replace these emissions with "148.0" in dataPreprocess.py, resolving outlier (3).""")
    print("*********************************************************************************************************************************************************************")

main()