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
from sklearn.model_selection import train_test_split

"""
Create visualisations of the train/test data sets from the "uk_gov_data_dense_preproc" table. Visualisations are saved in "results\models\train_test_vis".
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
    r"""
    Saves scatter graphs (for continuous fields) in 'results\models\train_test_vis'.
    """

    graphName = dep + "  VS  " + indep
    
    # Define "no_labels" number of colours for different labels in the "grouping" field, and plot points in a scatter with colour based on their label.
    for label in OrderedSet(df[grouping]):
        if label=="train":
            col = "#03bffe"
        else:
            col = "#ff845b"
        label_data = df[df[grouping]==label]
        plt.scatter(label_data[indep], label_data[dep], marker='x', color=col, alpha=0.3, label=label)
    plt.xlabel(indep)
    plt.ylabel(dep)
    plt.title(graphName)
    plt.legend(loc="best")

    fileName = graphName + ".png"
    plt.savefig('./results/models/train_test_vis/' + fileName, bbox_inches='tight')
    plt.clf()

def save_hist(df, var):
    r"""
    Saves histograms in 'results\models\train_test_vis'.
    """

    graphName = var + " Layered Histogram"
    
    # Construct histogram for "var" where bars for "test" are layered over the top of bars for "train".
    train_data = df[ df["prefix"]=="train" ][var]
    test_data = df[ df["prefix"]=="test" ][var]
    plt.hist( train_data, 50, color="#03bffe", density=False, histtype="bar", stacked=True, label="train")
    plt.hist( test_data, 50, color="#ff845b", density=False, histtype="bar", stacked=True, label="test")
    plt.xlabel(var)
    plt.ylabel("frequency")
    plt.title(graphName)
    plt.legend(loc="best")

    fileName = graphName + ".png"
    plt.savefig('./results/models/train_test_vis/' + fileName, bbox_inches='tight')
    plt.clf()

def save_disc_graph(df, indep, dep, label_angles):
    r"""
    Saves violin plots (for categorical fields) in 'results\models\train_test_vis'.
    """

    graphName = dep + "  VS  " + indep
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot violin plot of result distributions for each option in the categorical field.
    options = list(OrderedSet(df[indep])) # List of possible values for the field.
    no_options = len(options)
    pos = np.arange(0,no_options) # Integer positions to plot the violins over.

    # Plot a boxplot and violin of distribution of "indep" for each option in "dep" for training data.
    data_to_plot = []
    for i in range(0,no_options): # Put list of result values for each option into a 2-d list.
        is_option = df[ df["prefix"]=="train" ][indep] == options[i]
        data_to_plot.append(df[ df["prefix"]=="train" ][dep][is_option].values.flatten().tolist())
    bp = plt.boxplot(data_to_plot, positions=3*pos, widths=0.8, showfliers=False, notch=False, patch_artist=False,
                     boxprops=dict(linewidth=1, color="#7c7c7c"),
                     capprops=dict(linewidth=1, color="#7c7c7c"),
                     whiskerprops=dict(linewidth=1, color="#7c7c7c"),
                     medianprops=dict(color="#7c7c7c"))
    vp = plt.violinplot(data_to_plot, positions=3*pos, widths=0.8, showextrema=False, showmeans=False, showmedians=False)

    # Plot a boxplot and violin of distribution of "indep" for each option in "dep" for testing data.
    data_to_plot = []
    for i in range(0,no_options): # Put list of result values for each option into a 2-d list.
        is_option = df[ df["prefix"]=="test" ][indep] == options[i]
        data_to_plot.append(df[ df["prefix"]=="test" ][dep][is_option].values.flatten().tolist())
    bp = plt.boxplot(data_to_plot, positions=3*pos+1, widths=0.8, showfliers=False, notch=False, patch_artist=False,
                     boxprops=dict(linewidth=1, color="#7c7c7c"),
                     capprops=dict(linewidth=1, color="#7c7c7c"),
                     whiskerprops=dict(linewidth=1, color="#7c7c7c"),
                     medianprops=dict(color="#7c7c7c"))
    vp = plt.violinplot(data_to_plot, positions=3*pos+1, widths=0.8, showextrema=False, showmeans=False, showmedians=False)
    
    # Change x-axis integers to labels for each option.
    ax.set_xticks(3*pos+0.5)
    if label_angles>0:
        options_labels = ["..."+x[-7:] if len(x)>9 else x for x in options]
        ax.set_xticklabels(options_labels, rotation = label_angles) # Truncate to end of labels if vertical labels.
    else:
        ax.set_xticklabels(options, rotation = label_angles)

    marker_size = 400/(no_options+20) # Dynamic marker size. Decreases number of options in the field increases to save width.

    # Plot scatter of "indep" values for each option in "dep" for training data.
    for i in range(0,no_options):
        is_option = df[ df["prefix"]=="train" ][indep] == options[i]
        ys = df[ df["prefix"]=="train" ][dep][is_option].values.flatten().tolist()
        # Add jitter to the plot by using a normal distribution (with fixed seed) centred on the option position for the x values.
        xmean = np.array([3*i] * df[ df["prefix"]=="train" ][dep][is_option].shape[0])
        np.random.seed(0)
        xs = np.random.normal(xmean, 0.05, size=len(ys)).clip(xmean - 0.2, xmean + 0.2)
        if i==0: # Prevent legend displaying multiple "train" labels.
            ax = plt.scatter(xs, ys, marker='x', color='#03bffe', alpha=0.3, s=marker_size, label="train")
        else:
            ax = plt.scatter(xs, ys, marker='x', color='#03bffe', alpha=0.3, s=marker_size)

    # Plot scatter of "indep" values for each option in "dep" for testing data.
    for i in range(0,no_options):
        is_option = df[ df["prefix"]=="test" ][indep] == options[i]
        ys = df[ df["prefix"]=="test" ][dep][is_option].values.flatten().tolist()
        # Add jitter to the plot by using a normal distribution (with fixed seed) centred on the option position for the x values.
        xmean = np.array([3*i+1] * df[ df["prefix"]=="test" ][dep][is_option].shape[0])
        np.random.seed(0)
        xs = np.random.normal(xmean, 0.05, size=len(ys)).clip(xmean - 0.2, xmean + 0.2)
        if i==0: # Only label data on the first option to prevent legend displaying multiple "test" labels.
            ax = plt.scatter(xs, ys, marker='x', color='#ff845b', alpha=0.3, s=marker_size, label="test")
        else:
            ax = plt.scatter(xs, ys, marker='x', color='#ff845b', alpha=0.3, s=marker_size)

    plt.xlabel(indep)
    plt.ylabel(dep)
    plt.title(graphName)
    plt.legend(loc="best")

    fileName = graphName + ".png"
    plt.savefig('./results/models/train_test_vis/' + fileName, bbox_inches='tight')
    plt.clf()

def save_bar(df, indep, label_angles):
    r"""
    Saves bar charts (for number of samples in categorical fields) in 'results\models\train_test_vis'.
    """

    graphName = indep + " Bar Chart"
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot violin plot of result distributions for each option in the categorical field.
    options = list(OrderedSet(df[indep])) # List of possible values for the field.
    no_options = len(options)
    pos = np.arange(0,no_options) # Integer positions to plot the violins over.

    # Plot a bar for the total number of examples of each option in "dep" for training data.
    heights_to_plot = []
    option_totals = []
    for i in range(0,no_options): # Put list of result values for each option into a 2-d list.
        is_option = df[ df["prefix"]=="train" ][indep] == options[i]
        option_totals.append(df[ df[indep] == options[i] ].shape[0])
        heights_to_plot.append(df[ df["prefix"]=="train" ][is_option].shape[0])
    bp = plt.bar(3*pos, heights_to_plot, width=0.8, color="#03bffe", **{"label": "train"})
    # Add labels above each bar depicting the proportion of the total data set.
    for i in range(0,len(bp)):
        height = bp[i].get_height()
        ax.text(bp[i].get_x() + bp[i].get_width()/2., height,
                str(int(round( (heights_to_plot[i]/option_totals[i])*100) ))+"%",
                ha='center', va='bottom', size=6)

    # Plot a bar for the total number of examples of each option in "dep" for testing data.
    heights_to_plot = []
    for i in range(0,no_options): # Put list of result values for each option into a 2-d list.
        is_option = df[ df["prefix"]=="test" ][indep] == options[i]
        heights_to_plot.append(df[ df["prefix"]=="test" ][is_option].shape[0])
    bp = plt.bar(3*pos+1, heights_to_plot, width=0.8, color="#ff845b", **{"label": "test"})
    # Add labels above each bar depicting the proportion of the total data set.
    for i in range(0,len(bp)):
        height = bp[i].get_height()
        ax.text(bp[i].get_x() + bp[i].get_width()/2., height,
                str(int(round( (heights_to_plot[i]/option_totals[i])*100) ))+"%",
                ha='center', va='bottom', size=6)
    
    # Change x-axis integers to labels for each option.
    ax.set_xticks(3*pos+0.5)
    if label_angles>0:
        options_labels = ["..."+x[-7:] if len(x)>9 else x for x in options]
        ax.set_xticklabels(options_labels, rotation = label_angles) # Truncate to end of labels if vertical labels.
    else:
        ax.set_xticklabels(options, rotation = label_angles)

    plt.xlabel(indep)
    plt.ylabel("frequency")
    plt.title(graphName)
    plt.legend(loc="best")

    fileName = graphName + ".png"
    plt.savefig('./results/models/train_test_vis/' + fileName, bbox_inches='tight')
    plt.clf()

def main():
    r"""
    Create visualisations of the train/test data sets from the "uk_gov_data_dense_preproc" table. Visualisations are saved in "results\models\train_test_vis".
    """
    
    connection = create_database_connection("localhost", configprivate.username, configprivate.password, "vehicles")

    # Read the UK gov data from the "vehicles" database using pandas.
    govData = pd.read_sql("SELECT * FROM uk_gov_data_dense_preproc", connection)

    # Define training and testing sets (with "random_state" set to 1 to ensure the same sets are used in model training).
    testSize = 0.2
    train, test = train_test_split(govData, test_size = testSize, random_state=1)
    train.insert(11, "prefix", "train")
    test.insert(11, "prefix", "test")
    govDataSplit = pd.concat([train, test])

    # Save graphs for co2 emissions vs continuous independent fields, with separate labelling for different powertrains.
    for indep in ["engine_size_cm3", "power_ps"]:
        save_cts_graph(govDataSplit, indep, "co2_emissions_gPERkm", "prefix")

    # Save histograms comparing distributions of training and testing data.
    for var in ["co2_emissions_gPERkm", "engine_size_cm3", "power_ps"]:
        save_hist(govDataSplit, var)

    # Save graphs for co2 emissions vs categorical independent fields.
    for indep in ["transmission_type", "fuel", "powertrain"]:
        if indep == "transmission_type":
            label_angles = 0
        else:
            label_angles = 90
        save_disc_graph(govDataSplit, indep, "co2_emissions_gPERkm", label_angles)
        save_bar(govDataSplit, indep, label_angles)

    print("")
    print("*********************************************************************************************************************************************************************")
    print("TRAIN/TEST DATA VISUALISATION REPORT")
    print("=====================================================================================================================================================================")
    print(r"""Visualisations of the "uk_gov_data_dense_preproc" train/test data sets are saved in the "results\models\train_test_vis" folder.""")
    print("")
    print(r"""The primary findings upon reviewing the train/test data sets are:
        (1) VS categorical features: The medians and IQR's of the CO2 emissions for each option in "fuel", "powertrain", and "transmission_type" appear to be similar across the board.
        (2) VS continuous features: There doesn't appear to be any clear bias in where the train and test data have been selected, with the testing set also picking up some of the extreme examples.
        (3) Histograms: The distributions of "co2_emissions_gPERkm", "engine_size_cm3", and "power_ps" appear to have similar shapes between the train and test sets.
        (4) Barcharts: The frequencies of each option in each categorical feature are broadly as expected: 80% for train, 20% for test (except for LPG at 90%:10%).""")
    print("")
    print(r"""Overall, the train and test data each appear to be representative of the whole data, except for LPG vehicles being split 90%:10%. However, this is only a small portion of
        the whole data set and should not have a significant impact on model accuracy.""")
    print("""Therefore, it does not appear that either the train or the test set should be easier to predict than the other, and there shouldn't be significant bias in a trained model.""")
    print("*********************************************************************************************************************************************************************")

main()