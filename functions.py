import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

import seaborn as sns
import copy
import os

import datetime
import time

# Function that provide some information about the cvs files
def infos(old_df_names, months):
    """
    Print informations about databases
    input:
    - dataframe
    - months
    output:
    - months, number of NaN values in each column
    """
    for i in range(len(old_df_names)):
        
        df = pd.read_csv(old_df_names[i])

        print('Month %s :' %months[i])

        for i in df.columns:

            print('   -' + i + ' has number of Nan : %.d'  %int(df[i].isna().sum()))
        print('Total number of rows: %.d' %len(df))

    return 

# Function that clean the databases from NaN values
def clean_dataframe(df):
    """
    Clean the dataframe, removing NaN from columns
    input:
    - dataframe
    output:
    - cleaned dataframe
    """
    df.dropna(inplace = True)
    return df

# Function that create new csv files
def make_new_csv(old_df_names, df_names):
    """
    Make new csv files
    input:
    - dataframe
    output:
    - new csv files
    """
    for i in range(len(old_df_names)):
        
        df = pd.read_csv(old_df_names[i])
        # cleaning function
        df = clean_dataframe(df)
        df.to_csv(df_names[i], index=False)

    return
    

