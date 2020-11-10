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

            print('\t- ' + i + ' has number of Nan : %.d'  %int(df[i].isna().sum()))
        print('Total number of rows: %.d' %len(df))
        print('\n')

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
  

#RQ5
#Function that create a plot that for each day of the week shows the hourly average of visitors
def plot_hour_avg(df_names,months):
    '''
    create a plot
    input:
    -dataframe
    -months
    output:
    -plot
    '''
    for i in range(len(df_names)):
        df=pd.read_csv(df_names[i],parse_dates=['event_time'],date_parser=pd.to_datetime,usecols=['event_time','user_id'],na_filter=False)
        #hourly averege of visitors for each day
        domenica=df[df.event_time.dt.dayofweek==0].groupby(df.event_time.dt.hour).user_id.count()
        lunedi=df[df.event_time.dt.dayofweek==1].groupby(df.event_time.dt.hour).user_id.count()
        martedi=df[df.event_time.dt.dayofweek==2].groupby(df.event_time.dt.hour).user_id.count()
        mercoledi=df[df.event_time.dt.dayofweek==3].groupby(df.event_time.dt.hour).user_id.count()
        giovedi=df[df.event_time.dt.dayofweek==4].groupby(df.event_time.dt.hour).user_id.count()
        venerdi=df[df.event_time.dt.dayofweek==5].groupby(df.event_time.dt.hour).user_id.count()
        sabato=df[df.event_time.dt.dayofweek==6].groupby(df.event_time.dt.hour).user_id.count()

        plt.figure()
        plt.plot(domenica, '-o', color='royalblue', label = 'SUNDAY')
        plt.plot(lunedi, '-o', color='green', label = 'MONDAY')
        plt.plot(martedi, '-o', color='red', label = 'TUESDAY')
        plt.plot(mercoledi, '-o', color='yellow', label = 'WEDNESDAY')
        plt.plot(giovedi, '-o', color='orange', label = 'THURSDAY')
        plt.plot(venerdi, '-o', color='violet', label = 'FRIDAY')
        plt.plot(sabato, '-o', color='grey', label = 'SATURDAY')
        plt.xlabel('HOUR')
        plt.ylabel('VISITORS')
        plt.title("Daily average - %s " %months[i])
        plt.xticks(range(0,24))
        plt.legend()
        plt.show()
    return
    
