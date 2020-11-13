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
    
#RQ6
#Function that calculates the overall conversion rate of the products, creates the plot of the number of purchases by category and shows the conversion rate of each category in descending order 
def conversion_rate(df_names,months):
    """
    calculate overall conversion rate
    plot of purchase by category
    calculate conversion rate for each category
    input:
    - dataframe
    - months
    output:
    - overall conversion rate for each month
    - conversion rate for each category of each month
    - plot of purchase by category of each month
    """
    for i in range(len(df_names)):
        dataset=pd.read_csv(df_names[i],usecols=['event_type','category_code'],na_filter=False)
        #NUMBER OF ALL PURCHASE PRODUCTS
        purchase=dataset[dataset.event_type=='purchase']
        totpurc=len(purchase)
        #NUMBER OF ALL VIEW PRODUCTS
        view=dataset[dataset.event_type=='view']
        totview=len(view)
        #OVERALL CONVERSION RATE OF STORE
        cr=totpurc/totview
        print ('Overall conversion rate of %s'%months[i])
        print (cr)
        #CREATE A LIST THAT CONTAINS CATEGORY NAME
        c=dataset.index
        categorie=[]
        for j in c :
            categorie.append((dataset.category_code[j].split('.')[0]))
        #CONVERT LIST IN SERIES
        x=pd.Series(categorie)
        #DELETE FROM DATASET CATEGORY_CODE
        del dataset['category_code']
        #INSERT IN DATASET CATEGORY_NAME
        dataset.insert(0,'category_name',x)
        #NUMBER OF PURCHASE FOR CATEGORY
        purc_4_category=dataset[dataset.event_type=='purchase'].groupby('category_name').agg(purchase=('event_type','count'))
        #NUMBER OF VIEW FOR CATEGORY
        view_4_category=dataset[dataset.event_type=='view'].groupby('category_name').agg(view=('event_type','count'))
        #PLOT OF NUMBER OF PURCHASE FOR CATEGORY
        purc_4_category.plot.bar(figsize = (18, 7), title='Number of purchase of %s'%months[i])
        #CONVERSION RATE FOR CATEGORY
        cr_4_cat=(purc_4_category.purchase/view_4_category.view)
        dec=cr_4_cat.sort_values(axis=0, ascending=False)
        print ('Conversion rate of each category of %s'%months[i])
        print(dec)
    return

#RQ7
#Function that demonstrates the Pareto's principle
def pareto(df_names,months):
    """
    Apply Pareto's principle
    input:
    - dataframe
    - months
    output:
    - dimostration if Pareto's principle is apply for each month
    """
    for i in range(len(df_names)):
        dataset=pd.read_csv(df_names[i],usecols=['user_id','event_type','price'],na_filter=False)
        #PURCHASE BY USERS
        purchase_by_user=dataset[dataset.event_type == 'purchase'].groupby(dataset.user_id).agg(number_of_purchases=('user_id','count'),total_spent=('price','sum'))
        purchase_by_user=purchase_by_user.sort_values('total_spent',ascending=False)
        #20% OF USERS
        user_20=int(len(purchase_by_user)*20/100)
        purch_by_user20=purchase_by_user[:user_20]
        #TOTAL SPENT BY 20% OF USERS
        spent_by_20=purch_by_user20.agg(tpc_of_users=('total_spent','sum'))
        #TOTAL PROFIT OF STORE
        profit=dataset[dataset.event_type == 'purchase'].groupby(dataset.event_type).agg(gain=('price','sum'))
        #80% OF STORE'S TOTAL PROFIT
        profit_80=(profit*80)/100
        #PERCENTAGE CHANGE BETWEEN 80% OF PROFIT AND 20% OF USERS
        percent=int((float( spent_by_20.total_spent)/float(profit_80.gain))*100)
        print("%d%% of the profit for the month of %s comes from 20%% of the user's purchases"%(percent,months[i]))
        if (percent >= 80):
            print ("For the month of %s Pareto's principle is applied." %months[i])
        else:
            print ("For the month of %s Pareto's principle isn't applied." %months[i])
    return
