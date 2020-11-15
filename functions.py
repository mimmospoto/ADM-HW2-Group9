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

# RQ1 functions

# RQ1.1 functions
def compute_average_session(df_names):
    """
    Compute average number of times users perform view/cart/purchase within each session
    input:
    - list of names of csv files to open
    output:
    - series of average of each operation
    """    
    # init the daily average dict
    average_session_dict = {}

    for i in range(len(df_names)):
        average_session_dict[i] = {}
        # load the ith dataframe, taking the event_type and user_session columns
        df = pd.read_csv(df_names[i], usecols=['event_type', 'user_session'])

        for j in df['event_type'].unique():
            #print('{} of {:d} has average of : {:.2f} ' .format(j, i, float(df[df['event_type'] == j].groupby(['user_session']).count().mean())))
            average_session_dict[i][j] = df[df['event_type'] == j].groupby(['user_session']).count().mean()

    average_session_df = pd.DataFrame(average_session_dict).mean(axis=1)

    return average_session_df

def plot_average_session(average_session_df, months):
    """
    plots the average number of times users perform each operation
    """
    # plot average_session_df
    fig = plt.figure()
    X = np.arange(len(average_session_df))
    plt.bar(X, average_session_df)
    plt.xticks(np.arange(len(average_session_df)),average_session_df.index)
    plt.ylabel("average operation per session")
    plt.xlabel("operations")
    plt.title("Average number of times users perform each operation within a session")
    plt.grid(color ='silver', linestyle = ':')
    fig.set_figwidth(15)
    fig.set_figheight(5)

    return

# RQ1.2 functions
def compute_average_view_cart(df_names, months):
    """
    Compute average number of times a user views a product before adding it to the cart
    input:
    - list of names of csv files to open
    output:
    - the average of how many times a product is viewed before to be added to the cart 
    """   

    # init a dataframe with index as every months and column as the mean for each user
    df_mean_database = pd.DataFrame(index=months, columns=['mean'])
    for i in range(len(df_names)):
        # load the ith dataframe, taking the event_time, event_type, product_id, user_id columns
        df = pd.read_csv(df_names[i],
            usecols=['event_time','event_type', 'product_id', 'user_id'], nrows=100000,
            parse_dates=['event_time'])

        # cut off the 'purchase' variable from event_type
        df_2 = df[df['event_type'] != 'purchase']
        
        df_3 = df_2[df_2.event_type=='view'].groupby(by=['product_id']).agg(view=('event_type', 'count'))

        df_4 = df_2[df_2.event_type=='cart'].groupby(by=['product_id']).agg(cart=('event_type', 'count'))


        # get dataframe where event_type is equal to 'cart'
        df_cart = df_2[df_2['event_type']=='cart']

        # init a dataframe with index as every user and column as the mean for each user
        df_mean_user = pd.DataFrame(index=df_cart['user_id'].unique(), columns=['mean'])

        df_cart.groupby(by=['user_id']).count()
        for user in df_cart['user_id'].unique():
            # get dataframe with one user at a time
            df_user = df_2[df_2['user_id'] == user]
            # init the dict where the key are the products and the values are the mean of each product
            product_dict = {}

            for prod in df_user['product_id'].unique():
                # get dataframe with one product at a time
                df_product = df_user[df_user['product_id'] == prod]
                df_product_2 = df_product.copy()

                product_dict[prod] = []

                # init a list to append how many times 'view' appears before 'cart' for each product
                product_lst = []

                # check if at least a 'view' exist in the dataframe otherwise pass 
                if any(df_product_2['event_type'] == 'view') == True:
                    df_product_2_time = df_product_2[df_product_2['event_type'] == 'view'].event_time.reset_index(drop=True)[0]
                
                # check if there are some 'cart' event before the 'view' event (only for the first time of seeing the 'cart')
                if any(df_product_2[df_product_2['event_type'] == 'cart'].event_time <= df_product_2_time) == True:
                    df_product_3 = df_product_2[df_product_2.event_time <= df_product_2_time]
                    # drop any 'cart' events at the beginning
                    df_product_2 = df_product_2.drop(labels=df_product_3[df_product_3['event_type'] == 'cart'].index)

                # count how many times 'view' is before 'cart'
                if any(df_product_2['event_type'] == 'view') == True:
                    for index, row in df_product_2.iterrows():
                        
                        if row['event_type'] == 'cart':
                            
                            product_lst.append(np.sum(df_product_2[df_product['event_type'] == 'view'].event_time < row['event_time']))
                            df_product_2 = df_product_2[df_product_2.event_time > row['event_time']]
                
                # compute mean for each product
                if len(product_lst) > 0:
                    product_dict[prod] = [i for i in product_lst if i != 0]
                    product_dict[prod] = np.mean(product_dict[prod])
                else:
                    product_dict[prod].append(0)
            
            # compute mean for each user
            try:
                df_mean_user.loc[user,'mean'] = round(pd.DataFrame(product_dict).mean(axis=1)[0], 2)
            except ValueError:
                df_mean_user.loc[user,'mean'] = round(product_dict[prod], 2)
        
        # compute final average for a user for a product
        df_mean_user.dropna(inplace=True) 
        mean_prod_user = np.mean(df_mean_user)
    
        # add final average per month
        df_mean_database.loc[months[i], 'mean'] =  round(mean_prod_user[0], 2)

    df_mean_database.dropna(inplace=True) 
    final_mean = np.mean(df_mean_database)    
    return final_mean

# RQ1.3 functions
def compute_probability_cart_purchase(df_names, months):
    """
    Compute the probability that products are bought once is added to the cart
    input:
    - list of names of csv files to open
    output:
    - probability products are purchased once are added to the cart
    """   
    # init dictionary to merge each monthly datasets 
    df_database = {}
    for i in range(len(df_names)):
        # load the ith dataframe, taking only the event_type
        df = pd.read_csv(df_names[i],
            usecols=['event_type'], na_filter=False)
        
        # cut off the view variable from event_type
        df_database[months[i]] = df[df['event_type'] != 'view']

    # function to concatenate each dataset
    merged_df = pd.concat([df_database[months[i]] for i in range(len(df_database))])
    # compute probability as the ratio between purchase and cart events
    prob = round(merged_df[merged_df['event_type'] == 'purchase'].shape[0] / 
                 merged_df[merged_df['event_type'] == 'cart'].shape[0], 4) * 100
    
    return prob

# RQ1.4 functions
def compute_average_time_removed_item(df_names):
    """
    Compute the average time an item stays in the cart before being removed
    input:
    - list of names of csv files to open
    output:
    - average time
    """      
    df_mean_database = pd.DataFrame(index=months, columns=['mean'])
    for i in range(len(df_names)):
        # load the ith dataframe, taking only the 
        df = pd.read_csv(df_names[i],
            usecols=['event_time', 'event_type', 'product_id'], nrows=100000,
            parse_dates=['event_time'])

        # cut off the view variable from event_type
        df_2 = df[df['event_type'] != 'view']
        
        # init the dict where the key are the products and the values are the mean of each product
        product_dict = {}
        # loop through the event_type 'purchase' to find unique product_id
        for prod in df_2[df_2['event_type'] == 'purchase']['product_id'].unique():
            df_product = df_2[df_2['product_id'] == prod]

            # check if at least a 'cart' event exist             
            if df_product['event_type'].str.contains('cart').any():
                pass
            else:
                continue

            # check if there are some 'purchase' event before the 'cart' event (only for the first time of seeing the 'purchase')
            if any(df_product[df_product['event_type'] == 'purchase'].event_time <= 
                    df_product[df_product['event_type'] == 'cart'].event_time.reset_index(drop=True)[0]) == True:
                df_3 = df_product[df_product.event_time <= df_product[df_product['event_type'] == 'cart'].event_time.reset_index(drop=True)[0]]
                # drop any 'cart' events at the beginning
                df_product = df_product.drop(labels=df_3[df_3['event_type'] == 'purchase'].index)

            
            dist_prod = df_product.event_time[df_product.event_type == 'purchase'].values - df_product.event_time[df_product.event_type == 'cart'].values

            product_dict[prod].append(np.mean(dist_prod))

        # add final average per month
        df_mean_database.loc[months[i], 'mean'] =  pd.DataFrame(product_dict).mean(axis=1)[0]

# RQ1.5 functions
def compute_average_time_first_view(df_names, months):
    """
    Compute the probability that products are bought once is added to the cart
    input:
    - list of names of csv files to open
    output:
    - probability products are purchased once are added to the cart
    """   
    
    df_mean_database = pd.DataFrame(index=months, columns=['mean'])
    for i in range(len(df_names)):
        # load the ith dataframe, taking only the 
        df = pd.read_csv(df_names[i],
            usecols=['event_time', 'event_type', 'product_id'],
            parse_dates=['event_time'])

        # cut off the view variable from event_type
        df_2 = df.copy()
        
        df_3 = df[df['event_type'] != 'view']

        # init the dict where the key are the products and the values are the mean of each product
        product_dict = {}
        # loop through the event_type 'purchase' to find unique product_id
        for prod in df_3['product_id'].unique():
            df_product = df_2[df_2['product_id'] == prod]
            
            # 
            if any(df_product['event_type'] == 'view') == False:
                continue

            # check if there are some 'purchase' event before the 'view' event (only for the first time of seeing the 'purchase')
            if any(df_product[df_product['event_type'] == 'purchase'].event_time <= 
                df_product[df_product['event_type'] == 'view'].event_time.reset_index(drop=True)[0]) == True:
                df_3 = df_product[df_product.event_time <= df_product[df_product['event_type'] == 'view'].event_time.reset_index(drop=True)[0]]
                # drop any 'cart' events at the beginning
                df_product = df_product.drop(labels=df_3[df_3['event_type'] == 'purchase'].index)
        
            # check if there are some 'cart' event before the 'view' event (only for the first time of seeing the 'purchase')
            if any(df_product[df_product['event_type'] == 'cart'].event_time <= 
                    df_product[df_product['event_type'] == 'view'].event_time.reset_index(drop=True)[0]) == True:
                df_3 = df_product[df_product.event_time <= df_product[df_product['event_type'] == 'view'].event_time.reset_index(drop=True)[0]]
                # drop any 'cart' events at the beginning
                df_product = df_product.drop(labels=df_3[df_3['event_type'] == 'cart'].index)

            # 
            if any(df_product['event_type'] == 'purchase') == False:
                continue
            elif any(df_product['event_type'] == 'cart') == False:
                continue

            product_dict[prod] = []

            df_product.drop_duplicates(subset=['event_type'], keep='first', inplace=True)

            df_product.reset_index(inplace=True) 
            product_dict[prod].append(df_product.event_time[1] - df_product.event_time[0])
        
    
        # add final average per month
        df_mean_database.loc[months[i], 'mean'] =  pd.DataFrame(product_dict).mean(axis=1)[0]
        
    return df_mean_database
        

# RQ2 functions

def compute_number_sold_per_category(df_names, months):
    """
    Compute the probability that products are bought once is added to the cart
    input:
    - list of names of csv files to open
    output:
    - 
    """    
    # init a dataframe with index as months and column as most sold product
    df_final = pd.DataFrame(columns=months)
    df_final_2 = {}
    for i in range(len(df_names)):
        # load the ith dataframe, taking only the 
        df = pd.read_csv(df_names[i],
            usecols=['product_id', 'category_code'])

        new = df['category_code'].str.split(".", expand=True)
        df['category_1'] = new[0]
        df['category_1'].unique()
        df.drop(columns=['category_code'], inplace=True)

        df_final[months[i]] = pd.DataFrame(df.groupby(by=['category_1']).count().sort_values('product_id', ascending=False), columns=months[i])
        df_final_2[months[i]] = pd.DataFrame(df.groupby(by=['category_1']).count().sort_values('product_id', ascending=False))

    pd.DataFrame(df_final_2)


# RQ4 functions
def make_df_purchase(df_names, months):
    df_purchase = {}
    for i in range(len(df_names)):
        data = pd.read_csv(df_names[i], usecols=['brand', 'price', 'event_type'], nrows=1000000)
        df_purchase[months[i]] = data[data['event_type'] == 'purchase']
    return df_purchase

def earning_per_month(df_purchase, months):
    dict_earning = {}
    for i in range(len(df_purchase)):
        data = df_purchase[months[i]]
        dict_earning[months[i]] = data.loc[data.event_type == 'purchase'].groupby('brand', as_index=False).sum()
    return dict_earning

def brand_per_month(brand, dict_earning, months):
    df_profit = {}
    for i in range(len(months)):
        try:
            df_profit[months[i]] = dict_earning[months[i]].loc[dict_earning[months[i]].brand == brand, 'price'].values[0]
        except IndexError:
            df_profit[months[i]] = 0
    return df_profit

def find_3_worst_brand(dict_earning, months):
    data_frames = [dict_earning[months[i]] for i in range(len(dict_earning))]

    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['brand'],
                                        how='outer'), data_frames)

    df_merged.set_index('brand', inplace=True)
    df_merged.set_axis(months, axis=1, inplace=True)

    df_pct = df_merged.T.pct_change()

    worst_brand = []
    worst_value = []
    worst_months = []

    for i in range(0,3):

        worst_brand.append(df_pct.min().sort_values().index[i])
        worst_value.append(round(abs(df_pct.min().sort_values()[i])*100, 2))

        L = list(df_pct[df_pct[worst_brand[i]] == df_pct.min().sort_values()[i]].index.values)
        worst_months.append(''.join(L))


    for j in range(0,3):
        print('{} lost {}% bewteen {} and the month before'.format(worst_brand[j], worst_value[j], worst_months[j]), end=' \n')
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
