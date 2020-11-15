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
            usecols=['event_type'])
        
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
    df_final = {}
    for i in range(len(df_names)):
        # load the ith dataframe, taking only the 
        df = pd.read_csv(df_names[i],
            usecols=['product_id', 'category_code', 'event_type'], nrows=1000000)

        df = df[df['event_type'] == 'purchase']
        new = df['category_code'].str.split(".", expand=True)
        df['category_1'] = new[0]
        df.drop(columns=['category_code', 'event_type'], inplace=True)


        df_final[months[i]] = df.groupby(by=['category_1']).count().sort_values('product_id', ascending=False)
    df_final = [df_final[months[i]] for i in range(len(df_final))]

    return df_final

def plot_number_sold_per_category(df_final, months):
    """
    plots the number of sold product per category per month
    """
    # plot average_session_df
    fig, a = plt.subplots(4,2)
    # Plot 1
    df_final[0].reset_index().plot(kind='bar', y='product_id', x='category_1', ax=a[0][0])
    a[0][0].set(title=months[0], xlabel='Categories', ylabel='Total Sales')
    a[0][0].tick_params(labelrotation=45)
    a[0][0].get_legend().remove()
    a[0][0].grid(color ='silver', linestyle = ':')
    # Plot 2
    df_final[1].reset_index().plot(kind='bar', y='product_id', x='category_1', ax=a[0][1])
    a[0][1].set(title=months[1], xlabel='Categories', ylabel='Total Sales')
    a[0][1].tick_params(labelrotation=45)
    a[0][1].get_legend().remove()
    a[0][1].grid(color ='silver', linestyle = ':')
    # Plot 3
    df_final[2].reset_index().plot(kind='bar', y='product_id', x='category_1', ax=a[1][0])
    a[1][0].set(title=months[2], xlabel='Categories', ylabel='Total Sales')
    a[1][0].tick_params(labelrotation=45)
    a[1][0].get_legend().remove()
    a[1][0].grid(color ='silver', linestyle = ':')
    # Plot 4
    df_final[3].reset_index().plot(kind='bar', y='product_id', x='category_1', ax=a[1][1])
    a[1][1].set(title=months[3], xlabel='Categories', ylabel='Total Sales')
    a[1][1].tick_params(labelrotation=45)
    a[1][1].get_legend().remove()
    a[1][1].grid(color ='silver', linestyle = ':')
    # Plot 5
    df_final[4].reset_index().plot(kind='bar', y='product_id', x='category_1', ax=a[2][0])
    a[2][0].set(title=months[4], xlabel='Categories', ylabel='Total Sales')
    a[2][0].tick_params(labelrotation=45)
    a[2][0].get_legend().remove()
    a[2][0].grid(color ='silver', linestyle = ':')
    # Plot 6
    df_final[5].reset_index().plot(kind='bar', y='product_id', x='category_1', ax=a[2][1])
    a[2][1].set(title=months[5], xlabel='Categories', ylabel='Total Sales')
    a[2][1].tick_params(labelrotation=45)
    a[2][1].get_legend().remove()
    a[2][1].grid(color ='silver', linestyle = ':')
    # Plot 7
    df_final[6].reset_index().plot(kind='bar', y='product_id', x='category_1', ax=a[3][0])
    a[3][0].set(title=months[6], xlabel='Categories', ylabel='Total Sales')
    a[3][0].tick_params(labelrotation=45)
    a[3][0].get_legend().remove()
    a[3][0].grid(color ='silver', linestyle = ':')
    a[3][1].axis('off')

    # Title the figure
    fig.suptitle('Category of the most trending products overall', fontsize=14, fontweight='bold')

    fig.set_figwidth(20)
    fig.set_figheight(5)
    plt.show()
    return

def plot_most_visited_subcategories(df_names, months):
    """
    plots the most visited subcategories
    """    
    # init a dataframe with index as months and column as most sold product
    df_final = {}
    for i in range(len(df_names)):
        # load the ith dataframe, taking only the 
        df = pd.read_csv(df_names[i],
            usecols=['event_type', 'category_code'], nrows=100000)

        df = df[df['event_type'] == 'view']
        new = df['category_code'].str.split(".", expand=True)
        df['subcategory'] = new[1]
        df.drop(columns=['category_code'], inplace=True)


        df_final[months[i]] = df.groupby(by=['subcategory']).count().sort_values('event_type', ascending=False)
    
    df_final = [df_final[months[i]] for i in range(len(df_final))]
    merged_df = pd.concat([df_final[i] for i in range(len(df_final))]).reset_index()

    df_tot = merged_df.groupby(by=['subcategory']).sum().sort_values('event_type', ascending=False).rename(columns={'event_type': 'view'}).reset_index()

    # plot most visited subcategories
    fig = plt.figure()
    X = np.arange(len(df_tot))
    plt.barh(X, df_tot['view'])
    plt.yticks(np.arange(len(df_tot)),df_tot['subcategory'])
    plt.ylabel("views")
    plt.xlabel("subcategories")
    plt.title("Most visited subcategories")
    plt.grid(color ='silver', linestyle = ':')
    fig.set_figwidth(15)
    fig.set_figheight(15)
    plt.show()
    return

def plot_10_most_sold(df_final, months):
    """
    plots the 10 most sold product per category
    """    
    merged_df = pd.concat([df_final[i] for i in range(len(df_final))]).reset_index()
    df_tot = merged_df.groupby(by=['category_1']).sum().sort_values('product_id', ascending=False).rename(columns={'event_type': 'view'})[:10]
    return df_tot


# RQ3 functions
# Function used for showing the values of the bars in the plots of RQ3
def plot_values_in_barh(y):
    for index, value in enumerate(y):
        plt.text(value, index, str(round(value, 2)))


# Function that given a category in input, returns a plot with the average price per brand for the selected category
def plot_average_price_per_category(category, df_names):
    # Initializing an empty list where we will put every grouped-by DataFrame later on
    l = []
    # Starting a for loop to read every DataFrame
    for i in range(len(df_names)):
        # Selecting the columns to use for this task
        data = pd.read_csv(df_names[i], usecols=['category_code', 'brand', 'price'])
        # For every category_code and brand, calculating the average price of the products, then i reset the index
        # because i do not want to work with MultiIndex
        a = data.groupby(['category_code', 'brand']).mean().reset_index()
        # Appending the DataFrame analyzed for 1 month to the list l
        l.append(a)
    # Concatenating every DataFrame of each month grouped by category_code and brand in one DataFrame that will not
    # be memory expensive
    final = pd.concat(l)
    # Grouping again by category_code and brand after the concatenation. We reset again the index for the same
    # reason as before
    final2 = final.groupby(['category_code', 'brand']).mean().reset_index()
    # Selecting the category_code we want to analyze
    fplot = final2.loc[final2['category_code'] == category]
    # Setting the values to show in the plot at the end of the bars 
    y = list(fplot['price'])
    # Assigning a variable to the plot
    end = fplot.plot(x='brand', kind='barh', figsize=(20, 60))
    # Returning the plot and calling the function to show the prices on the top of the bars
    return end, plot_values_in_barh(y)


# Function that returns for each category, the brand with the highest price
def brand_with_highest_price_for_category(df_names):
    # Initializing an empty list where we will put our Dataframes later on
    l = []
    # Starting a for loop to read every DataFrame
    for i in range(len(df_names)):
        # Selecting the columns to use for this task
        data = pd.read_csv(df_names[i], usecols=['category_code', 'brand', 'price'])
        # For every category_code and brand, calculating the average price of the products
        a = data.groupby(['category_code', 'brand']).mean()
        # Selecting the rows with the higher average price for each category
        a1 = a.loc[a.groupby(level='category_code')['price'].idxmax()]
        # Appending the analyzed DataFrame for 1 month to the list l
        l.append(a1)
    # Concatenating every DataFrame of each month grouped by category_code and brand in one DataFrame that will not
    # be memory expensive
    final = pd.concat(l)
    # Resetting the index because i do not want to work with MultiIndex
    rfinal = final.reset_index()
    # Selecting again only the rows with the higher average price for category after concatenating the DataFrames
    last_final = rfinal.loc[rfinal.groupby('category_code')['price'].idxmax()]
    # Return the output
    return last_final.sort_values(by=['price'])


# RQ4 functions

# Function that is used to see if the prices of different brands are significantly different
def average_price_per_brand(df_names):
    # Initializing an empty list
    l = []
    # Starting the loop to read the dataframes of every month
    for i in range(len(df_names)):
        # Selecting just the columns referring to the brand and price
        data = pd.read_csv(df_names[i], usecols=['brand', 'price'])
        # Grouping by brand and calculating the average price per brand
        a = data.groupby('brand').mean()
        # Appending the obtained DataFrame regarding the results of one month in the starting empty list
        l.append(a)
    # Concatenating every DataFrame of each month in one DataFrame that will not be memory expensive
    t = pd.concat(l)
    # Resetting the index because i do not want to work with MultiIndex
    rt = t.reset_index()
    # Grouping by brand the full DataFrame regarding all months and calculating the mean price
    u = rt.groupby('brand').mean()
    # Returning the Dataframe, the minimum and the maximum to compare the results
    return u, u.min(), u.max()


# Function that is used to reduce the number of data we want to analyze for the RQ4
def make_df_purchase(df_names, months):
    df_purchase = {}
    # Reading the data of all months and selecting only purchase events from the DataFrame
    for i in range(len(df_names)):
        data = pd.read_csv(df_names[i], usecols=['brand', 'price', 'event_type'])
        df_purchase[months[i]] = data[data['event_type'] == 'purchase']
    # Appending the results of every months to a dictionary
    return df_purchase


# Function that returns the profit of every brand in each month
def earning_per_month(df_purchase, months):
    dict_earning = {}
    # Calculating the earning per month of each brand grouping by brand and doing the sum of the prices of every sold
    # product
    for i in range(len(df_purchase)):
        data = df_purchase[months[i]]
        dict_earning[months[i]] = data.groupby('brand', as_index=False).sum()
    return dict_earning


# Function that given a brand in input, returns the total profit for month of that brand
def brand_per_month(brand, dict_earning, months):
    df_profit = {}
    # For every month selecting the profit from the dictionary of earnings created before. If there is no profit for the
    # selected brand, we set it equal to 0
    for i in range(len(months)):
        try:
            df_profit[months[i]] = dict_earning[months[i]].loc[dict_earning[months[i]].brand == brand, 'price'].values[
                0]
        except IndexError:
            df_profit[months[i]] = 0
    return df_profit


# Function that given the earnings of every brand, returns the top 3 brands that have suffered the biggest losses
# between one month and the previous one
def find_3_worst_brand(dict_earning, months):
    # Selecting the dictionary obtained from the total profits of the brands and then merging them in one DataFrame
    # where on the columns we have the months and on the rows we have the brands. The values are the earnings of each
    # brand for every month
    data_frames = [dict_earning[months[i]] for i in range(len(dict_earning))]

    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['brand'],
                                                    how='outer'), data_frames)

    df_merged.set_index('brand', inplace=True)
    df_merged.set_axis(months, axis=1, inplace=True)

    # Transposing the DataFrame and applying the pct_change to calculate the percentage change between every month
    # and the month before
    df_pct = df_merged.T.pct_change()

    worst_brand = []
    worst_value = []
    worst_months = []
    # Selecting the minimum of the percentage change(which means the bigger loss) in our DataFrame, the brand that
    # corresponds to it and the month that refers to it. We append those values to the lists we defined before
    for i in range(0, 3):
        worst_brand.append(df_pct.min().sort_values().index[i])
        worst_value.append(round(abs(df_pct.min().sort_values()[i]) * 100, 2))

        L = list(df_pct[df_pct[worst_brand[i]] == df_pct.min().sort_values()[i]].index.values)
        worst_months.append(''.join(L))

    # Showing the result of the request
    for j in range(0, 3):
        print('{} lost {}% bewteen {} and the month before'.format(worst_brand[j], worst_value[j], worst_months[j]),
              end=' \n')
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
        df=pd.read_csv(df_names[i],parse_dates=['event_time'],date_parser=pd.to_datetime,usecols=['event_time','user_id'])
        #hourly averege of visitors for each day
        domenica=df[df.event_time.dt.dayofweek==0].groupby(df.event_time.dt.hour).user_id.count()
        lunedi=df[df.event_time.dt.dayofweek==1].groupby(df.event_time.dt.hour).user_id.count()
        martedi=df[df.event_time.dt.dayofweek==2].groupby(df.event_time.dt.hour).user_id.count()
        mercoledi=df[df.event_time.dt.dayofweek==3].groupby(df.event_time.dt.hour).user_id.count()
        giovedi=df[df.event_time.dt.dayofweek==4].groupby(df.event_time.dt.hour).user_id.count()
        venerdi=df[df.event_time.dt.dayofweek==5].groupby(df.event_time.dt.hour).user_id.count()
        sabato=df[df.event_time.dt.dayofweek==6].groupby(df.event_time.dt.hour).user_id.count()

        plt.figure(figsize=[10.0,5.0])
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
        plt.legend('upper-left')
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
        dataset=pd.read_csv(df_names[i],usecols=['event_type','category_code'])
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
        dataset=pd.read_csv(df_names[i],usecols=['user_id','event_type','price'])
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