import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

import seaborn as sns
import copy
import os

import datetime
import time



# Function that provide some information about the cvs files
def infos(df_names, months):
    
    for i in range(len(df_names)):
        
        df = pd.read_csv(df_names[i], index_col=0,  parse_dates=True, nrows = 100)


        print('Month %s' %months[i])

        df.head()
        
