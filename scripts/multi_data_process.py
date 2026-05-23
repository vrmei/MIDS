import pandas as pd
import numpy as np
from tqdm import *

def preprocessdata(datacsv):
    df = datacsv
    columns = ['timestamp','code','size','col1','col2','col3','col4','col5','col6','col7','col8','RorL']
    df = df.set_axis(columns,axis='columns')
    df = df.drop('timestamp',axis=1)
    needdrop = []
    df = df[df['size'] == 8] 
    print(df)
    for i in range(1, 11):
        try:
            df[columns[i]] = df[columns[i]].str[::].apply(int, base = 16)
        except:
            print(df[columns[i]])
    df = df.drop('size',axis=1)
    df = df.drop('RorL',axis=1)