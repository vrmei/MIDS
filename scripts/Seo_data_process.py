import pandas as pd
import numpy as np
from tqdm import *

def preprocessdata(datacsv):
    datacsv
df = pd.read_csv('data/CAN-intrusion-dataset/gear_dataset.csv')
print(df)
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

dfdata = df.loc[:, 'code':'col8']
dflabel = df.loc[:, 'RorL']
newdata = None
templist =[]
first = True
for i in tqdm(range(len(dfdata))):
    if i%9 != 0 or i == 0:
        templist.append(dfdata.iloc[i,:].values)
    else:
        non = templist[0]
        for arr in templist[1:]:
            non = np.concatenate((non, arr))
        if first:
            newdata = non
            first = False
        else:
            newdata = np.vstack((newdata, non))
        templist = []
        templist.append(dfdata.iloc[i,:].values)
        
first = True
newlabel = []
labellist = []
for i in tqdm(range(len(dflabel))):
    if i%9 != 0 or i == 0:
        if dflabel.iloc[i] == 'T':
            labellist.append(1)
        else:
            labellist.append(0)
    else:
        if 1 in labellist:
            templabel = 1
        else:
            templabel = 0
        labellist = []
        newlabel.append(templabel)


dfdata = pd.DataFrame(newdata)
print(dfdata)

c = {"Label" : newlabel}
dflabel = pd.DataFrame(c)
print(dflabel)
df = df.drop('RorL',axis=1)
print(df.info())
dfdata.to_csv('gear_data.csv', index=False)
dflabel.to_csv('gear_label.csv', index=False)