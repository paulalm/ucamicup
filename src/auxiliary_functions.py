#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 17:39:58 2018

@author: paula
"""
import numpy as np
import pandas as pd
from ast import literal_eval
import math as m

def read_files(dataset, day, time):
    
    sensor_filename = '../data/'+dataset+'/'+day+'/'+day+'-'+time+'/'+day+'-'+time+'-sensors.csv'
    print(sensor_filename)
    sensors = pd.read_csv(sensor_filename, sep=';',header=0,encoding='latin1')
    sensors['TIMESTAMP'] = pd.to_datetime(sensors['TIMESTAMP'],format='%Y/%m/%d %H:%M:%S')
    sensors['TIME_ID']=sensors.apply(lambda x:m.floor((x['TIMESTAMP'].hour*3600+x['TIMESTAMP'].minute*60+x['TIMESTAMP'].second)/30), axis=1)
    
    
    
    #floorfile = '../data/'+dataset+'/'+day+'/'+day+'-'+time+'/'+day+'-'+time+'-floor.csv'
    #floor = pd.read_csv(floorfile, sep=';',header=0,encoding='latin1')
    
    proximityfile = '../data/'+dataset+'/'+day+'/'+day+'-'+time+'/'+day+'-'+time+'-proximity.csv'
    proximity = pd.read_csv(proximityfile, sep=';',header=0,encoding='latin1')
    proximity = proximity.dropna(axis=0, how='any')
    proximity['TIMESTAMP']= pd.to_datetime(proximity['TIMESTAMP'],format='%Y/%m/%d %H:%M:%S')
    proximity['TIME_ID']=proximity.apply(lambda x:m.floor((x['TIMESTAMP'].hour*3600+x['TIMESTAMP'].minute*60+x['TIMESTAMP'].second)/30), axis=1)

    return sensors, proximity

def read_activities(dataset, day, time):
    actfile = '../data/'+dataset+'/'+day+'/'+day+'-'+time+'/'+day+'-'+time+'-activity.csv'
    activities = pd.read_csv(actfile, sep=';',header=0,encoding='latin1')
    activities.drop(['HABITANT'], inplace=True, axis=1)
    activities['DATE BEGIN']= pd.to_datetime(activities['DATE BEGIN'],format='%Y/%m/%d %H:%M:%S')
    activities['DATE END']= pd.to_datetime(activities['DATE END'],format='%Y/%m/%d %H:%M:%S')
    activities['duration'] = activities['DATE END']-activities['DATE BEGIN']
    activities['prev_act'] = activities.shift()['ACTIVITY']
    activities.drop(['DATE BEGIN','DATE END'], axis=1, inplace=True)
    return activities

def read_labels(day, time):
    labelsfile = '../data/string-labels-'+day+'-'+time+'-activity.csv'
    labelsf = pd.read_csv(labelsfile, header=0, converters={"LABEL": literal_eval})
    labels = pd.DataFrame()
    if {'LABEL'}.issubset(labelsf.columns):
        labels['LABEL']=labelsf['LABEL']
    if {'Timestamp'}.issubset(labelsf.columns):
        labels['TIMESTAMP']= pd.to_datetime(labelsf['Timestamp'],format='%Y/%m/%d %H:%M:%S')
    else:
        labels['TIMESTAMP']= pd.to_datetime(labelsf['TIMESTAMP'],format='%Y/%m/%d %H:%M:%S')
    labels['TIME_ID']=labels.apply(lambda x:m.floor((x['TIMESTAMP'].hour*3600+x['TIMESTAMP'].minute*60+x['TIMESTAMP'].second)/30), axis=1)
    return labels

'''
Calculates whether location changed from the last frame
Return the correlation between frame and last one
'''
def row_difference(r1, r2):
    loc_dict = {'SM1':0,'SM3':1, 'SM4':2, 'SM5':3, 'Other':4}
    cols = []
    if {'SM1'}.issubset(r1.columns):
        cols.append('SM1')
    if {'SM3'}.issubset(r1.columns):
        cols.append('SM3')
    if {'SM4'}.issubset(r1.columns):
        cols.append('SM4')
    loc_cols = r1[cols]
    max_locs = loc_cols.idxmax(axis=1)
    location1 = max_locs.apply(lambda x:loc_dict[x])
    max_locs = r2[cols].idxmax(axis=1)
    max_locs.fillna('Other', inplace=True)
    location2 = max_locs.apply(lambda x:loc_dict[x])
    r2.fillna(0, inplace=True)
    corr = r1.corrwith(r2, axis=1)
    corr.fillna(0, inplace=True)
    location = location1==location2
    return location, location1, corr

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.apply(lambda x:str(x))
    targets = df_mod.unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod.replace(map_to_int)
    return (map_to_int,df_mod, targets)

def get_sensor_label(x, labels):
    lab = labels[labels.TIME_ID==x]['LABEL'].values
    if len(lab)==0:
        labs = []
    else:
        labs = lab
    return labs

def get_last_state(x, sensors):
    columns = x.index
    columns=columns.drop(['SM1','SM3','SM4','SM5','LABEL','TIMESTAMP','DATE_ID'], errors='ignore')
    sol = {}
    sol['DATE_ID']=x.DATE_ID
    for c in columns:
        last_value=sensors[(sensors['OBJECT']==c)&(sensors['DATE_ID']<=x.DATE_ID)].tail(1)['STATE'].values
        if len(last_value)>0:
            sol[c+'_STATE']=last_value[0]
        else:
            sol[c+'_STATE']=0
    
    return sol

def get_model_probabilities(labels, classes):
    m_probs = np.zeros((1, len(classes)))
    
    if isinstance(labels, str):
        size = int(len(labels)/5)
        for i in range(size):
            act = labels[i*5:((i+1)*5)]
            ci = np.where(classes==act)[0][0] if len(np.where(classes==act)) > 0  and len(np.where(classes==act)[0]) > 0 else -1
            if ci > -1:
                m_probs[0,ci] += 1/size
    return m_probs