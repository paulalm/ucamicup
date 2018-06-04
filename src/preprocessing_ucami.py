#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 19:40:17 2018

@author: paula
"""

from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import auxiliary_functions as af
import numpy as np

def time_to_float(time):
    if time=='A':
        return 0
    elif time == 'B':
        return 1
    else:
        return 2
    
def complete_sensor_data(sensors, filenumber, time):
    #Complete sensor data
    sensors['DAYPART']=time_to_float(time)
    sensors['DATE_ID']=sensors.apply(lambda x:filenumber*10000+x['TIME_ID'], axis=1)
    
    #Change states to binary
    state_dict = {'Movement':1,'Open':1, 'Pressure':1, 'Present':1, 'No movement':0, 'Close':0, 'No Pressure':0, 'No movement':0, 'No present':0}
    sensors['STATE'] = sensors['STATE'].apply(lambda x:state_dict[x])
    return sensors

def complete_proximity_data(proximity, i, time):
    proximity['DATE_ID']=proximity.apply(lambda x:i*10000+x['TIME_ID'], axis=1)
    prox_grouped = proximity.groupby(['DATE_ID','OBJECT'])['RSSI'].mean()>=-97
    prox_grouped = prox_grouped[prox_grouped==True]
    signals=prox_grouped.reset_index()
    signals.drop('RSSI', axis=1, inplace=True)
    return signals

def obtain_proximity_features(signals_complete):
    groupedbydate=signals_complete.groupby('DATE_ID')['OBJECT']
    groupedbydate=groupedbydate.apply(lambda x: x.values)
    binarizer = MultiLabelBinarizer()
    data=binarizer.fit_transform(groupedbydate) 
    proxto= pd.DataFrame(data, columns=list(binarizer.classes_), index = groupedbydate.index)
    proxto.reset_index(inplace=True)
    return proxto

def obtain_features(signals_complete, sensors_complete, labels):
    proxto = obtain_proximity_features(signals_complete)
    features = pd.pivot_table(sensors_complete, values='STATE', index=['DATE_ID'],columns=['OBJECT'], aggfunc=len, fill_value=0)
    
    features.reset_index(inplace=True)
    features = labels.merge(features, on='DATE_ID', how='left')
    features.fillna(0, inplace=True)
    features.sort_values(by='DATE_ID', inplace=True)
    last_v_list = features.apply(af.get_last_state, args=(sensors_complete, ), axis=1)
    last_values = pd.DataFrame(list(last_v_list))
    features = features.merge(last_values, on='DATE_ID', how='left')
    
    features = features.merge(proxto, on='DATE_ID', how='left')
    
    
    #Drop columns not used
    cols = []
    if {'DATE_ID'}.issubset(features.columns):
        cols.append('DATE_ID')
    if {'LABEL_STATE'}.issubset(features.columns):
        cols.append('LABEL_STATE')
    if {'LABEL'}.issubset(features.columns):
        cols.append('LABEL')
    if {'TIMESTAMP'}.issubset(features.columns):
        cols.append('TIMESTAMP')
    if {'TIMESTAMP_STATE'}.issubset(features.columns):
        cols.append('TIMESTAMP_STATE')
    X = features.drop(cols, axis=1)
    X.fillna(0, inplace=True)
    locations, curr_loc, correlations = af.row_difference(X, X.shift())
    X['prev_loc']=locations
    X['curr_loc']=curr_loc
    X['corr_prev']=correlations>0.5
    return X
    
    
    
def get_Y (labels):    
    targets=labels['LABEL'].values
    binarizer = MultiLabelBinarizer()
    data=binarizer.fit_transform(targets) 
    Y = pd.DataFrame(data, columns=list(binarizer.classes_))
    return Y, binarizer.classes_

def mean_dur_priors_and_transisitions(activity_info, priors, classes, total_observations, times):
    activity_labels = activity_info.ACTIVITY.unique()
    total_observations += 1
    prior_add = 1/len(activity_labels)
    
    
    activity_info['segment_duration']=activity_info['duration'].apply(lambda x:x.total_seconds()/30)
    transitions= activity_info.groupby(['prev_act', 'ACTIVITY'])['duration'].count().reset_index()
    mean_dur = activity_info.groupby(['ACTIVITY'])['segment_duration'].mean()
    transition_matrix= np.zeros((len(activity_labels), len(activity_labels)))


    prior_probabilities = []
    for act in classes:
        if(priors.get(act)==None):
            priors[act]=float(prior_add/total_observations)
        else:
            priors[act]=float((priors[act]+prior_add)/total_observations)
        prior_probabilities.append(float(priors[act]))
    
    first_index = -1
    if 'C' in times:
        first_index=np.where(classes=='Act10')[0][0] #from all to enter is 0
    if 'A' in times:
        first_index=np.where(classes=='Act24')[0][0] # wake_up
    if 'B' in times:
        first_index=np.where(classes=='Act10')[0][0] # enter
    
    for act in classes:
        trans = transitions[transitions.prev_act==act]
        suma = trans.duration.sum()+(1)
        i= np.where(classes==act)[0][0]
        for act2 in classes:
          j = np.where(classes==act2)[0][0]  
          transition_matrix[i][j]=(trans[trans.ACTIVITY==act2].duration+(1/len(classes)))/suma if len(trans[trans.ACTIVITY==act2]) else (1/len(classes))/suma
        transition_matrix[i][first_index]=0
    
    if 'A' in times:
        #correct the improbable transitions model night
        #from breakfast to prep breakfast
        i = np.where(classes=='Act05')[0][0]
        j= np.where(classes=='Act02')[0][0]
        transition_matrix[i][j] = 0
        
        #from brush teeth to breakfast
        i = np.where(classes=='Act17')[0][0]
        j= np.where(classes=='Act05')[0][0]
        transition_matrix[i][j] = 0
        
        i = np.where(classes=='Act13')[0][0]
        transition_matrix[i] = prior_probabilities
        
    if 'B' in times:
        #correct the improbable transitions model night
        #from luncg to prep lunch
        i = np.where(classes=='Act06')[0][0]
        j= np.where(classes=='Act03')[0][0]
        transition_matrix[i][j] = 0
        
        #from brush teeth to breakfast
        i = np.where(classes=='Act17')[0][0]
        j= np.where(classes=='Act06')[0][0]
        transition_matrix[i][j] = 0
        
        #from leave only enter
        i = np.where(classes=='Act13')[0][0]
        transition_matrix[i] = prior_probabilities
        
    if 'C' in times:
        #correct the improbable transitions model night
        #from dinner to prep dinner
        i = np.where(classes=='Act07')[0][0]
        j= np.where(classes=='Act04')[0][0]
        transition_matrix[i][j] = 0
    
        #from brush teeth to dinner
        i = np.where(classes=='Act17')[0][0]
        j= np.where(classes=='Act07')[0][0]
        transition_matrix[i][j] = 0
        
        i = np.where(classes=='Act23')[0][0]
        transition_matrix[i] = prior_probabilities
    return mean_dur, prior_probabilities, transition_matrix