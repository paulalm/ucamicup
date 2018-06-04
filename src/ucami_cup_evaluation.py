#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 22:03:32 2018

@author: paula
"""
import auxiliary_functions as af
import preprocessing_ucami as p
import pandas as pd
import numpy as np
import math as m
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#All columns
all_columns  =['BATHROOM TAP', 'BED', 'BOOK', 
               'C01', 'C01_STATE', 'C04', 'C04_STATE', 'C05', 'C05_STATE', 'C07', 'C07_STATE',
       'C08', 'C08_STATE', 'C09', 'C09_STATE', 'C10', 'C10_STATE', 'C12',
       'C12_STATE', 'C13', 'C13_STATE', 'C14', 'C14_STATE', 'D01', 'D01_STATE',
       'D02', 'D02_STATE', 'D03', 'D03_STATE', 'D04', 'D04_STATE', 'D07',
       'D07_STATE', 'D09', 'D09_STATE', 'D10', 'D10_STATE', 'ENTRANCE DOOR',
       'FOOD CUPBOARD', 'FRIDGE', 'GARBAGE CAN', 'H01', 'H01_STATE',
       'LAUNDRY BASKET', 'M01', 'M01_STATE', 'MEDICINE BOX', 'POT DRAWER',
       'PYJAMAS DRAWER', 'S09', 'S09_STATE', 'SM1', 'SM3', 'SM4',  'SM5', 'TV CONTROLLER', 'TV0','TV0_STATE',
       'TOOTHBRUSH', 'WARDROBE DOOR',
       'WATER BOTTLE', 'corr_prev', 'curr_loc', 'prev_loc', 'prev_act']
#Read training files
dataset = 'Training'
train_days = ['2017-10-31', '2017-11-02', '2017-11-03', '2017-11-08','2017-11-10','2017-11-15','2017-11-20']
times = ['A']

sensors_train = pd.DataFrame(columns=['TIMESTAMP', 'DATE_ID','OBJECT','STATE', 'HABITANT', 'DAYPART', 'LABEL'])
labels_train = pd.DataFrame(columns=['DATE_ID', 'LABEL'])
proximity_train = pd.DataFrame(columns=['DATE_ID', 'OBJECT'])
activities_train = pd.DataFrame(columns=['ACTIVITY', 'duration', 'next_act'])

priors = {}

total_observations = 0

i=1
X_train = pd.DataFrame(columns=all_columns)

for day in train_days:
    for time in times:
        labels = af.read_labels(day, time)
        activities = af.read_activities(dataset,day, time)
        sensors, proximity = af.read_files(dataset, day, time)
        #Complete sensor data
        sensors = p.complete_sensor_data(sensors, i, time)
        sensors["LABEL"]=sensors.TIME_ID.apply(af.get_sensor_label, args=(labels,)) #only in training
        sensors_train=pd.concat([sensors_train, sensors])
        
        #Complete proximity data
        signals = p.complete_proximity_data(proximity, i, time)
        proximity_train = pd.concat([proximity_train, signals])
        
        #Complete label data
        labels['DATE_ID']=labels.apply(lambda x:i*10000+x['TIME_ID'], axis=1)
        labels_train=pd.concat([labels_train, labels[['DATE_ID','LABEL']]])
        
        
        #for markov probabilities
        first_activity = activities.head(1)['ACTIVITY'].values[0]
        if(priors.get(first_activity)==None):
            priors[first_activity]=0
        priors[first_activity]+=1
        total_observations +=1
        activities_train = pd.concat([activities_train, activities])
        
        X_train = pd.concat([X_train, p.obtain_features(signals, sensors, labels[['DATE_ID','LABEL']])])
        #increase file number
        i+=1
        
#Preprocessing files
X_train.fillna(0, inplace=True)

Y_train, classes = p.get_Y(labels_train)


#obtain markov model parameters
mean_dur, prior_probabilities, transition_matrix = p.mean_dur_priors_and_transisitions(activities_train, priors, classes, total_observations, times)


#Model creation
layer_size=(300,)
activation = 'logistic'

seed = 7
estimators = []
estimators.append(('standardize', StandardScaler()))     
estimators.append(('clf', MLPClassifier(hidden_layer_sizes=layer_size, activation=activation,max_iter=5000, solver='lbfgs', learning_rate_init =0.0001, random_state=seed, momentum=0.3)))
model = Pipeline(estimators)

#Fit 
model.fit(X_train, Y_train)
        
        


dataset = 'Test'
test_days = ['2017-11-09', '2017-11-13','2017-11-21']


i=0
for day in test_days:
    for time in times:
        X_test = pd.DataFrame(columns=X_train.columns)
        #Read test files
        sensors_test = pd.DataFrame(columns=['TIMESTAMP', 'DATE_ID','OBJECT','STATE', 'HABITANT', 'DAYPART', 'LABEL'])
        labels_test = pd.DataFrame(columns=['DATE_ID', 'LABEL'])
        proximity_test = pd.DataFrame(columns=['DATE_ID', 'OBJECT'])
        #X_test['DATE_ID']=[]
        model_predictions = pd.DataFrame(columns=['DATE_ID','TIMESTAMP'])

        sensors_test, proximity_test = af.read_files(dataset,day, time)
        #Complete sensor data
        sensors_test = pd.concat([sensors_test,p.complete_sensor_data(sensors_test, i, time)])
        proximity_test = pd.concat([proximity_test,p.complete_proximity_data(proximity_test, i, time)])
        
        labels_test = pd.read_csv('../data/string-labels-'+day+'-'+time+'-activity.csv')
        #labels_test = pd.DataFrame(columns=['DATE_ID', 'LABEL'])
        #labels_test = af.read_labels(day, time)
        #Complete label data
        #labels_test['DATE_ID']=labels_test.apply(lambda x:i*10000+x['TIME_ID'], axis=1)
        #labels_test=pd.concat([labels_test, labels[['DATE_ID','TIMESTAMP']]])
        l=labels_test[['DATE_ID', 'TIMESTAMP']]
        X_test = pd.concat([X_test, p.obtain_features(proximity_test, sensors_test, l)])
        
        #Read Model Predictions
        filename = '../data/model/'+day+'-'+time+'-model.csv'
        mpredictions = pd.read_csv(filename)
        mpredictions['DATE_ID'] = mpredictions.apply(lambda x:i*10000+x['TIME_ID'], axis=1)
        model_predictions = pd.concat([model_predictions, mpredictions[['DATE_ID','LABEL']]])
        
        #preprocess
        X_test.fillna(0, inplace=True)
        
        #initial variables for hidden markov model management
        prev_act = -1;
        x=0 #for poisson distribution of duration in a state 
        
        prev_prob = []
        predictions =  np.empty((0,Y_train.shape[1]))
                    
        for t in X_test.iterrows():
            index, instance = t
            instance['prev_act']=prev_act;
            #prediction = model.predict(instance.values.reshape(1, -1)) 
            
            model_labels = model_predictions.loc[index]
            mbased_probs = af.get_model_probabilities(model_labels.LABEL, classes)
            
            #sensor-based probabilities
            probabilities = model.predict_proba(instance.values.reshape(1, -1))
            
            #adjust probabilities using priors and transitions
            #Use also previous probabilities 
            if(prev_act == -1): #use priors
                probabilities = np.multiply(probabilities, prior_probabilities)
            else:
                #probabilidad de seguir en el estado actual 
                beta= mean_dur[classes[prev_act]]
                p_stay = 1/beta*m.exp(-x/beta)
                
                #probabilidad de cambiar 
                p_change = 1-p_stay
                tr_probs = np.multiply(p_change, transition_matrix[prev_act])
                
                #cambiar solo la tr_prob de stay
                tr_probs[prev_act]=p_stay
                probabilities = np.multiply(probabilities, tr_probs)
            if len(prev_prob)>0:
                probabilities = np.multiply(probabilities, prev_prob)
            
            
            probabilities = af.softmax(probabilities)
            prev_prob = probabilities
            
            if mbased_probs.sum()>0:
                probabilites = np.multiply(probabilities, 0.5)
                mbased_probs = np.multiply(mbased_probs, 0.5)
                probabilities = np.add(probabilities, mbased_probs)
                print('Using model probabilities', mbased_probs, probabilities)
            
            prediction = probabilities > 1/len(classes) + 0.000001 #account for float precision
            
            #Do I stay or change 
            if(probabilities.argmax() ==prev_act):
                x+=1
            else:
                x=0
        
            prev_act = probabilities.argmax() if sum((prediction>=1).reshape(Y_train.shape[1],))>0 else -1
            predictions = np.append(predictions, prediction,  axis=0)
                        
        #Write file
        timestamps = pd.Series(labels_test.TIMESTAMP.values)
        Y_predict = pd.DataFrame(predictions, columns = classes)
        Y_predict = Y_predict==1
        Y_predict['time']=timestamps
        Y_predict=Y_predict.loc[:, (Y_predict != 0).any(axis=0)]
        Y_predict.to_csv('results/results-'+day+'-'+time+'.csv')

#aux
def get_sensor_label(x):
    lab = labels[labels.TIME_ID==x]['LABEL'].values
    if len(lab)==0:
        labs = []
    else:
        labs = lab
    return labs