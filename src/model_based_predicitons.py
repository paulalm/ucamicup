#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 15:56:23 2018

@author: paula
"""

import pandas as pd
import auxiliary_functions as af


#days = ['2017-10-31', '2017-11-02', '2017-11-03', '2017-11-08','2017-11-10','2017-11-15','2017-11-20']
days = ['2017-11-09','2017-11-13','2017-11-21']
#days=['2017-11-15']
times = ['A','B','C']
#Read training files
activities = None

for day in days:
    for time in times:
        filename = '../data/model/'+day+'-'+time+'-result.txt'
        model_predictions = pd.read_csv(filename, sep=';',header=None)
        model_predictions.columns = ['DATE_BEGIN','DATE_END','LABEL','HABITANT']
        model_predictions.DATE_BEGIN = pd.to_datetime(model_predictions['DATE_BEGIN'],format='%Y/%m/%d %H:%M:%S')
        model_predictions.DATE_END = pd.to_datetime(model_predictions['DATE_END'],format='%Y/%m/%d %H:%M:%S')
        
        filename = '../data/Test/'+day+'/'+day+'-'+time+'/'+day+'-'+time+'-activity.csv'
        activities = af.read_labels(day, time)
        labels = []
        predictions = pd.DataFrame(columns=['TIMESTAMP', 'TIME_ID', 'LABEL'])
        for begin in activities.TIMESTAMP:
            end = begin + pd.Timedelta(seconds=30)
            #(StartA <= EndB) and (EndA >= StartB)
            in_interval = model_predictions[model_predictions.apply(lambda x:x.DATE_BEGIN < end and x.DATE_END > begin, axis=1)]
            labels.append("".join([ch for ch in str(in_interval.LABEL.values) if ch not in " \"[]'"]))
        predictions['TIMESTAMP']=activities['TIMESTAMP']
        predictions['TIME_ID']=activities['TIME_ID']
        predictions['LABEL']=labels
        filename = '../data/model/'+day+'-'+time+'-model.csv'
        predictions.to_csv(path_or_buf=filename, index=False)
        print('Done' , day, time)