#!/usr/bin/env python
# coding: utf-8

import pandas
import xgboost as xgb
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

import crossValidate
import performance
import models

df = pandas.read_csv('osu18_cerenkov_feat_mat.tsv', sep='\t')

testDf, trainDf = models.testTrainSplitDataframe(df, test_size=0.2)

trainDf, labels = models.dataFrameGetLabels(trainDf, labelType=int)

smt = RandomUnderSampler(sampling_strategy='auto')
trainDf, labels = models.resampleData(trainDf, labels, resampler=smt)

#trainDf, labels = RandomUnderSampler().fit_resample(trainDf, labels)

print(trainDf.shape)

k = 5
foldsCollection = crossValidate.getKfolds(trainDf, 'osu18_groups.tsv', k, nreps=1)

modelList = []
for rep, foldGroup in enumerate(foldsCollection):
    foldList = foldGroup[0]
    labelList = foldGroup[1]
    
    print('Rep {}'.format(rep))

    avg_auc = 0
    for idx in range(len(foldList)):
        testX = foldList[idx]
        testY = labelList[idx]
        print('\tFold {}'.format(idx))
        print('\t\tTest Size: {}'.format(testX.shape[0]))
        #trainX = crossValidate.getRemainder(foldList, testX)
        #trainY = crossValidate.getRemainder(labelList, testY)
        
        trainX = np.empty(shape=[0, testX.shape[1]])
        trainY = np.empty(shape=[0,])
        
        for j in range(len(foldList)):
            if j != idx:
                trainX = np.concatenate((trainX, foldList[j]), axis=0)
                trainY = np.concatenate((trainY, labelList[j]), axis=0)
        
        X_smt = trainX
        y_smt = trainY
        
        _RANDOM_STATE = 1337
        # class_balance = len(y) / sum(y) - 1  # n_negative / n_positive
        rare_event_rate = sum(y_smt) / len(y_smt)

        param_dist = dict(max_depth=10,
                    learning_rate=0.1,
                    n_estimators=100,
                    gamma=10,
                    scale_pos_weight=1,
                    base_score=rare_event_rate,
                    subsample=1,
                    colsample_bytree=0.3,
                    objective= 'binary:logistic' )

        #param_dist = { 'objective':'binary:logistic', 'n_estimators': 2 }

        curr_auc, curr_accuracy, clf = models.trainAndTestXGBoost(X_smt, y_smt, testX, testY, param_dist)

        print('Current fold AUC: {}'.format(curr_auc))
        print('Current fold accuracy: {}'.format(curr_accuracy))
        avg_auc += curr_auc
        
        modelList.append(clf)
        
    avg_auc /= k
    print('Average K-Fold AUC for all folds: {}'.format(avg_auc))

testX, testY = models.dataFrameCleanGetLabels(testDf)

testX = np.array(testX)
testY = np.array(testY)

ensembledPredictions = models.getEnsemblePredictionsXGBoost(modelList, testX)

print('Final Ensemble Predictions')
performance.printStats(testY, ensembledPredictions)
