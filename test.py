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
import models


df = pandas.read_csv('osu18_cerenkov_feat_mat.tsv', sep='\t')

smt = None
#smt = SMOTETomek(sampling_strategy='auto')
#smt = RandomUnderSampler(sampling_strategy='auto')
#smt = TomekLinks(sampling_strategy='auto')
#smt = ClusterCentroids(sampling_strategy='auto')
#enn = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=7)
#smote = SMOTE(sampling_strategy='auto', k_neighbors=3)
#smt = SMOTEENN(sampling_strategy='auto', smote=smote, enn=None)

testDf, trainDf = models.testTrainSplitDataframe(df, test_size=0.2)
trainX, trainY = models.dataFrameCleanGetLabels(trainDf)
testX, testY = models.dataFrameCleanGetLabels(testDf)

print(Counter(trainY))

X_smt, y_smt = models.resampleData(trainX, trainY, resampler=smt)

print(Counter(y_smt))

# class_balance = len(y) / sum(y) - 1  # n_negative / n_positive

param_dist = dict(max_depth=7,
            learning_rate=0.1,
            n_estimators=40,
            gamma=10,
            scale_pos_weight=1,
            base_score=sum(y_smt) / len(y_smt),
            subsample=1,
            #colsample_bytree=0.3,
            objective= 'binary:logistic' )

#param_dist = { 'objective':'binary:logistic', 'n_estimators': 2 }

_, _, clf = models.trainAndTestXGBoost(X_smt, y_smt, testX, testY, param_dist, verbose=True)

#cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
#cross_val_score(clf, X, y, cv=cv)

evals_result = clf.evals_result()

import performance

num_round=25
preds = clf.predict(testX)
performance.printStats(testY, preds)