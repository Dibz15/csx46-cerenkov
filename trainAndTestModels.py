import pandas
import xgboost as xgb
from xgboost import plot_importance
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE

from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

import crossValidate
import models
import performance

import matplotlib.pyplot as plt

def xgBoost(df, test_split=0.2, sampler=None, param_dist=None, nreps=1, pca=None, plotImportance=False):
  smt = sampler

  avg_auc = 0
  avg_acc = 0
  clf = None
  for rep in range(nreps):

    testDf, trainDf = models.testTrainSplitDataframe(df, test_size=test_split)
    trainX, trainY = models.dataFrameCleanGetLabels(trainDf)
    X_smt, y_smt = models.resampleData(trainX, trainY, resampler=smt)

    if param_dist is None:
      param_dist = dict(max_depth=7,
              learning_rate=0.1,
              n_estimators=40,
              gamma=10,
              scale_pos_weight=1,
              base_score=sum(y_smt) / len(y_smt),
              subsample=1,
              #colsample_bytree=0.3,
              objective= 'binary:logistic' )

    print('XGBoost training class distribution:',Counter(y_smt))

    # class_balance = len(y) / sum(y) - 1  # n_negative / n_positive
    #param_dist = { 'objective':'binary:logistic', 'n_estimators': 2 }

    testX, testY = models.dataFrameCleanGetLabels(testDf)

    if pca:
      X_smt = pca.transform(X_smt)
      testX = pca.transform(testX)

    auc, acc, clf = models.trainAndTestXGBoost(X_smt, y_smt, testX, testY, param_dist, verbose=True)
    avg_auc += auc
    avg_acc += acc

  avg_auc /= nreps
  avg_acc /= nreps

  if plotImportance:
    plot_importance(clf)
    plt.show()
  
  return avg_auc, avg_acc

def xgBoostKFold(df, test_split=0.2, sampler=None, param_dist=None, k=10, nreps=1):

  if test_split == 0.0:
    testDf = None
    trainDf = df
  else:
    testDf, trainDf = models.testTrainSplitDataframe(df, test_size=test_split)

  trainDf, labels = models.dataFrameGetLabels(trainDf, labelType=int)
  trainDf, labels = models.resampleData(trainDf, labels, resampler=sampler)

  print('XGBoost k-fold training class distribution:',Counter(labels))

  k = 5
  foldsCollection = crossValidate.getKfolds(trainDf, 'osu18_groups.tsv', k, nreps=nreps)

  modelList = []

  avg_auc = 0
  avg_accuracy = 0

  for rep, foldGroup in enumerate(foldsCollection):
    foldList = foldGroup[0]
    labelList = foldGroup[1]
    
    print('Rep {}'.format(rep))

    fold_auc = 0
    fold_accuracy = 0
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

      if param_dist is None:
        param_dist = dict(max_depth=7,
                learning_rate=0.1,
                n_estimators=40,
                gamma=10,
                scale_pos_weight=1,
                base_score=rare_event_rate,
                subsample=1,
                #colsample_bytree=0.3,
                objective= 'binary:logistic' )

      #param_dist = { 'objective':'binary:logistic', 'n_estimators': 2 }

      curr_auc, curr_accuracy, clf = models.trainAndTestXGBoost(X_smt, y_smt, testX, testY, param_dist)

      print('Current fold AUC: {}'.format(curr_auc))
      print('Current fold accuracy: {}'.format(curr_accuracy))
      fold_auc += curr_auc
      fold_accuracy += curr_accuracy
      
      modelList.append(clf)
        
    fold_auc /= k
    fold_accuracy /= k
    print('Average K-Fold AUC for all folds: {}'.format(fold_auc))
  
    avg_auc += fold_auc
    avg_accuracy += fold_accuracy

  avg_auc /= nreps
  avg_accuracy /= nreps

  if testDf is not None:
    testX, testY = models.dataFrameCleanGetLabels(testDf)

    testX = np.array(testX)
    testY = np.array(testY)

    ensembledPredictions = models.getEnsemblePredictionsXGBoost(modelList, testX)

    print('Final Ensemble Predictions')
    ens_auc = performance.getAUC(testY, ensembledPredictions)
    ens_acc = performance.getAccuracy(testY, ensembledPredictions)
    return ens_auc, ens_acc
  else:
    return avg_auc, avg_accuracy

def runTSNE():
  df = pandas.read_csv('osu18_cerenkov_feat_mat.tsv', sep='\t')
  #trainDf, labels = models.dataFrameGetLabels(df, labelType=float)
  trainX, trainY = models.dataFrameCleanGetLabels(df, labelType=float)

  scaler = StandardScaler()

  trainX = scaler.fit_transform(trainX)

  pca = PCA(n_components=40)
  pca.fit_transform(trainX)

  pca_variance = pca.explained_variance_ratio_

  pca_variance = [v for v in pca_variance if v > 1e-3]

  print(pca_variance)

  """
  plt.figure(figsize=(8, 6))
  plt.bar(range(len(pca_variance)), pca_variance, alpha=0.5, align='center', label='individual variance')
  plt.legend()
  plt.ylabel('Variance ratio')
  plt.xlabel('Principal components')
  plt.show()
  """

  df = pandas.read_csv('osu18_cerenkov_feat_mat.tsv', sep='\t')
  #trainDf, labels = models.dataFrameGetLabels(df, labelType=float)
  trainX, trainY = models.dataFrameCleanGetLabels(df, labelType=float)
  trainX, trainY = models.resampleData(trainX, trainY)

  scaler = StandardScaler()

  trainX = scaler.fit_transform(trainX)
  x_transformed = pca.transform(trainX)

  tsne = TSNE(n_components=2)

  x_embedded = tsne.fit_transform(x_transformed)

  x_plt, y_plt = zip(*x_embedded)

  plt.figure(figsize=(8,6))
  plt.scatter(x_plt, y_plt, s=[0.1, 0.2], c = trainY, cmap = mpl.colors.ListedColormap([[0.1, 0.1, 0.1, 0.2], [1.0, 0, 0, 0.9]]))
  plt.show()

def plotImportance():
  df = pandas.read_csv('osu18_cerenkov_feat_mat.tsv', sep='\t')
  auc, acc = xgBoost(df, test_split=0.2, sampler=None, nreps=5, plotImportance=True)


plotImportance()
