
"""
Authors: Austin Dibble, Michael Lee (dibblea and leem2)
Class: CS434
Date: 5/28/2019
Filename: performance.py
Info: This has some helper functions for monitoring classifier performance. 
"""

import numpy as np
from sklearn import metrics

def printStats(expected, predicted):
  print('=== Performance Stats ===\n')
  print('Classification Report:')
  print(getClassificationReport(expected, predicted))
  print('\nSensitivity (ability to correctly predict true): {}'.format(getSensitivity(expected, predicted)))
  print('Specificity (ability to correctly predict false): {}'.format(getSpecificity(expected, predicted)))
  print('Informedness (probability of informed decision): {}'.format(getInformedness(expected, predicted)))
  print('Accuracy: {}'.format(getAccuracy(expected, predicted)))
  print('ROC AUC: {}'.format(getAUC(expected, predicted)))

def getClassificationReport(expected, predicted):
  return metrics.classification_report(expected, predicted)

def getConfusionMatrix(expected, predicted):
  return metrics.confusion_matrix(expected, predicted)

def getAccuracy(expected, predicted):
  return metrics.accuracy_score(expected, predicted)

def getAUC(expected, predicted):
  return metrics.roc_auc_score(expected, predicted)

#Ability to correctly predict true
def getSensitivity(expected, predicted):
  confusionMatrix = getConfusionMatrix(expected, predicted)
  tp = confusionMatrix[1][1]
  fn = confusionMatrix[1][0]

  return float(tp) / float(tp + fn)

#ability to correctly predict false
def getSpecificity(expected, predicted):
  confusionMatrix = getConfusionMatrix(expected, predicted)
  tn = confusionMatrix[0][0]
  fp = confusionMatrix[0][1]

  return float(tn) /  float(tn + fp)

#Youden's Index
def getInformedness(expected, predicted):
  sensitivity = getSensitivity(expected, predicted)
  specificity = getSpecificity(expected, predicted)
  return sensitivity + specificity - 1.0