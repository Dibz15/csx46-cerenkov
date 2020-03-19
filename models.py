import performance
import xgboost as xgb
import pandas as pd
import torch
import numpy as np

def getEnsemblePredictionsNN(modelsList, testX):
  predictions = np.zeros(shape=[testX.shape[0], 1])
  for model in modelsList:
      model.eval()
      model.to(torch.device("cpu"))

      #print(torch.from_numpy(testX).shape)
      preds = model(torch.from_numpy(testX))
      #print(preds.shape)
      preds = preds.detach().numpy()

      #Sum up all predictions for averaging
      predictions += preds

  #Average the predictions
  predictions = predictions / len(modelsList)

  #Force the predictions to a binary value for testing
  predictions = [1 if val >= 0.5 else 0 for val in predictions]
  return predictions

def getEnsemblePredictionsXGBoost(modelsList, testX):
  predictions = np.zeros(shape=[testX.shape[0],])
  for model in modelsList:
      preds = model.predict(testX)
      #Sum up all predictions for averaging
      predictions += preds

  #Average the predictions
  predictions = predictions / len(modelsList)

  #Force the predictions to a binary value for testing
  predictions = [1 if val >= 0.5 else 0 for val in predictions]
  return predictions


def resampleData(X, y, resampler=None):
  if resampler is None:
    return X, y
  
  return resampler.fit_resample(X, y)


def dataFrameCleanGetLabels(df, labelType=int):
  del df['name']
  df = df.fillna(0)
  df = df.astype(float)
  labels = df['label']
  del df['label']
  labels = labels.fillna(0)
  labels = labels.astype(labelType)

  return df, labels

def dataFrameGetLabels(df, labelType=int):
  df = df.drop(df.query('label != 0 & label != 1').index)

  labels = df['label']

  labels = labels.astype(labelType)

  return df, labels


def testTrainSplitDataframe(df, test_size=0.2):

  if test_size != 0.0:
    randomRows = df.sample(frac=test_size).index

    testDf = df.iloc[randomRows,:]
    trainDf = df.drop(randomRows)

    testDf= testDf.drop(testDf.query('label != 0 & label != 1').index)
  else:
    testDf = None
    trainDf = df

  return testDf, trainDf

"""
Trains XGBoost model, given trainX (features) and trainY (labels) and a dictionary of parameters param_dist.
Returns the AUC value of the model tested against the test data, and the model itself.
"""
def trainAndTestXGBoost(trainX, trainY, testX, testY, param_dist, verbose=False):
  X_smt = trainX
  y_smt = trainY

  _RANDOM_STATE = 1337

  #param_dist = { 'objective':'binary:logistic', 'n_estimators': 2 }

  clf = xgb.XGBClassifier(**param_dist, booster='gbtree', n_jobs=-1, random_state=_RANDOM_STATE)

  clf.fit( X_smt, y_smt,
          eval_set=[(X_smt, y_smt), (testX, testY)],
          eval_metric='logloss',
          verbose=False)

  preds = clf.predict(testX)
  curr_auc = performance.getAUC(testY, preds)
  accuracy = performance.getAccuracy(testY, preds)

  return curr_auc, accuracy, clf