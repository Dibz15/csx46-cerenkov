import ensembleNN2
import trainAndTestModels
import pandas

from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

df = pandas.read_csv('osu18_cerenkov_feat_mat.tsv', sep='\t')

testRuns = []

nreps = 5

auc, acc = trainAndTestModels.xgBoost(df, test_split=0.2, sampler=None, nreps=nreps)
testRuns.append({'name' : 'xgboost_8020', 'accuracy': acc, 'AUROC': auc})

auc, acc = trainAndTestModels.xgBoost(df, test_split=0.2, sampler=RandomUnderSampler(), nreps=nreps)
testRuns.append({'name' : 'xgboost_8020_eqClasses', 'accuracy': acc, 'AUROC': auc})

k=10

auc, acc = trainAndTestModels.xgBoostKFold(df, test_split=0.0, sampler=None, k=k, nreps=nreps)
testRuns.append({'name' : 'xgboost_{}Fold'.format(k), 'accuracy': acc, 'AUROC': auc})

auc, acc = trainAndTestModels.xgBoostKFold(df, test_split=0.0, sampler=RandomUnderSampler(), k=k, nreps=nreps)
testRuns.append({'name' : 'xgboost_{}Fold_eqClasses'.format(k), 'accuracy': acc, 'AUROC': auc})

auc, acc = trainAndTestModels.xgBoostKFold(df, test_split=0.2, sampler=None, k=k, nreps=nreps)
testRuns.append({'name' : 'xgboost_{}Fold_8020Ensemble'.format(k), 'accuracy': acc, 'AUROC': auc})

auc, acc = trainAndTestModels.xgBoostKFold(df, test_split=0.2, sampler=RandomUnderSampler(), k=k, nreps=nreps)
testRuns.append({'name' : 'xgboost_{}Fold_8020Ensemble_eqClasses'.format(k), 'accuracy': acc, 'AUROC': auc})

auc, acc = ensembleNN2.feedforward(df, test_split=0.2, sampler=None, batchSize=150, num_models=nreps)
testRuns.append({'name' : 'NN_8020Ensemble', 'accuracy': acc, 'AUROC': auc})

batchSize = 150

auc, acc = ensembleNN2.feedforward(df, test_split=0.2, sampler=RandomUnderSampler(), batchSize=batchSize, num_models=nreps)
testRuns.append({'name' : 'NN_8020Ensemble_eqClasses', 'accuracy': acc, 'AUROC': auc})

auc, acc = ensembleNN2.feedforwardKFold(df, test_size=0.2, sampler=None, batchSize=150, k=k, nreps=nreps)
testRuns.append({'name' : 'NN_{}Fold_8020Ensemble'.format(k), 'accuracy': acc, 'AUROC': auc})

auc, acc = ensembleNN2.feedforwardKFold(df, test_size=0.2, sampler=RandomUnderSampler(), batchSize=batchSize, k=k, nreps=nreps)
testRuns.append({'name' : 'NN_{}Fold_8020Ensemble_eqClasses'.format(k), 'accuracy': acc, 'AUROC': auc})

with open('testLog.csv', 'w') as wf:
  wf.write('model,accuracy,AUROC\n')
  for run in testRuns:
    wf.write(str(run['name']) + ',' + str(run['accuracy']) + ',' + str(run['AUROC']) + '\n')
