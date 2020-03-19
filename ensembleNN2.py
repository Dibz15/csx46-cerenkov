import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Feedforward
import matplotlib.pyplot as plt
import numpy as np
import pandas
import csv
import performance
from collections import Counter

from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

import models
import crossValidate

def train(model, training_data_loader, optimizer, criterion, device, epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        inputs, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        target = target.float()

        outputs = model(inputs)
        #print("Shape: Input - ", input.size(), ", Output - ", outputs.size())

        loss = criterion(outputs, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        #print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    #print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

    return epoch_loss / len(training_data_loader)

def test(model, testing_data_loader, criterion, device):
    avg_mse = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            inputs, target = batch[0].to(device), batch[1].to(device)

            target = target.float()

            prediction = model(inputs)
            mse = criterion(prediction, target)
            #psnr = 10 * log10(1 / mse.item())
            avg_mse += mse
    #print("===> Avg. MSE: {:.4f} dB".format(avg_mse / len(testing_data_loader)))

    return avg_mse / len(testing_data_loader)


def feedforward(df, test_split=0.2, sampler=None, opt=None, batchSize=200, num_models=1):
    if opt is None:
        opt = dict(
            cuda= True,
            batchSize= batchSize, 
            testBatchSize= 80,
            lr= 0.00005,
            nEpochs= 60,
            threads= 4,
            seed= 123,
            checkpoint_dir='.'
        )

    if opt['cuda'] and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without cuda enabled.")

    #torch.manual_seed(opt['seed'])
    device = torch.device("cuda" if opt['cuda'] else "cpu")

    testDf, trainDf = models.testTrainSplitDataframe(df, test_size=0.2)

    trainX, trainY = models.dataFrameCleanGetLabels(trainDf, labelType=float)
    testX, testY = models.dataFrameCleanGetLabels(testDf, labelType=float)

    trainX = np.array(trainX)
    testX = np.array(testX)
    trainY = np.array(trainY)
    testY = np.array(testY)

    num_features = trainX.shape[1]

    modelsList = []
    for idx in range(num_models):
        print('Ensemble Model', idx)
        print('\tResampling model training data')

        trainX, validateX, trainY, validateY = train_test_split(trainX, trainY, test_size=0.2)

        X_smt, y_smt = models.resampleData(trainX, trainY, resampler=sampler)

        train_data = []
        for i in range(len(X_smt)):
            train_data.append([X_smt[i], y_smt[i]])

        test_data = []
        for i in range(len(testX)):
            test_data.append([testX[i], testY[i]])       

        training_data_loader = DataLoader(dataset=train_data, num_workers=opt['threads'], batch_size=opt['batchSize'], shuffle=True)
        testing_data_loader = DataLoader(dataset=test_data, num_workers=opt['threads'], batch_size=opt['testBatchSize'], shuffle=False)

        model = Feedforward(num_features).to(device)

        criterion = nn.BCELoss()
        #criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=opt['lr'])

        print('\tTraining Model...')
        print('\tEpoch ... ', end='')
        for epoch in range(1, opt['nEpochs'] + 1):
            train(model, training_data_loader, optimizer, criterion, device, epoch)
            test(model, testing_data_loader, criterion, device)
            print(' {}'.format(epoch), end='')
        
        #print(' Model trained.')
        #print('\tModel final BCE -- Train: {}, Validate: {}'.format(trainMSE, testMSE))

        #print('\tChecking model against validation data')

        model.eval()
        model.to(torch.device("cpu"))
        preds = model(torch.from_numpy(validateX))
        preds = preds.detach().numpy()
        preds = [1 if val >= 0.5 else 0 for val in preds]

        print('\n\tModel AUC:', performance.getAUC(validateY, preds))

        modelsList.append(model)

    ensembledPredictions = models.getEnsemblePredictionsNN(modelsList, testX)

    auc = performance.getAUC(testY, ensembledPredictions)
    acc = performance.getAccuracy(testY, ensembledPredictions)

    return auc, acc

def feedforwardKFold(df, test_size=0.2, sampler=None, opt=None, batchSize=100, k=5, nreps=1):
    if opt is None:
        opt = dict(
            cuda= True,
            batchSize= batchSize, 
            testBatchSize= 80,
            lr= 0.00005,
            nEpochs= 60,
            threads= 4,
            seed= 123,
            checkpoint_dir='.'
        )

    if opt['cuda'] and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without cuda enabled.")

    #torch.manual_seed(opt['seed'])
    device = torch.device("cuda" if opt['cuda'] else "cpu")

    testDf, trainDf = models.testTrainSplitDataframe(df, test_size=test_size)

    if test_size == 0.0:
        testDf = None
        trainDf = df
    else:
        testDf, trainDf = models.testTrainSplitDataframe(df, test_size=test_size)

    trainDf, labels = models.dataFrameGetLabels(df, labelType=float)
    
    trainDf, labels = models.resampleData(trainDf, labels, resampler=sampler)

    foldsCollection = crossValidate.getKfolds(trainDf, 'osu18_groups.tsv', k, nreps=nreps)

    modelList = []

    avg_auc = 0
    avg_accuracy = 0
    for rep, foldGroup in enumerate(foldsCollection):
        foldList = foldGroup[0]
        labelList = foldGroup[1]
        
        print('Rep {}'.format(rep))

        fold_auc = 0
        fold_acc = 0
        for idx in range(len(foldList)):
            testX = foldList[idx]
            testY = labelList[idx]
            print('\tFold {}'.format(idx))
            print('\t\tTest Size: {}'.format(testX.shape[0]))
            #trainX = crossValidate.getRemainder(foldList, testX)
            #trainY = crossValidate.getRemainder(labelList, testY)
            
            trainX = np.empty(shape=[0, testX.shape[1]])
            trainY = np.empty(shape=[0,])

            num_features = trainX.shape[1]
            
            for j in range(len(foldList)):
                if j != idx:
                    trainX = np.concatenate((trainX, foldList[j]), axis=0)
                    trainY = np.concatenate((trainY, labelList[j]), axis=0)
            
            X_smt = trainX
            y_smt = trainY
            
            train_data = []
            for i in range(len(X_smt)):
                train_data.append([X_smt[i], y_smt[i]])

            test_data = []
            for i in range(len(testX)):
                test_data.append([testX[i], testY[i]])       

            training_data_loader = DataLoader(dataset=train_data, num_workers=opt['threads'], batch_size=opt['batchSize'], shuffle=True)
            testing_data_loader = DataLoader(dataset=test_data, num_workers=opt['threads'], batch_size=opt['testBatchSize'], shuffle=False)

            model = Feedforward(num_features).to(device)

            criterion = nn.BCELoss()
            #criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=opt['lr'])

            print('\tTraining Model...')
            print('\tEpoch ... ', end='')
            for epoch in range(1, opt['nEpochs'] + 1):
                train(model, training_data_loader, optimizer, criterion, device, epoch)
                test(model, testing_data_loader, criterion, device)
                print(' {}'.format(epoch), end='')
            
            #print(' Model trained.')
            #print('\tModel final BCE -- Train: {}, Validate: {}'.format(trainMSE, testMSE))

            #print('\tChecking model against validation data')

            model.eval()
            model.to(torch.device("cpu"))
            preds = model(torch.from_numpy(testX))
            preds = preds.detach().numpy()
            preds = [1 if val >= 0.5 else 0 for val in preds]

            print('\tModel AUC:', performance.getAUC(testY, preds))

            curr_auc = performance.getAUC(testY, preds)
            curr_accuracy = performance.getAccuracy(testY, preds)
            fold_auc += curr_auc
            fold_acc += curr_accuracy
            
            modelList.append(model)

        fold_auc /= k
        fold_acc /= k
        print('Average K-Fold AUC for all folds: {}'.format(fold_auc))

        avg_auc += fold_auc
        avg_accuracy += fold_acc
    
    avg_auc /= nreps
    avg_accuracy /= nreps

    if testDf is not None:
        testX, testY = models.dataFrameCleanGetLabels(testDf)

        testX = np.array(testX)
        testY = np.array(testY)

        ensembledPredictions = models.getEnsemblePredictionsNN(modelList, testX)

        print('Final Ensemble Predictions')
        ens_auc = performance.getAUC(testY, ensembledPredictions)
        ens_acc = performance.getAccuracy(testY, ensembledPredictions)
        return ens_auc, ens_acc
    else:
        return avg_auc, avg_accuracy