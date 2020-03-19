from __future__ import print_function
import argparse
from math import log10

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

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Training settings
parser = argparse.ArgumentParser(description='Super Resolution')
parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.1, help='Learning Rate. Default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--checkpoint_dir', type=str, default=".", help="directory to save checkpoints into")
parser.add_argument('--log', action='store_true', help="log training stats?")
parser.add_argument('--log_file', type=str, default="log.csv", help="csv file to log training stats into")
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

print('==> Loading datasets')

df = pandas.read_csv('osu18_cerenkov_feat_mat.tsv', sep='\t')
df.head()

names = df['name']
del df['name']

df = df.astype(float)

labels = df['label']
del df['label']

df = df.fillna(0)
labels = labels.fillna(0)
#labels = labels.astype(int)

X = df
y = labels

num_features = X.shape[1]
#print('Num Features:', num_features)

print('==> Splitting Data Set into Train/Test sets')

X = np.array(X)
y = np.array(y)
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)

print('==> Applying sampling strategy')

#smt = None
#smt = SMOTETomek(sampling_strategy='auto')
#smt = RandomUnderSampler(sampling_strategy='auto')
smt = TomekLinks(sampling_strategy='auto')
#smt = ClusterCentroids(sampling_strategy='auto')
#smt = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=9)
#smote = SMOTE(sampling_strategy='auto', k_neighbors=3)
#smt = SMOTEENN(sampling_strategy='auto', smote=None, enn=None)

if smt is not None:
    X_smt, y_smt = smt.fit_resample(trainX, trainY)
else:
    X_smt, y_smt = trainX, trainY

print(Counter(y_smt))

train_data = []
for i in range(len(X_smt)):
   train_data.append([X_smt[i], y_smt[i]])

test_data = []
for i in range(len(testX)):
    test_data.append([testX[i], testY[i]])       

training_data_loader = DataLoader(dataset=train_data, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_data, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
#model = SRCNN(upscale_factor=opt.upscale_factor).to(device)
#model = Net(upscale_factor=opt.upscale_factor).to(device)
#model = Full(upscale_factor=opt.upscale_factor).to(device)
#model = UpConv(upscale_factor=opt.upscale_factor).to(device)
model = Feedforward(num_features).to(device)


criterion = nn.BCELoss()
#criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def train(epoch):
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

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

    return epoch_loss / len(training_data_loader)

def test():
    avg_mse = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            inputs, target = batch[0].to(device), batch[1].to(device)

            target = target.float()

            prediction = model(inputs)
            mse = criterion(prediction, target)
            #psnr = 10 * log10(1 / mse.item())
            avg_mse += mse
    print("===> Avg. MSE: {:.4f} dB".format(avg_mse / len(testing_data_loader)))

    return avg_mse / len(testing_data_loader)


def checkpoint(epoch):
    model_out_path = "{}/model_epoch_{}.pth".format(opt.checkpoint_dir, epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

if opt.log:
    with open(opt.log_file, 'w', newline='') as csvfile:
        logWriter = csv.writer(csvfile)
        logWriter.writerow(['Batch Size', 'LR', 'nEpochs'])
        logWriter.writerow([str(opt.batchSize), str(opt.lr), str(opt.nEpochs)])

        logWriter.writerow(['Epoch', 'Avg. MSE', 'Avg. PSNR'])

        for epoch in range(1, opt.nEpochs + 1):
            mseLoss = train(epoch)
            psnrLoss = test()
            #checkpoint(epoch)
            logWriter.writerow([str(epoch), str(mseLoss), str(psnrLoss)])
        checkpoint(opt.nEpochs)
else:
    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        test()
            #checkpoint(epoch)
    #checkpoint(opt.nEpochs)

print('Checking model stats')

model.eval()
model.to(torch.device("cpu"))

print(torch.from_numpy(testX).shape)
preds = model(torch.from_numpy(testX))
print(preds.shape)
preds = preds.detach().numpy()
preds = [1 if val >= 0.4 else 0 for val in preds]

performance.printStats(testY, preds)
