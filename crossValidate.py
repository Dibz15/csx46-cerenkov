# rmf 3.9.2020, last modified 3.10.2020

import sys, random
import pandas
import numpy as np
from sklearn.model_selection import cross_validate
import xgboost as xgb
# import code to train/test somehow?

# USAGE
usage = 'python ' + sys.argv[0] + ' <data file> <groups file>'
if len(sys.argv) != 3 or '-h' in sys.argv or '--help' in sys.argv:
    print(usage)
    sys.exit()

# SUBROUTINES
def splitData(n, groups, data):
    # get group IDs
    groupIDs = random.sample(list(groups.keys()), k=n)  # sample without replacement
    # retrieve actual data corresponding to IDs
    SNPlist = []
    for ID in groupIDs:
        SNPs = groups[ID]
        SNPlist.extend(SNPs)
    # avoid key error
    check = data['name'].isin(SNPlist)
    SNPfeatures = data[check.values]
    # save labels
    labels = SNPfeatures['label'].to_numpy()
    SNPfeatures.drop(['name','label'],axis='columns',inplace=True)
    SNPfeatures = SNPfeatures.to_numpy()
    return SNPfeatures, labels

def runCV(folds,labels):
    for i in range(0,len(folds)):
        trainMatrix = folds[i]
        trainLabels = labels[i]
        validateMatrix = getRemainder(folds,i)
        validateLabels = getRemainder(labels,i)
        # run classifier
        # save output
        
def getRemainder(data,i):
    remainder = [j for j in data if data.index(j) != i]
    remainderList = []
    for x in remainder:
        remainderList.extend(x)
    return remainderList
    
# ARGUMENTS and MAIN
dataFile = sys.argv[1]
groupsFile = sys.argv[2]

# read data file
dataDF = pandas.read_csv(dataFile, sep='\t')

# read groups file
groups,sizes = {},{}
with open(groupsFile,'r') as groupsFile:
    next(groupsFile) # skip header
    for line in groupsFile:
        chrom, start, stop, SNPname, groupID, groupSize = line.strip().split('\t')
        # save SNP IDs for each group
        if groupID not in groups:
            groups[groupID] = []
        groups[groupID].append(SNPname)
        # save size of each group
        if groupID not in sizes:
            sizes[groupID] = groupSize
groupsFile.close()

# split data
k = 10
nreps = 5
n = round(len(groups)/k)  # number of groups per fold
for i in range(0,nreps):
    print(i)
    folds, labelList = [],[] # folds will be a list of k numpy arrays
    for i in range(0,k):
        SNPfeatures, labels = splitData(n,groups,dataDF) # split data into k groups
        folds.append(SNPfeatures)
        labelList.append(labels)
    runCV(folds,labelList)  # run k-fold cross validation
