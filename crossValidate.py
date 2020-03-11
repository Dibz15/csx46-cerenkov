# rmf 3.9.2020, last modified 3.11.2020

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
def splitData(k, n, groups, data):
    folds = [] # folds will be a list of k dataframes
    # make a copy of the groups dict
    groupsCopy = groups.copy()
    for i in range(0,k):
#        print(len(groupsCopy.keys()))  # to check that we start with all groups for each rep
        # get group IDs
        groupIDs = random.sample(list(groupsCopy.keys()), k=n)  # sample without replacement
        # retrieve actual data corresponding to IDs
        SNPlist = []
        for ID in groupIDs:
            SNPs = groupsCopy[ID]
            SNPlist.extend(SNPs)
        # avoid key error
        check = data['name'].isin(SNPlist)
        SNPfeatures = data[check.values]
        # remove used groupIDs from groups dict
        for ID in groupIDs:
            if ID in groupsCopy:
                del groupsCopy[ID]
        folds.append(SNPfeatures)
    return folds

def balanceFoldSizes(folds):
    foldList = []
    minFold = min(len(featureDF) for featureDF in folds) # get fold with fewest SNPs
#    print(str(minFold) + ' minimum fold') # check size of minimum fold size
    # reduce number of all other folds
    for featureDF in folds:
        if len(featureDF) > minFold:
            toDrop = len(featureDF) - minFold  # number of cSNPs to remove
            # remove random cSNPs; 0 is the negative label; .index gets row indices as list-like object, req. input for .drop()
            featureDF = featureDF.drop(featureDF.query('label == 0').sample(n=toDrop).index)
        foldList.append(featureDF)
    return foldList

def sliceLabels(folds):
    foldList, labelList = [],[]
    # save labels
    for SNPfeatures in foldList:
        labels = SNPfeatures['label'].to_numpy()
        SNPfeatures.drop(['name','label'],axis='columns',inplace=True)
        SNPfeatures = SNPfeatures.to_numpy()
        foldList.append(SNPfeatures)
        labelList.append(labels)
    return foldList, labelList
    
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
    folds = splitData(k,n,groups,dataDF) # split data into k groups
    balancedFolds = balanceFoldSizes(folds) # make sure we have same number of SNPs per fold
    foldList, labelList = sliceLabels(balancedFolds) # remove labels from feature data and save separately
    runCV(foldList,labelList)  # run k-fold cross validation
