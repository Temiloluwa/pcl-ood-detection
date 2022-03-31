# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import misc


def tpr95(name):
    #calculate the falsepositive error when tpr is 95%
    # calculate baseline
    T = 1
    start = 0
    end = 1
    cifar = np.loadtxt('./ood/odin/softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./ood/odin/softmax_scores/confidence_Base_Out.txt', delimiter=',')
     
	      
    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
    if total == 0.0:
        fprBase = 0
    else:
        fprBase = fpr/total

    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('./ood/odin/softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./ood/odin/softmax_scores/confidence_Our_Out.txt', delimiter=',')

    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    total = 0.0
    fpr = 0.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        if tpr <= 0.9505 and tpr >= 0.9495:
            fpr += error2
            total += 1
    if total == 0.0:
        fprNew = 0
    else:
        fprNew = fpr/total
            
    return fprBase, fprNew

def auroc(name):
    #calculate the AUROC
    # calculate baseline
    T = 1
    cifar = np.loadtxt('./ood/odin/softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./ood/odin/softmax_scores/confidence_Base_Out.txt', delimiter=',')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    start = 0
    end = 1 

    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    
    aurocBase = 0.0
    aurocs = []
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        val = (-fpr+fprTemp)*tpr
        aurocBase += val
        aurocs.append(val)
        fprTemp = fpr
    aurocBase += fpr * tpr
    aurocs.append(fpr * tpr)
    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('./ood/odin/softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./ood/odin/softmax_scores/confidence_Our_Out.txt', delimiter=',')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]

    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
   
    aurocNew = 0.0
    fprTemp = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fpr = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        aurocNew += (-fpr+fprTemp)*tpr
        fprTemp = fpr
    aurocNew += fpr * tpr
    print(f"auroc base {aurocBase}, auroc new {aurocNew}")
    return aurocBase, aurocNew

def auprIn(name):
    #calculate the AUPR
    # calculate baseline
    T = 1
    cifar = np.loadtxt('./ood/odin/softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./ood/odin/softmax_scores/confidence_Base_Out.txt', delimiter=',')
    if name == "CIFAR10": 
        start = 0.1
        end = 1 
    if name == "CIFAR100": 
        start = 0.01
        end = 1    
    gap = (end- start)/100000
    precisionVec = []
    recallVec = []
        #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        precisionVec.append(precision)
        recallVec.append(recall)
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision
    #print(recall, precision)

    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('./ood/odin/softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./ood/odin/softmax_scores/confidence_Our_Out.txt', delimiter=',')
    if name == "CIFAR10": 
        start = 0.1
        end = 0.12 
    if name == "CIFAR100": 
        start = 0.01
        end = 0.0104    
    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    auprNew = 0.0
    recallTemp = 1.0
    for delta in np.arange(start, end, gap):
        tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
        fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
        if tp + fp == 0: continue
        precision = tp / (tp + fp)
        recall = tp
        #precisionVec.append(precision)
        #recallVec.append(recall)
        auprNew += (recallTemp-recall)*precision
        recallTemp = recall
    auprNew += recall * precision
    return auprBase, auprNew

def auprOut(name):
    #calculate the AUPR
    # calculate baseline
    T = 1
    cifar = np.loadtxt('./ood/odin/softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./ood/odin/softmax_scores/confidence_Base_Out.txt', delimiter=',')
    if name == "CIFAR10": 
        start = 0.1
        end = 1 
    if name == "CIFAR100": 
        start = 0.01
        end = 1    
    gap = (end- start)/100000
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    auprBase = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprBase += (recallTemp-recall)*precision
        recallTemp = recall
    auprBase += recall * precision
        
    
    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('./ood/odin/softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./ood/odin/softmax_scores/confidence_Our_Out.txt', delimiter=',')
    if name == "CIFAR10": 
        start = 0.1
        end = 0.12 
    if name == "CIFAR100": 
        start = 0.01
        end = 0.0104    
    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    auprNew = 0.0
    recallTemp = 1.0
    for delta in np.arange(end, start, -gap):
        fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
        if tp + fp == 0: break
        precision = tp / (tp + fp)
        recall = tp
        auprNew += (recallTemp-recall)*precision
        recallTemp = recall
    auprNew += recall * precision
    return auprBase, auprNew



def detection(name):
    #calculate the minimum detection error
    # calculate baseline
    T = 1
    cifar = np.loadtxt('./ood/odin/softmax_scores/confidence_Base_In.txt', delimiter=',')
    other = np.loadtxt('./ood/odin/softmax_scores/confidence_Base_Out.txt', delimiter=',')
    if name == "CIFAR10": 
        start = 0.1
        end = 1 
    if name == "CIFAR100": 
        start = 0.01
        end = 1    
    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    errorBase = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorBase = np.minimum(errorBase, (tpr+error2)/2.0)

    # calculate our algorithm
    T = 1000
    cifar = np.loadtxt('./ood/odin/softmax_scores/confidence_Our_In.txt', delimiter=',')
    other = np.loadtxt('./ood/odin/softmax_scores/confidence_Our_Out.txt', delimiter=',')
    if name == "CIFAR10": 
        start = 0.1
        end = 0.12 
    if name == "CIFAR100": 
        start = 0.01
        end = 0.0104    
    gap = (end- start)/100000
    #f = open("./{}/{}/T_{}.txt".format(nnName, dataName, T), 'w')
    Y1 = other[:, 2]
    X1 = cifar[:, 2]
    errorNew = 1.0
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        errorNew = np.minimum(errorNew, (tpr+error2)/2.0)
            
    return errorBase, errorNew




def metric(indis, ood, arch):
    fprBase, fprNew = tpr95(indis)
    errorBase, errorNew = detection(indis)
    aurocBase, aurocNew = auroc(indis)
    auprinBase, auprinNew = auprIn(indis)
    auproutBase, auproutNew = auprOut(indis)
    print("{:31}{:>22}".format("Neural network architecture:", arch))
    print("{:31}{:>22}".format("In-distribution dataset:", indis))
    print("{:31}{:>22}".format("Out-of-distribution dataset:", ood))
    print("")
    print("{:>34}{:>19}".format("Baseline", "Our Method"))
    print("{:20}{:13.1f}%{:>18.1f}% ".format("FPR at TPR 95%:",fprBase*100, fprNew*100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("Detection error:",errorBase*100, errorNew*100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUROC:",aurocBase*100, aurocNew*100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUPR In:",auprinBase*100, auprinNew*100))
    print("{:20}{:13.1f}%{:>18.1f}%".format("AUPR Out:",auproutBase*100, auproutNew*100))
    return {
        "base auroc": aurocBase*100,
        "odin auroc": aurocNew*100,
        "base fpr@tpr95": fprBase*100,
        "odin fpr@tpr95": fprNew*100
    }

