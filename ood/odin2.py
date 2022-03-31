import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import dataloader
import torchvision.transforms as transforms
from datasets.get_data import get_ensemble_eval_loaders
from utils import detach, load_distilled_kd_model
from tqdm import tqdm
from distilled.models import ResNet18
from torch.autograd import Variable


def softmax_t(logits, temp=1):
    logits = detach(logits)
    logits = logits/temp
    _max = np.expand_dims(np.max(logits, axis=-1), axis=-1)
    probs = np.exp(logits - _max)
    _sum = np.expand_dims(np.sum(probs, axis=-1), axis=-1)
    return probs/_sum


def top1(data, axis):
    return np.max(data, axis=axis), np.argmax(data, axis=axis)


def get_probs(id_dataset, loader, model, temperature, noise_magn, gpu):
    probs = []
    criterion = nn.CrossEntropyLoss()

    means = {
        "CIFAR10":[0.491, 0.482, 0.447],
        "CIFAR100": [0.5071, 0.4865, 0.4409],
    }

    stds = {
        "CIFAR10":[0.247, 0.243, 0.262],
        "CIFAR100": [0.2673, 0.2564, 0.2762]
    }
    
    normalize = transforms.Normalize(mean=means[id_dataset],
                                     std=stds[id_dataset])
    model.eval()
    for imgs, _ in tqdm(loader):
        imgs = Variable(imgs.cuda(gpu), requires_grad = True)
        o = model(imgs)
        max_idx = torch.argmax(o, dim=1)
        o = o / temperature
        
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        labels = Variable(max_idx.long()).cuda(gpu)
        loss = criterion(o, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(imgs.grad.data, 0).float()
        #gradient = (gradient.float() - 0.5) * 2

        # Normalizing the gradient to the same space of image
        gradient = normalize(gradient)
        imgs = Variable(torch.add(imgs.data, other=gradient, alpha=-noise_magn))
        
        with torch.no_grad():
            o = model(imgs)
            o = softmax_t(o, temperature)
            probs.append(o)

    probs = np.vstack(probs)
    return probs



def get_odin_data(id_dataset, ood_dataset, batch_size):
    print(f"id dataset {id_dataset}, ood dataset {ood_dataset}")
    i_train_loader, i_test_loader = get_ensemble_eval_loaders(id_dataset, id_dataset, batch_size)
    o_train_loader, o_test_loader = get_ensemble_eval_loaders(ood_dataset, id_dataset, batch_size)

    return i_train_loader, i_test_loader, o_test_loader


def prep_odin_for_auroc(args, ood_dataset, model):
    probs = []
    data_loaders = get_odin_data(args.data, ood_dataset, args.batch_size)
    for loader in data_loaders:
        probs.append(get_probs(args.data, loader, model, args.temperature, args.magnitude, args.eval_gpu))

    train_probs, train_pred = top1(probs[0], axis=-1)
    test_probs, test_pred = top1(probs[1], axis=-1)
    ood_probs, ood_pred = top1(probs[2], axis=-1)
    
    pred = np.concatenate([test_pred, ood_pred])
    ood_gt = np.concatenate([np.zeros_like(test_probs), np.ones_like(ood_probs)])
    
    class_idx = [np.where(train_pred == i) for i in range(args.num_classes)]
    group_scores = [train_probs[i] for i in class_idx]
    if args.use_train_groups:
        print("Using train groups")
        thresh_intvl = [(np.min(dat), np.max(dat)) for dat in group_scores]
    else:
        thresh_intvl = [(0, 1) for dat in group_scores]


    return thresh_intvl, test_probs, ood_probs, pred, ood_gt
