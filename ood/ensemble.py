import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from datasets.get_data import get_ensemble_eval_loaders
from utils import load_finetuned_model, detach, load_distilled_kd_model
from sklearn import metrics
from distilled.models import ResNet18
from utils import save_pickle


def softmax_t(logits, temp=1):
    logits = detach(logits)
    logits = logits/temp
    _max = np.expand_dims(np.max(logits, axis=-1), axis=-1)
    probs = np.exp(logits - _max)
    _sum = np.expand_dims(np.sum(probs, axis=-1), axis=-1)
    return probs/_sum


def top1(data, axis):
    return np.max(data, axis=axis), np.argmax(data, axis=axis)


def get_feats(loader, model, gpu):
    probs = []
    logits = []
    with torch.no_grad():
        for i, (imgs, targ) in enumerate(loader):
            imgs = imgs.cuda(gpu)
            o = model(imgs)
            logits.append(o)
            o = F.softmax(o, dim=1)
            probs.append(o)
            if i == 0:
                targs = targ
            else:
                targs = torch.cat([targs, targ], dim=0)

    probs = torch.vstack(probs).detach().cpu()
    logits = torch.vstack(logits).detach().cpu()
    targs = targs.detach().cpu()
    return probs, logits, targs



def get_ensemble_data(args, ood_dataset, ckpt_paths):
    ens = args.ood.ensemble
    id_dataset = args.data
    
    ood_results = []
    print(f"id dataset {id_dataset}, ood dataset {ood_dataset}")
    i_train_loader, i_test_loader = get_ensemble_eval_loaders(id_dataset, id_dataset, args.batch_size, args.center_crop)
    o_train_loader, o_test_loader = get_ensemble_eval_loaders(ood_dataset, id_dataset, args.batch_size, args.center_crop)

    train_probs = []
    train_logits = []
    train_targs = []
    test_probs = []
    test_logits = []
    test_targs = []
    ood_probs = []
    ood_logits = []
    ood_targs = []
    for ckpt in ckpt_paths:
        if args.model_type == "finetuned":
            model = load_finetuned_model(ens, ckpt)
        elif args.model_type == "distilled_kd":
            model = load_distilled_kd_model(ens, ckpt)

        
        model.eval()    
        probs, logits, targs = get_feats(i_train_loader, model, ens.eval_gpu)
        train_probs.append(probs)
        train_logits.append(logits)
        train_targs.append(targs)
        probs, logits, targs = get_feats(i_test_loader, model, ens.eval_gpu)
        test_probs.append(probs)
        test_logits.append(logits)
        test_targs.append(targs)
        probs, logits, targs = get_feats(o_test_loader, model, ens.eval_gpu)
        ood_probs.append(probs)
        ood_logits.append(logits)
        ood_targs.append(targs)

    train_probs = torch.stack(train_probs).numpy()
    test_probs = torch.stack(test_probs).numpy()
    ood_probs = torch.stack(ood_probs).numpy()

    train_logits = torch.stack(train_logits).numpy()
    test_logits = torch.stack(test_logits).numpy()
    ood_logits = torch.stack(ood_logits).numpy()

    train_targs = torch.stack(train_targs).numpy()
    test_targs = torch.stack(test_targs).numpy()
    ood_targs = torch.stack(ood_targs).numpy()

    return train_logits, test_logits, ood_logits


def prep_for_auroc(args, temp, train_logits, test_logits, ood_logits, idx):
    if train_logits.ndim > 2 and idx != -1:
        train_logits = train_logits[idx]
        test_logits = test_logits[idx]
        ood_logits = ood_logits[idx]

    ens = args.ood.ensemble
    train_logits = softmax_t(train_logits, temp=temp)
    test_logits = softmax_t(test_logits, temp=temp)
    ood_logits = softmax_t(ood_logits, temp=temp)
    
    if train_logits.ndim > 2:
        train_logits = np.mean(train_logits, axis=0)
        test_logits = np.mean(test_logits, axis=0)
        ood_logits = np.mean(ood_logits, axis=0)

    train_probs, train_pred = top1(train_logits, axis=-1)
    test_probs, test_pred = top1(test_logits, axis=-1)
    ood_probs, ood_pred = top1(ood_logits, axis=-1)
    
    pred = np.concatenate([test_pred, ood_pred])
    ood_gt = np.concatenate([np.zeros_like(test_probs), np.ones_like(ood_probs)])
    
    class_idx = [np.where(train_pred == i) for i in range(ens.num_classes)]
    group_scores = [train_probs[i] for i in class_idx]
    thresh_intvl = [(np.min(dat) , np.max(dat)) for dat in group_scores]


    return thresh_intvl, test_probs, ood_probs, pred, ood_gt


def cache_predictions(data, root_path):
    _, test_probs, ood_probs, pred, ood_gt = data
    if not os.path.exists(root_path):
        os.makedirs(root_path, exist_ok=True)
        
    save_pickle(root_path, "test_probs", test_probs)
    save_pickle(root_path, "ood_probs", ood_probs)
    save_pickle(root_path, "pred", pred)
    save_pickle(root_path, "ood_gt", ood_gt)


def get_auroc(thresh_intvl, test_probs, ood_probs, pred, ood_gt):
    num_threshs = 100
    pred_probs = np.concatenate([test_probs, ood_probs])
    metrics_results = []
    thresh_ranges = np.array([list(np.linspace(i[0], i[1], num_threshs)) for i in thresh_intvl])
    for thresh in range(num_threshs):
        thresh_scores = np.array([thresh_ranges[i, thresh] for i in pred])
        ood_pred = pred_probs < thresh_scores
        tn, fp, fn, tp = metrics.confusion_matrix(ood_gt , ood_pred).ravel()
        metrics_results.append({"tpr": tp/(tp+fn), "fpr": fp/(fp+tn), "tnr": tn/(tn+fp)})

    confusion_df = pd.DataFrame(metrics_results)
    df_auroc = confusion_df.sort_values(by=['fpr'], ascending=False)
    auroc = metrics.auc(df_auroc['fpr'], df_auroc['tpr'])
    auroc = np.round(auroc, 3)
    
    df_tpr95 = confusion_df.sort_values(by=['tpr'], ascending=False)
    tnr_at_tpr95 = df_tpr95[df_tpr95.tpr >= 0.95].iloc[-1]['tnr']
    tnr_at_tpr95 = np.round(tnr_at_tpr95, 3)

    return auroc, tnr_at_tpr95