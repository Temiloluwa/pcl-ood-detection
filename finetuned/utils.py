import os
import random
import torch
import torch.nn as nn
from time import time
from torchvision import models
from easydict import EasyDict as edict
from ood import linear_evaluate_finetune, OodEvaluator
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR
from utils import load_contrastive_model, prepare_finetune_paths, save_pickle, load_pickle
from pprint import pprint


def create_model(args, model):
    checkpoint_path = os.path.join(args.exp_root, "checkpoints")
    model = load_contrastive_model(checkpoint_path, args, args.dim, model)
    
    def init(m):
        mean = random.triangular(-1, 1)
        std = random.triangular(0.5, 1.5)
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean=mean, std=std)
            m.bias.data.fill_(random.random())

    if args.random_init:
        model.fc.apply(init)

    model = attach_hook(model)
    return model
    

def save_model(args, model, optimizer, scheduler, all_results, ckpt_path):
    state = {
        "all_results": all_results,
        "epoch": args.epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if args.scheduler.use else None
    }

    torch.save(state, ckpt_path)
    print("==> Finetune Model Saved")

    msg = "==> args \n"
    for k in vars(args):
        v = getattr(args, k)
        if type(v) is dict:
            continue
        msg += f"{k}: {v} \n"
    
    msg += "\n ==> Results \n"
    for k, v in all_results[-1].items():
        msg += f"{k}: {v} \n"    

    with open(os.path.join(ckpt_path[:-18] + "log.txt"), "w") as f:
        f.write(msg)


    
def load_model(args, model, optimizer, scheduler, ckpt_path, gpu):
    checkpoint = torch.load(ckpt_path, map_location=f"cuda:{gpu}")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if args.scheduler.use:
        scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    all_results = checkpoint['all_results']
    print(f"Model Loaded and Training resumes from epoch: {epoch}")
    return model, optimizer, scheduler, epoch, all_results


def freeze_bn(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()
        print(f"{str(module)} is in eval mode")


def un_freeze_bn(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.train()
        print(f"{str(module)} is in train mode")


def freeze_encoder(model, logging, freeze=True):
    for name, param in model.named_parameters():
        if "fc" not in name:
            if freeze:
                if param.requires_grad:
                    param.requires_grad = False
                    print(f"param {name} frozen")
                    logging.info(f"param {name} frozen")
                model.apply(freeze_bn)
            else:
                if not param.requires_grad:
                    param.requires_grad = True
                    print(f"param {name} unfrozen")
                    logging.info(f"param {name} unfrozen")
                model.apply(un_freeze_bn)
    return model


def combine_dicts(dics):
    combo = dict()
    for dic in dics:
        for k,v in dic.items():
            combo[k] = v
    return combo


def get_trainable_params(args, model):
    if args.freeze_bn:
        # no batch norm
        trainable_parameters = [v for n,v in model.named_parameters() if "bn" not in n]
        trainable_parameters = nn.ModuleList(*trainable_parameters)
    else:
        trainable_parameters = model
    return trainable_parameters


def attach_hook(model):
    def hook(self, input, output):
        output = output.view(output.shape[0], -1)
        model.hook_outputs = output
    model.avgpool.register_forward_hook(hook)
    return model


def perform_ood_detection(args, model):
    results = []
    for iid_data, ood_data in linear_evaluate_finetune(args, model, use_hook=True):
        prot = 0 if ood_data.shape[1] in [10, 100] else args.prot
        ood_evaluator = OodEvaluator(*iid_data, prot, args.com)
        ood_evaluator(ood_data, args.met).get_scores()
        ood_evaluator.get_auroc()
        results.append((ood_evaluator.auroc, ood_evaluator.tnr_at_tpr95))
    
    print("************* ood results *****************")
    results = { "epochs":args.epoch,
                "metric": args.met,
                "pca components": args.com,
                "clusters": args.prot,
                "avgpool_auroc": results[0][0],
                "avgpool_tnr@tpr95": results[0][1],
                "fc_auroc": results[1][0],
                "fc_tnr@tpr95": results[1][1]
            }
    pprint(results)
    print("******************************************")
    return results


def get_scheduler(args, optimizer):
    if args.use:
        if args.type == "cosine":
            cos = args.cosine
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cos.T_0, T_mult=cos.T_mult)
            print("using cosine scheduler")
        else:
            lin = args.linear
            scheduler = MultiStepLR(optimizer, milestones=lin.ms, gamma=lin.gamma)
            print("using linear scheduler")
    else:
        scheduler = None
        print("using no scheduler")
    return scheduler



def finetune_ckpt_evaluator(args):
    args.scheduler = edict(args.scheduler)
    save_path = os.path.join(args.checkpoint_path, "finetune")
    ckpt_root = os.path.join(save_path, prepare_finetune_paths(args))
   
    model = models.__dict__['resnet50'](num_classes=args.num_classes).cuda(args.eval_gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    scheduler = get_scheduler(args.scheduler, optimizer)
    best_avg_auroc = 0.0
    best_fc_auroc = 0.0
    best_avg_epoch = 0
    best_fc_epoch = 0

    
    def epochs_to_delete(best_avg_epoch, best_fc_epoch, epoch):
        available_ckpts = [int(i.split("checkpoint_")[1].split(".pth.tar")[0]) for i in os.listdir(ckpt_root) if ".pth" in i]
        available_ckpts = [ i for i in available_ckpts if i in available_ckpts if i <= epoch]
        del_ckpts = list(set(available_ckpts) - set([best_avg_epoch, best_fc_epoch]))
        del_ckpts = [os.path.join(ckpt_root, f"checkpoint_{i:04d}.pth.tar") for i in del_ckpts]
        return del_ckpts

    if os.path.exists(os.path.join(ckpt_root, "train_results.pickle")):
        train_results = load_pickle(ckpt_root, "train_results")
    else:
        train_results = []
    
    start_epoch = args.unfreeze_epoch if args.unfreeze_epoch else 0
    for epoch in range(start_epoch, args.epochs):
        args.epoch = epoch
        ckpt_path = os.path.join(ckpt_root, f"checkpoint_{epoch:04d}.pth.tar")
        if not os.path.exists(ckpt_path):
            print(f"skipped epoch {epoch}")
            continue
        model, ___, ___, __, _ = load_model(args, model, optimizer, scheduler, ckpt_path, args.eval_gpu)
        model = attach_hook(model)
        ood_results = perform_ood_detection(args, model)
        
        if ood_results["avgpool_auroc"] > best_avg_auroc:
            best_avg_auroc = ood_results["avgpool_auroc"]
            best_avg_epoch = epoch

        if ood_results["fc_auroc"] >  best_fc_auroc:
            best_fc_auroc = ood_results["fc_auroc"]
            best_fc_epoch = epoch

        print(f"=> **** Best Epoch: {best_avg_epoch}, Best Avg AUROC: {best_avg_auroc}, Best Epoch: {best_fc_epoch}, Best FC AUROC: {best_fc_auroc}  ****")
        train_results.append(ood_results)
        save_pickle(ckpt_root, "train_results", train_results)
    
        for ckpt in epochs_to_delete(best_avg_epoch, best_fc_epoch, epoch):
            os.remove(ckpt)
            print(f"{ckpt} deleted")
