import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from . import load_features, save_features, detach
#from distilled.models import model_dict
from easydict import EasyDict as edict

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        m = len(x)
        x = x.view(m, -1)
        return x

def load_contrastive_model(checkpoint_path, config, num_classes=128, model=None):
    checkpoint_path ='{}/checkpoint_{:04d}.pth.tar'.format(checkpoint_path, config.ckpt)
    
    if model is None:
        model = models.__dict__[config.arch](num_classes=num_classes).cuda(config.eval_gpu)
    
    encoder = 'module.encoder_k' if config.encoder == 'key' else 'module.encoder_q'
    checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{config.eval_gpu}")
    state_dict = checkpoint['state_dict']
    print(f"loaded contrastive model @ ckpt: {checkpoint_path}")
    for k in list(state_dict.keys()):
        if k.startswith(encoder):
            new_prefix = k[len(f"{encoder}."):]
            state_dict[new_prefix] = state_dict[k]
        
        del state_dict[k]
    
    msg = model.load_state_dict(state_dict, strict=False)

    if config.output_layer == "avg_pool":
        model.fc = Identity()

    if type(model.fc) == torch.nn.modules.linear.Linear:
        print(f"Contrastive model loaded with dims: {model.fc.out_features}")
    else:
        print(f"Outputing contrastive model from Avgpool Layer")
        
    return model


def cache_contrastive_features(args, dataset_name, save_path, train_features, test_features, train_targets, test_targets):
    
    if args.model_type == "contrastive":
        output_layer = args.ood.metrics.output_layer
    elif args.model_type == "finetuned":
        output_layer = args.output_layer
    else:
        output_layer = args.teacher.output_layer
    
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    save_features(save_path, f'{dataset_name}_{output_layer}_train_features.npy', train_features)
    save_features(save_path, f'{dataset_name}_{output_layer}_test_features.npy', test_features)


    np.testing.assert_allclose(load_features(save_path,\
                                f'{dataset_name}_{output_layer}_train_features.npy'), \
                                detach(train_features))
    np.testing.assert_allclose(load_features(save_path, \
                                f'{dataset_name}_{output_layer}_test_features.npy'), \
                                detach(test_features))

    save_features(save_path, f'{dataset_name}_{output_layer}_train_target.npy', train_targets)
    save_features(save_path, f'{dataset_name}_{output_layer}_test_target.npy', test_targets)

    np.testing.assert_allclose(load_features(save_path, \
                        f'{dataset_name}_{output_layer}_train_target.npy'), \
                            detach(train_targets))
    np.testing.assert_allclose(load_features(save_path,\
                            f'{dataset_name}_{output_layer}_test_target.npy'), \
                            detach(test_targets))


def load_distilled_model(args):
    num_classes = 10 if args.data == "CIFAR10" else 100
    args.model_path = os.path.join(args.checkpoint_path, 'distilled', 'finetuned')
    args.model_name = 's_{}_t_{}_{}_gamma_{}_alpha_{}_beta_{}_trial_{}'.format(args.model_s, "resnet50",  args.distill,
                                                                args.gamma, args.alpha, args.beta, args.trial_no)
    save_path = os.path.join(args.model_path, args.model_name)
    if args.use_best_ckpt:
        save_file = os.path.join(save_path, '{}_best.pth'.format(args.model_s))
    else:
        save_file = os.path.join(save_path, f'ckpt_epoch_{args.ds_ckpt}.pth')
    model = model_dict[args.model_s](num_classes=num_classes)
    checkpoint = torch.load(save_file, map_location=f"cuda:{args.eval_gpu}")
    model.load_state_dict(checkpoint['model'])
    if args.output_layer == "avg_pool":
        model.linear = Identity()
    model.cuda(args.eval_gpu)
    return model


def load_distilled_kd_model(args, ckpt_path):
    model = models.__dict__['resnet18'](num_classes=args.num_classes).cuda(args.eval_gpu)
    checkpoint = torch.load(ckpt_path, map_location=f"cuda:{args.eval_gpu}")
    model.load_state_dict(checkpoint['model'])
    print(f"distilled_kd model loaded @ ckpt {ckpt_path}")
    model.cuda(args.eval_gpu)
    return model
    

def load_finetuned_model(args, ckpt_path):
    num_classes = args.num_classes
    model = models.__dict__[args.arch](num_classes=num_classes)
    checkpoint = torch.load(ckpt_path, map_location=f"cuda:{args.eval_gpu}")
    model.load_state_dict(checkpoint['state_dict'])
    print(f"finetuned model loaded @ ckpt {ckpt_path}")
    model.cuda(args.eval_gpu)
    return model



def load_model_features(args, dataset):
    
    metrics = args.ood.metrics
    output_layer = metrics["output_layer"]
    encoder = metrics["encoder"] 
    checkpoint_path = os.path.join(args.checkpoint_path, \
        "feats_cache", f"{args.model_type}", f"feat_cache_{encoder}_ckpt_{args.ckpt}")
       

    file_names = [f'{dataset}_{output_layer}_train_features.npy',
                        f'{dataset}_{output_layer}_test_features.npy',
                        f'{dataset}_{output_layer}_train_target.npy',
                        f'{dataset}_{output_layer}_test_target.npy']

    
    features = [load_features(checkpoint_path, v) for v in file_names]

    return features

