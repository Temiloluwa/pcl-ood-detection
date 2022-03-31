import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models
from datasets.get_data import get_contrastive_loaders, get_contrastive_targets
from utils import compute_features, train_linear_model
from pprint import pprint
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        m = len(x)
        x = x.view(m, -1)
        return x

class Config:
    encoder = "key"
    output_layer = "avg_pool"

class Args:
    epochs = 300
    bs = 256
    lr = 0.01
    iid = "CIFAR10"


args = Args()
config = Config()
checkpoint_path = "/data/temiloluwa.adeoti/checkpoints/CIFAR10_clus_2816_neg_2560/exp_100/checkpoints"

results = {}
best_model = 0
best_acc = 0



def load_contrastive_model(checkpoint_path, config, num_classes=128):
    #checkpoint_path ='{}/checkpoint_{:04d}.pth.tar'.format(checkpoint_path, config.ckpt)
    
    model = models.__dict__['resnet50'](num_classes=num_classes)
    if config.output_layer == "avg_pool":
        model.fc = Identity()
    
    encoder = 'module.encoder_k' if config.encoder == 'key' else 'module.encoder_q'
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith(encoder) and not k.startswith(f'{encoder}.fc'):
            new_prefix = k[len(f"{encoder}."):]
            state_dict[new_prefix] = state_dict[k]
        del state_dict[k]
    
    msg = model.load_state_dict(state_dict, strict=False)
    print("model loaded")

    return model




for ckpt_path in Path(checkpoint_path).iterdir():
    ckpt = int(ckpt_path.stem.split("checkpoint_")[1].split(".pth")[0])
    if ckpt > 40:
        print(f"Using Checkpoint {ckpt}")
        model = load_contrastive_model(ckpt, config, num_classes=128)

    train_loader, test_loader = get_contrastive_loaders(args, args.data)
    train_features = compute_features(train_loader, model)
    test_features = compute_features(test_loader, model)
    train_targets, test_targets = get_contrastive_targets(args.data)

    _, train_acc, test_acc = train_linear_model(args, train_features, train_targets, test_features, test_targets)

    results[f"model_{ckpt}"] = {"train acc": train_acc, "test_acc": test_acc}
    if test_acc > best_acc:
        best_acc = test_acc
        best_model = f"model_{ckpt}"

        print(f"==> best model {best_model} with test accuracy {best_acc}")
    pprint(results)
    print("***********************************************")
