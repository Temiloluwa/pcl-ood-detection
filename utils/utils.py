import torch
import os
import numpy as np
import pickle
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from collections import namedtuple
from itertools import product
from easydict import EasyDict as edict

Feats_n_targs = namedtuple("feats_n_targs", "train_features, test_features, train_targets, test_targets")


def compute_features(eval_loader, model, use_hook=False):
    print('Computing features...')

    for param in model.parameters():
        device_id = param.get_device()
        break
    if model.training:
        raise ValueError("model should be in eval state")
    else:
        print("Model in Eval state")

    feats = []
    for i, (images, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(device_id)
            feats.append(detach(model(images)))
    feats = np.vstack(feats)
    print(f"feats shape: {feats.shape}")
    return feats
            

def save_features(root_path, file_name, features):
    with open(os.path.join(root_path, file_name), 'wb') as f:
        np.save(f, detach(features))
        print(f"saved features @ {os.path.join(root_path, file_name)}")
    

def load_features(root_path, file_name):
    with open(os.path.join(root_path, file_name), 'rb') as f:
        return np.load(f)


def detach(array):
    return array if type(array) == np.ndarray else \
            array.detach().cpu().numpy()


def normalize(x):
    return  x/np.linalg.norm(x, axis=1, keepdims=True)


def train_linear_model(args, train_X, train_y, test_X, test_y, model=None):
    epochs = args.epochs
    bs = args.batch_size
    lr = args.lr
    train_X = torch.from_numpy(train_X)
    test_X = torch.from_numpy(test_X)

    n_classes = int(np.unique(train_y).max() + 1)
    accuracy = lambda pred, y : np.mean(pred == y) * 100

    if model is None:
        model = torch.nn.Sequential(
            torch.nn.Linear(train_X.shape[1], n_classes)
        )
    device = torch.device('cuda:1')
    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    train_idx = np.arange(len(train_X))
    num_bs = len(train_X) // bs + 1
    print_freq = 10
    model.to(device)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):
        np.random.shuffle(train_idx)
        epoch_loss = []
        epoch_accuracy = []
        for i in range(num_bs):
            idx = train_idx[i*bs: (i+1)*bs]
            train_data = train_X[idx].to(device)
            target = train_y[idx].to(device)
            output = model(train_data)
            loss = criterion(output, target.long())
            optimizer.zero_grad()
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()
            train_pred = torch.argmax(output, axis=-1).detach().cpu().numpy()
            epoch_accuracy.append(accuracy(train_pred, target.detach().cpu().numpy()))
        
        scheduler.step()
        
        train_losses.append(np.mean(epoch_loss))
        train_accuracies.append(np.mean(epoch_accuracy))
        epoch_loss = []
        epoch_accuracy = []
        with torch.no_grad():
            for j in range(len(test_X) // bs + 1):
                test_data = test_X[j*bs: (j+1)*bs].to(device)
                target = test_y[j*bs: (j+1)*bs].to(device)
                output = model(test_data)
                loss = criterion(output, target.long())
                epoch_loss.append(loss.item())
                test_pred = torch.argmax(output, axis=-1).detach().cpu().numpy()
                epoch_accuracy.append(accuracy(test_pred, target.detach().cpu().numpy()))
        
        test_losses.append(np.mean(epoch_loss))
        test_accuracies.append(np.mean(epoch_accuracy))
        print_statement = f"Epoch {epoch+1:02d} : train loss {train_losses[-1]:.2f}, "
        print_statement += f"test loss {test_losses[-1]:.2f}, train accuracy {train_accuracies[-1]:.2f}%, "
        print_statement += f"test accuracy {test_accuracies[-1]:.2f}% "
        
        if epoch % print_freq == 9:
            print(print_statement)
    
    return model, train_accuracies[-1], test_accuracies[-1]


def save_pickle(root, filename, data):
    filename = os.path.join(root, 
                     f'{filename}.pickle')
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"saved {filename}")

        
def load_pickle(root, filename):
    filename = os.path.join(root, 
                     f'{filename}.pickle')
    with open(filename, 'rb') as f:
        return pickle.load(f)


def prepare_finetune_paths(args):
    ckpt_name = "{}_ckpt_{}_mn_{}_lp_{}".format(

        args.encoder, args.ckpt, args.model_no, 10
    )

    return ckpt_name

    

def save_results_excel(args, result_path):
    q = pd.DataFrame(load_pickle(result_path, "results"))
    p_m = q[['auroc', 'tnr@tpr95', 'ood', 'metric', 'clusters', 'pca components']]
    for _val in ['auroc', 'tnr@tpr95']:
        for ood in args.ood_datasets:
            save_path = f"{result_path}/{ood}_{_val}.xlsx"
            p_m[p_m.ood == ood].pivot(index=['metric', 'pca components'], columns=['clusters'], values=[_val])\
                                .apply(lambda x: round(x, 3)).to_excel(save_path)


def gen_configs(model_config):
    configs = []
    
    k_v_p = []
    for k, v in model_config.items():
        k_v = []
        for i in range(len(v)):
            k_v.append((k, v[i]))
        k_v_p.append(k_v)
    
    for kv in product(*k_v_p):
        configs.append({k:v for k, v in kv})
    
    return configs


def update_args(args, config):
    for k, v in vars(config).items():
        if type(v) is dict:
            v = edict(v)
        setattr(args, k, v)
    return args
