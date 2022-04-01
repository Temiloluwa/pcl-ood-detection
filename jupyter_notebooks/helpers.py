import os
import torch
import torch.nn as nn
import numpy as np
import faiss
from torchvision import models, transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from tqdm import tqdm

means = {"CIFAR10":[0.491, 0.482, 0.447],
        "CIFAR100": [0.5071, 0.4865, 0.4409],
        "STL10":[0.4914, 0.4822, 0.4465],
         "SVHN": [0.4377, 0.4438, 0.4728],
         "LSUNCrop":[0.5, 0.5, 0.5],
         "LSUNResize": [0.5, 0.5, 0.5],
         "Imagenet30": [0.485, 0.456, 0.406]
    } 

stds = {"CIFAR10":[0.247, 0.243, 0.262],
        "CIFAR100": [0.2673, 0.2564, 0.2762],
        "STL10":[0.2471, 0.2435, 0.2616],
        "SVHN":[0.1980, 0.2010, 0.1970],
        "LSUNCrop": [0.5, 0.5, 0.5],
        "LSUNResize": [0.5, 0.5, 0.5],
        "Imagenet30": [0.229, 0.224, 0.225]
} 


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        m = len(x)
        x = x.view(m, -1)
        return x


def load_model(checkpoint_path, ckpt, arch, num_classes, output_layer, encoder_type):
    checkpoint_path ='{}/checkpoint_{:04d}.pth.tar'.format(checkpoint_path, ckpt)
    
    model = models.__dict__[arch](num_classes=num_classes)
    
    encoder = 'module.encoder_k' if encoder_type  == 'key' else 'module.encoder_q'
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    print(f"loaded contrastive model @ ckpt: {checkpoint_path}")
    for k in list(state_dict.keys()):
        new_prefix = k[len(f"{encoder}."):]
        state_dict[new_prefix] = state_dict[k]
        
        del state_dict[k]
    
    msg = model.load_state_dict(state_dict, strict=False)

    if output_layer == "avg_pool":
        model.fc = Identity()
    
    if type(model.fc) == torch.nn.modules.linear.Linear:
        print(f"Contrastive model loaded with dims: {model.fc.out_features}")
    else:
        print(f"Outputing contrastive model from Avgpool Layer")
    return model


def run_kmeans(x, num_cluster):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    # intialize faiss clustering parameters
    d = x.shape[1]
    k = int(num_cluster)
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    clus.niter = 20
    clus.nredo = 5
    clus.seed = np.random.randint(0, 1000)
    clus.max_points_per_centroid = 1000
    clus.min_points_per_centroid = 10

    #res = faiss.StandardGpuResources()
    #cfg = faiss.GpuIndexFlatConfig()
    #cfg.useFloat16 = False
    #cfg.device = args.gpu    
    #index = faiss.GpuIndexFlatL2(res, d, cfg)
    index = faiss.IndexFlatL2(d)   

    clus.train(x, index)   

    D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
    im2cluster = [int(n[0]) for n in I]
    
    # get cluster centroids
    centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
    
    # sample-to-centroid distances for each cluster 
    Dcluster = [[] for c in range(k)]          
    for im,i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])
    
    # concentration estimation (phi)        
    density = np.zeros(k)
    for i,dist in enumerate(Dcluster):
        if len(dist)>1:
            d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
            density[i] = d     
            
    #if cluster only has one point, use the max to estimate its concentration        
    dmax = density.max()
    for i,dist in enumerate(Dcluster):
        if len(dist)<=1:
            density[i] = dmax 

    density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
    density = 0.2 *density/density.mean()  #scale the mean to temperature 
    
    # convert to cuda Tensors for broadcast
    #centroids = torch.Tensor(centroids).cuda()
    #centroids = nn.functional.normalize(centroids, p=2, dim=1)    

    #im2cluster = torch.LongTensor(im2cluster).cuda()               
    #density = torch.Tensor(density).cuda()
    
    results['centroids'].append(centroids)
    results['density'].append(density)
    results['im2cluster'].append(im2cluster)
        
    return results


def save_features(root_path, file_name, features):
    with open(os.path.join(root_path, file_name), 'wb') as f:
        np.save(f, features)
    

def load_features(root_path, file_name):
    with open(os.path.join(root_path, file_name), 'rb') as f:
        return np.load(f)


def load_model_features(checkpoint_path,  dataset, output_layer, norm_features=False):
    file_names = [f'{dataset}_{output_layer}_train_features.npy',
                        f'{dataset}_{output_layer}_test_features.npy',
                        f'{dataset}_{output_layer}_train_target.npy',
                        f'{dataset}_{output_layer}_test_target.npy']

    
    features = [load_features(checkpoint_path, v) for v in file_names]
    if norm_features:
        features  = [check_norm(v) if i < 2 else v for i, v in enumerate(features)]
    return features

def norm_feats(x):
    return  x/np.linalg.norm(x, axis=1, keepdims=True)


def check_norm(feat):
    n, _ = feat.shape
    tar = np.full(n, 1.0)

    try:
        np.testing.assert_allclose(\
                np.sum(feat **2, axis=1), tar, rtol=1e-05)
        normed = True
        print("normed")
    except:
        normed = False
        print("not normed")
    print(feat.shape)
    if normed:
        return feat
    else:
        return norm_feats(feat)




def train_linear_model(train_X, train_y, test_X, test_y, gpu, **kwargs):
    epochs = kwargs.get("epochs", 200)
    bs = kwargs.get("bs", 512)
    lr = kwargs.get("lr", 0.01)

    print(f"epochs: {epochs}, bs: {bs}, lr: {lr}")
 
    n_classes = int(np.unique(train_y).max() + 1)
    accuracy = lambda pred, y : np.mean(pred == y) * 100

    
    model = torch.nn.Sequential(
            torch.nn.Linear(train_X.shape[1], n_classes)
        )
    device = torch.device(f'cuda:{gpu}')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=0.9,
                                weight_decay=1e-4)

    train_idx = np.arange(len(train_X))
    num_bs = len(train_X) // bs + 1
    print_freq = 10
    model.to(device)

    train_X = torch.from_numpy(train_X).to(device)
    train_y = torch.from_numpy(train_y).to(device)
    test_X = torch.from_numpy(test_X).to(device)
    test_y = torch.from_numpy(test_y).to(device)

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


def run_clustering(data, num_cluster):
    clus_result = run_kmeans(data, num_cluster)
    im2cluster = np.array(clus_result['im2cluster']).flatten()
    prototypes = np.array(clus_result['centroids'][0])
    density = np.array(clus_result['density']).flatten()
    return im2cluster, prototypes, density


def softmax_t(logits, temp=1):
    logits = logits/temp
    _max = np.expand_dims(np.max(logits, axis=-1), axis=-1)
    probs = np.exp(logits - _max)
    _sum = np.expand_dims(np.sum(probs, axis=-1), axis=-1)
    return probs/_sum


def cluster_purity(kmeans_targets, in_targets):
    k_classes = np.unique(kmeans_targets).astype(int)
    k_class_idx = [np.nonzero(np.equal(cls_, kmeans_targets)) for cls_ in k_classes]
    in_classes_in_k_clstr = [in_targets[idx] for idx in k_class_idx]
    purity_list = []

    for cluster_k in in_classes_in_k_clstr:
        unique, counts = np.unique(cluster_k, return_counts=True)
        purity_list.append(np.round(np.asarray(counts).max()/len(cluster_k), 5))

    return purity_list