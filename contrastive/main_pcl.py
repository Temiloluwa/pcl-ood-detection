import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import faiss

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from contrastive.pcl import loader
from contrastive.pcl import builder
from datasets.constrastive_datasets import means, stds
from pprint import pprint
from utils import save_pickle, load_pickle
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def save_args(args, path):
    msg = "==> args \n"
    for k in vars(args):
        v = getattr(args, k)
        if type(v) is dict:
            continue
        msg += f"{k}: {v} \n"
    
    msg += "\n ==> Results \n"

    with open(os.path.join(path , "args.txt"), "w") as f:
        f.write(msg)

def contrastive_main(args):
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')


    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    args.exp_dir = os.path.join(args.exp_root, "checkpoints")

    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    
    ngpus_per_node = torch.cuda.device_count()
    save_args(args, args.exp_dir)

    pprint(args)
    
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master    
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
        
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = builder.MoCo(
        models.__dict__[args.arch],
        args.dim, args.pcl_r, args.moco_m, args.temperature, args.mlp, args.mlp2, args.pretrained)
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    
    # optionally resume from a checkpoint
    if args.resume:
        max_cpt = 0
        for fn in  os.listdir(args.exp_dir):
            if ".pth.tar" in fn:
                ckpt = int(fn.split("checkpoint_")[1].split(".pth.tar")[0])
                max_cpt = ckpt if ckpt > max_cpt else max_cpt

        filename ='{}/checkpoint_{:04d}.pth.tar'.format(args.exp_dir, max_cpt)
        filename = os.path.join(args.exp_dir, filename)
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            if args.gpu is None:
                checkpoint = torch.load(filename)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(0)
                checkpoint = torch.load(filename, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

    cudnn.benchmark = True

    # Data loading code
    #traindir = os.path.join(args.data, 'train')
    datadir = "/data/datasets"

    traindir = os.path.join(datadir, args.data)
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=means[args.data],
                                     std=stds[args.data])

    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709

        if args.data == "Imagenet30":
            print("Training with ImageNet30 Transform")
            augmentation = [
                #loader.Rot90(),
                transforms.RandomResizedCrop((args.input_image_height, args.input_image_width), scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
        ]
        else:
            augmentation = [
            transforms.RandomResizedCrop((args.input_image_height, args.input_image_width), scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]


    else:
        # MoCo v1's aug: same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop((args.input_image_height, args.input_image_width), scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        
    # center-crop augmentation
    if args.data == "Imagenet30":
        #print("Evaluating with ImageNet30 Transform")
        #eval_augmentation = transforms.Compose([
        #    #loader.Rot90(),
        #    transforms.Resize((args.input_image_height, args.input_image_width)),
        #    transforms.CenterCrop(224),
        #    transforms.ToTensor(),
        #    normalize
        #])

        eval_augmentation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
            ])
    else:
    #    eval_augmentation = transforms.Compose([
    #        transforms.Resize((args.input_image_height, args.input_image_width)),
    #        transforms.CenterCrop(224),
    #        transforms.ToTensor(),
    #        normalize
    #        ])

        eval_augmentation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
            ])

    
    if args.data == "CIFAR10":
        train_dataset = loader.CIFAR10Instance(traindir,
            train=True,
            transform=loader.TwoCropsTransform(transforms.Compose(augmentation)))
        

        eval_dataset = loader.CIFAR10Instance(traindir,
            train=True,
            transform=eval_augmentation)

    elif  args.data == "CIFAR100":
        train_dataset = loader.CIFAR100Instance(traindir,
            train=True,
            transform=loader.TwoCropsTransform(transforms.Compose(augmentation)))
        
        eval_dataset = loader.CIFAR100Instance(traindir,
            train=True,
            transform=eval_augmentation)

    elif args.data == "SVHN":
        train_dataset = loader.SVHNInstance(os.path.join(datadir, "svhn-data"),
            split="train",
            transform=loader.TwoCropsTransform(transforms.Compose(augmentation)))
        
        eval_dataset = loader.SVHNInstance(traindir,
            split="train",
            transform=eval_augmentation)

    elif args.data == "STL10":
        augmentation = [transforms.Resize(48)] + augmentation
        train_dataset = loader.STL10Instance(datadir,
            split="train",
            transform=loader.TwoCropsTransform(transforms.Compose(augmentation)))

        eval_dataset = loader.STL10Instance(datadir,
            split="train",
            transform=eval_augmentation)
    
    elif args.data == "Imagenet30":
        train_dataset = loader.ImageNet30(root=datadir + "/ImageNet30/train",
            transform=loader.TwoCropsTransform2(transforms.Compose(augmentation)))

        eval_dataset = loader.ImageNet30(root=datadir + "/ImageNet30/train",
            transform=eval_augmentation)
    

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset,shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
        
    
    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size*5, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers, pin_memory=True)

    
    for epoch in range(args.start_epoch, args.epochs):
        print(f"*** Epoch {epoch} ***")
        
        cluster_result = None
        if epoch>=args.warmup_epoch:
            # compute momentum features for center-cropped images
            features = compute_features(eval_loader, model, args)
                    
      
            # placeholder for clustering result
            cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
            for num_cluster in args.num_cluster:
                cluster_result['im2cluster'].append(torch.zeros(len(eval_dataset),dtype=torch.long).cuda())
                cluster_result['centroids'].append(torch.zeros(int(num_cluster),args.dim).cuda())
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda()) 
                

            if args.gpu == 0:
                features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
                features = features.numpy()
                cluster_result = run_kmeans(features,args)  #run kmeans clustering on master node
                # save the clustering result
                # torch.save(cluster_result,os.path.join(args.exp_dir, 'clusters_%d'%epoch))  
                
            dist.barrier()  
            # broadcast clustering result
            for k, data_list in cluster_result.items():
                for data_tensor in data_list:                
                    dist.broadcast(data_tensor, 0, async_op=False)   

            
    
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        adjust_learning_rate(optimizer, epoch, args)

        tb_dir = os.path.join(args.exp_dir, "tb")
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)

        prototypes_dir = os.path.join(args.exp_dir, "prototypes")
        os.makedirs(prototypes_dir, exist_ok=True)
        if cluster_result:
            if 'centroids' in cluster_result:
                save_pickle(prototypes_dir, "prototypes", cluster_result['centroids'])

        # train for one epoch
        data_dict = train(train_loader, model, criterion, optimizer, epoch, args, cluster_result)

        if args.rank == 0:
            add_to_tensorboard(writer, data_dict, epoch)
        
        if (epoch+1)%args.ckpt_freq==0 and (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0)):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.exp_dir,epoch))
        

def add_to_tensorboard(writer, data_dict, epoch):
    writer.add_scalar('loss', data_dict["loss"], epoch)
    writer.add_scalar('acc@Inst', data_dict["acc@Inst"] , epoch)
    writer.add_scalar('acc@Proto', data_dict["acc@Proto"], epoch)
    writer.add_scalar('info-nce', data_dict["info-nce"], epoch)
    writer.add_scalar('proto-loss', data_dict["proto-loss"], epoch)


def train(train_loader, model, criterion, optimizer, epoch, args, cluster_result=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')   
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')
    data_dict = {}
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_inst, acc_proto],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, index) in enumerate(train_loader):

        if epoch == 0:
            print(f"Input image shape : {images[0].shape}")
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
                
        # compute output
        output, target, output_proto, target_proto = model(im_q=images[0], im_k=images[1], cluster_result=cluster_result, index=index)
        
        # InfoNCE loss
        loss = criterion(output, target)

        data_dict["info-nce"] = loss
        data_dict["proto-loss"] = 0.0
        
        # ProtoNCE loss
        if output_proto is not None:
            loss_proto = 0
            for proto_out,proto_target in zip(output_proto, target_proto):
                loss_proto += criterion(proto_out, proto_target)  
                accp = accuracy(proto_out, proto_target)[0] 
                acc_proto.update(accp[0], images[0].size(0))
                
            # average loss across all sets of prototypes
            loss_proto /= len(args.num_cluster)
            data_dict["proto-loss"] = loss_proto 
            loss += loss_proto   

        losses.update(loss.item(), images[0].size(0))
        acc = accuracy(output, target)[0] 
        acc_inst.update(acc[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    data_dict["loss"] = losses.avg
    data_dict["acc@Inst"] = acc_inst.avg
    data_dict["acc@Proto"] = acc_proto.avg
    return data_dict
            
def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),args.dim).cuda()
    for i, (images, index) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            feat = model(images,is_eval=True) 
            features[index] = feat
    dist.barrier()        
    dist.all_reduce(features, op=dist.ReduceOp.SUM)     
    return features.cpu()

    
def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
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
        density = args.temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)
        
    return results

    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

