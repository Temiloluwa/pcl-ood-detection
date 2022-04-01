import os
import torch
import torch.nn as nn
import logging
from torch.utils.tensorboard import SummaryWriter
from time import time
from torchvision import models
from easydict import EasyDict as edict
from distilled.helper.util import accuracy, AverageMeter
from datasets.get_data import get_finetune_train_loaders
from utils import prepare_finetune_paths
from finetuned.utils import get_scheduler, load_model, attach_hook, \
    create_model, save_model, freeze_encoder


train_time = AverageMeter()
train_loss = AverageMeter()
test_loss = AverageMeter()
train_accuracy = AverageMeter()
test_accuracy = AverageMeter()
start_time = time()
all_results = []


def add_to_tensorboard(writer, data_dict, epoch):
    writer.add_scalar('train loss', data_dict["train loss"], epoch)
    writer.add_scalar('test loss', data_dict["test loss"], epoch)
    writer.add_scalar('train acc', data_dict["train acc"], epoch)
    writer.add_scalar('test acc', data_dict["test acc"], epoch)


def train(args, train_loader, test_loader, model, criterion, optimizer, scheduler):
    _start_time = time()
    data_dict = {}
    model.train()
    epoch = args.epoch
    total_train_images = 0
    total_test_images = 0

    for i, (images, target) in enumerate(train_loader):
        if epoch == 0:
            total_train_images += len(images)

        images = images.cuda(args.eval_gpu)
        target = target.cuda(args.eval_gpu)
        output = model(images)
        output = output/args.temperature
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), images.size(0))
        acc = accuracy(output, target, topk=(1,))[0]
        train_accuracy.update(acc.item(), images.size(0))

    if args.scheduler.use:
        scheduler.step()

    with torch.no_grad():
        model.eval()
        for i, (images, target) in enumerate(test_loader):
            if epoch == 0:
                total_test_images += len(images)
            images = images.cuda(args.eval_gpu)
            target = target.cuda(args.eval_gpu)
            output = model(images)
            output = output/args.temperature
            loss = criterion(output, target)
            acc1 = accuracy(output, target, topk=(1,))[0]
            test_loss.update(loss.item(), images.size(0))
            test_accuracy.update(acc1.item(), images.size(0))
    
    epoch_time = (time() - _start_time)/60
    train_time.update(epoch_time)
    
    if (epoch + 1) % args.print_freq == 0:
        print_msg =  f"Epoch: {epoch + 1}, Epoch time: {epoch_time:.2f} minutes, "
        print_msg += f"Total time: {train_time.sum:.2f} minutes \n"
        print_msg += f"Train loss: {train_loss.avg:.2f}, "
        print_msg += f"Test loss: {test_loss.avg:.2f} \n"
        print_msg += f"Train acc: {train_accuracy.avg:.2f}%, "
        print_msg += f"Test acc: {test_accuracy.avg:.2f}% \n"
        print(print_msg)
        logging.info(print_msg)

    if epoch == 0:
        print(f"GPU {args.eval_gpu}: Training on {total_train_images} train images")
        logging.info(f"GPU {args.eval_gpu}: Training on {total_train_images} train images")
        print(f"GPU {args.eval_gpu}: Evaluating on {total_test_images} test images")
        logging.info(f"GPU {args.eval_gpu}: Evaluating on {total_test_images} test images")
    
    data_dict["epoch"] = epoch
    data_dict["start time"] = start_time
    data_dict["total time"] = train_time.sum/60
    data_dict["train loss"] = train_loss.avg
    data_dict["test loss"] = test_loss.avg
    data_dict["train acc"] = train_accuracy.avg
    data_dict["test acc"] = test_accuracy.avg

    return model, data_dict


def finetune_main(args, CF):
    for k, v in CF.items():
        setattr(args, k, v)
        
    global start_time
    global all_results
    best_test_acc = 0
    args.scheduler = edict(args.scheduler)
    save_path = os.path.join(args.checkpoint_path, "finetune")
    ckpt_root = os.path.join(save_path, prepare_finetune_paths(args))
    os.makedirs(ckpt_root, exist_ok=True)
    if "checkpoint_" in os.listdir(ckpt_root):
        last_ckpt = max([int(i.split("checkpoint_")[1].split(".pth.tar")[0]) for i in os.listdir(ckpt_root) if ".pth" in i])
        ckpt_path = os.path.join(ckpt_root, f"checkpoint_{last_ckpt:04d}.pth.tar")
    else:
        ckpt_path = None
    
    tb_dir = os.path.join(ckpt_root, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)

    logging.basicConfig(filename=os.path.join(ckpt_root, "train.log"),\
            level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')


    model = models.__dict__['resnet50'](num_classes=args.num_classes).cuda(args.eval_gpu)
    criterion = nn.CrossEntropyLoss().cuda(args.eval_gpu)

    if ckpt_path and args.resume:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        scheduler = get_scheduler(args.scheduler, optimizer)
        model, optimizer, scheduler, start_epoch, all_results = load_model(args, model, optimizer, scheduler, ckpt_path, args.eval_gpu)
        model = attach_hook(model)
        start_time = all_results[-1]["start time"]
    else:
        start_epoch = 0
        model = create_model(args, model)
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        scheduler = get_scheduler(args.scheduler, optimizer)
        if args.unfreeze_epoch:
            model = freeze_encoder(model, logging, freeze=True)
        
    train_loader, test_loader = get_finetune_train_loaders(args.data, args.label_percent, args.batch_size, args.data_augmentation)

    print(f"==> Training on GPU {args.eval_gpu}")
    logging.info(f"==> Training on GPU {args.eval_gpu}")

    for epoch in range(start_epoch, args.epochs):
        print(f"==> epoch {epoch}")
        logging.info(f"==> epoch {epoch}")
        ckpt_path = os.path.join(ckpt_root, f"checkpoint_{epoch:04d}.pth.tar")
        best_ckpt_path = os.path.join(ckpt_root, f"checkpoint_{args.epochs:04d}.pth.tar")
        ckpt_freq = args.epochs // 5        
        args.epoch = epoch

        if epoch == args.unfreeze_epoch and args.unfreeze_epoch:
            model = freeze_encoder(model, logging, freeze=False)
            
        model, data_dict = train(args, train_loader, test_loader, model, criterion, optimizer, scheduler)
        all_results.append(data_dict)
        add_to_tensorboard(writer, data_dict, epoch)
        if best_test_acc < data_dict["test acc"]:
            save_model(args, model, optimizer, scheduler, all_results, best_ckpt_path)
            best_test_acc  = data_dict["test acc"]
        
        if epoch % ckpt_freq == 0:
            save_model(args, model, optimizer, scheduler, all_results, ckpt_path)