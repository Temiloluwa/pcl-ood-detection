import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from torchvision import models
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter
from distilled.helper.util import accuracy, AverageMeter
from datasets.get_data import get_distill_loaders_2
from utils import prepare_finetune_paths
from pprint import pprint
from finetuned.utils import get_scheduler, load_model, attach_hook, \
    create_model, save_model

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


def train(args, train_loader, test_loader, model_s, model_t, optimizer):
    _start_time = time()
    data_dict = {}
    model_s.train()
    model_t.eval()
    epoch = args.epoch
    total_train_images = 0
    total_test_images = 0

    for i, (images, target) in enumerate(train_loader):
        if epoch == 0:
            total_train_images += len(images)
        
        target = target.cuda(args.eval_gpu)
        images = images.cuda(args.eval_gpu)
        with torch.no_grad():
            t_logits = model_t(images)
 
        s_logits = model_s(images)
        loss = criterion(s_logits, t_logits.cuda(args.eval_gpu), args.kd_T)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), images.size(0))
        acc = accuracy(F.softmax(s_logits, dim=1), target, topk=(1,))[0]
        train_accuracy.update(acc.item(), images.size(0))

    model_s.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(test_loader):
            images = images.cuda(args.eval_gpu)
            if epoch == 0:
                total_test_images += len(images)
            
            target = target.cuda(args.eval_gpu)
            t_logits = model_t(images)
            s_logits = model_s(images)
            loss = criterion(s_logits, t_logits.cuda(args.eval_gpu), args.kd_T)
            acc1 = accuracy(F.softmax(s_logits, dim=1), target, topk=(1,))[0]
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

    if epoch == 0:
        print(f"GPU {args.eval_gpu}: Training on {total_train_images} train images")
        print(f"GPU {args.eval_gpu}: Evaluating on {total_test_images} test images")
    
    data_dict["epoch"] = epoch
    data_dict["start time"] = start_time
    data_dict["total time"] = train_time.sum/60
    data_dict["train loss"] = train_loss.avg
    data_dict["test loss"] = test_loss.avg
    data_dict["train acc"] = train_accuracy.avg
    data_dict["test acc"] = test_accuracy.avg

    return model_s, data_dict


def criterion(s_logits, t_logits, temp):
    bs, _ = s_logits.shape
    t_probs = F.softmax(t_logits/temp, dim=1)
    s_logits = s_logits/temp
    loss = torch.log(torch.sum(torch.exp(s_logits), dim=1).reshape(bs, -1)) - s_logits 
    loss = torch.mean(temp **2 * torch.sum(t_probs * loss, dim=1))
    return loss


def load_teacher(args):

    def hook(self, input, output):
        output = output.view(output.shape[0], -1)
        model.avgpool_output = output

    if args.teacher.model_type == "finetuned":
        args.teacher.ckpt = args.ckpt
        ckpt_root = os.path.join(args.checkpoint_path, "finetune", prepare_finetune_paths(args.teacher))
        ckpt_path = os.path.join(ckpt_root, f"checkpoint_{args.teacher.ft_ckpt:04d}.pth.tar")
        print('==> loading teacher model')
        model = models.__dict__['resnet50'](num_classes=args.num_classes)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])
        print('==> done')
        model.avgpool.register_forward_hook(hook)
    return model


def distill_kd_main(args, CD):
    global start_time
    global all_results

    for k, v in CD.items():
        setattr(args, k, v)

    best_acc = 0
    args.num_classes = 10 if args.data == "CIFAR10" else 100
    args.eval_gpu = args.teacher.gpu
    
    args.teacher = edict(args.teacher)
    args.teacher.ckpt = args.ckpt
    save_path = os.path.join(args.checkpoint_path, 'distilled_kd', f'{prepare_finetune_paths(args.teacher)}')
    os.makedirs(save_path, exist_ok=True)

    logging.basicConfig(filename=os.path.join(save_path, "train.log"),\
            level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
   
    msg = "==> teacher config \n"
    for k, v in args.teacher.items():
        msg += f"{k}: {v} \n"

    msg += "==> model config \n"
    for k in vars(args):
        v = getattr(args, k)
        if hasattr(v, "teacher"):
            continue
        msg += f"{k}: {v} \n"
        
    with open(os.path.join(save_path, "teacher_config.txt"), "w") as f:
        f.write(msg)

    logging.info(msg)

    model_t = load_teacher(args)
    model_t.cuda(args.eval_gpu)
    model_s = models.__dict__['resnet18'](num_classes=args.num_classes).cuda(args.eval_gpu)

    # dataloader   
    train_loader, test_loader = get_distill_loaders_2(args)

    optimizer = torch.optim.SGD(model_s.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

    print(f"==> Training on GPU {args.eval_gpu}")
    logging.info(f"==> Training on GPU {args.eval_gpu}")

    tb_dir = os.path.join(save_path, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        args.epoch = epoch
        print(f"==> epoch {epoch}")
        logging.info(f"==> epoch {epoch}")
        model_s, data_dict = train(args, train_loader, test_loader, model_s, model_t, optimizer)
        all_results.append(data_dict)
        test_acc = data_dict["test acc"]

        ckpt_freq = args.epochs // 5 

        add_to_tensorboard(writer, data_dict, epoch)
    
        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(save_path, '{}_best.pth'.format(args.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % ckpt_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(save_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

            # This best accuracy is only for printing purpose.
            # The results reported in the paper/README is from the last epoch. 
            print('best accuracy:', best_acc)