#!/bin/bash
#oods = CIFAR100 SVHN LSUNCrop LSUNResize
# Imagenet30 , CUB SD
python run.py CIFAR10 --oods "CIFAR100 SVHN LSUNResize"\
        --operation ood_detection \
        --model_type contrastive --ood_detector metrics \
        --epochs 400 \
        --batch-size  256 --lr 0.01\
        --ckpt 199 \
        --num-cluster 768 --pcl-r 256 \
        --exp-number 1 --model_no 3 --ckpt_freq 20 --exp_dir /data/temiloluwa.adeoti/fourth_experiments \
