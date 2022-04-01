#!/bin/bash  
data_dir="/data/datasets"
cifar_10_dir="$data_dir/CIFAR10"
cifar_100_dir="$data_dir/CIFAR100"
svhn_dir="$data_dir/svhn-data"
lsun_crop_dir="$data_dir/LSUN_datasets/LSUN"
#imagenet30_dir="$datadir/ImageNet30"
datasets_dirs=($cifar_10_dir $cifar_100_dir $svhn_dir $lsun_crop_dir)

# link sources: https://github.com/facebookresearch/odin
lsuncrop_link=https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
lsunresize_link=https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz

# make root directory
mkdir -p $data_dir && chmod 777 "/data"

# make directories
for dir_ in ${datasets_dirs[@]}; do
    mkdir -p $dir_
done

# download lsun crop dataset
cd "$data_dir/LSUN_datasets"
wget $lsuncrop_link
tar zxvf LSUN.tar.gz
rm LSUN.tar.gz

# download lsun resize dataset
wget $lsunresize_link
tar zxvf LSUN_resize.tar.gz
rm LSUN_resize.tar.gz

cd "$data_dir/LSUN_datasets" && \
        mv LSUN_resize temp  &&  \
        mv "/data/datasets/LSUN_datasets/temp/LSUN_resize" "/data/datasets/LSUN_datasets/LSUN_resize" &&
        rm -rd temp

# download imagenet 30 dataset