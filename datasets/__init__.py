from contrastive.pcl.loader import CIFAR100Instance, CIFAR10Instance
from .constrastive_datasets import \
        train_dataset_dict, test_dataset_dict, target_train_dataset, target_test_dataset

       
from .distill_datasets import get_distill_testset, get_distill_trainset, \
        distill_test_transform_dict, distill_train_transform_dict
