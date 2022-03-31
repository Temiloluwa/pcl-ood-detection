class ContrastiveConfig():
    pass

CC = ContrastiveConfig()
# network architecture
CC.arch = "resnet50"
# number of workers
CC.workers = 4
CC.start_epoch = 0
CC.schedule =  [120, 160]
CC.momentum = 0.9
CC.weight_decay = 1e-4
CC.print_freq = 5
CC.resume = True
CC.world_size = 1
CC.rank = 0
CC.dist_url = "tcp://localhost:10005"
CC.dist_backend = "nccl"
CC.seed = None
CC.multiprocessing_distributed = True
CC.gpu = None
CC.dim = 512
CC.moco_m = 0.999
CC.temperature = 0.2
CC.mlp = True
CC.mlp2 = False
CC.aug_plus = True
CC.cos = True
CC.warmup_epoch = 20
CC.input_image_width = 224
CC.input_image_height = 224
CC.pretrained = True