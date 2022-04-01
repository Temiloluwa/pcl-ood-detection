import argparse
import os
import sys
from contrastive import contrastive_main, CC
#from distilled import distilled_main, distill_kd_main, distill_con_main, CD
#from finetuned import finetune_main, CF
#from ood import CO
#from ood_detection import metrics_ood_detector, ensemble_ood_detector, odin_ood_detector
from utils import gen_configs

parser = argparse.ArgumentParser()

parser.add_argument('data', metavar='DIR', help='in distribution dataset')

parser.add_argument('--oods', type=str, default="CIFAR100", \
    help='comma seperated string of out of distribution datasets')

parser.add_argument('--operation', choices=['train_model', 'ood_detection'], \
    help='train a model (contrastive, finetuned or distilled model) or perform ood detection ')

parser.add_argument('--ood_detector', choices=['metrics', 'ensemble', 'odin'], \
    default="metrics", help='type of ood detection method')

parser.add_argument('--model_type', choices=['contrastive', 'finetuned', 'distilled'],
                 help='type of model that can be trained')

parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('-bs', '--batch-size', default=256, type=int)

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float)

parser.add_argument('--pcl-r', default=2560, type=int,\
    help='queue size; number of negative pairs; needs to be smaller than num_cluster')
            
parser.add_argument('--num-cluster', default="2816", type=str, help='number of prototypical clusters')

parser.add_argument('--exp-number', type=int, help='contrastive experiment number')

parser.add_argument('--exp_dir', type=str, default="/data/temiloluwa.adeoti/fourth_experiments/",\
    help="root directory of all experiment data")

parser.add_argument('--ckpt', type=int, default=199, \
    help="contrastive model experiment ckpt to perform analysis on ")

parser.add_argument('--model_no', type=int, default=0, help='finetune model number')

parser.add_argument('--trial_no', type=int, default=0, help='distilled model number')

parser.add_argument('--ckpt_freq', type=int, default=20)


args = parser.parse_args()
args.ood_datasets = [i.strip(" ") for i in args.oods.split(" ")]
args.num_cluster = [i.strip(" ") for i in args.num_cluster.split(" ")]
args.exp_root = os.path.join(args.exp_dir,
                            f"{args.data}_clus_{'_'.join(args.num_cluster)}_neg_{args.pcl_r}", 
                            f"exp_{args.exp_number}")
args.checkpoint_path = os.path.join(args.exp_root, f"checkpoint_{args.ckpt}_results")


if __name__ == '__main__':
    if args.operation == "train_model":
        if args.model_type == "contrastive":
            contrastive_main(args, CC)

        """
        elif args.model_type == "finetuned":
            args = update_args(args, CF)
            finetune_main(args)
        
        elif args.model_type == "distilled":
            args = update_args(args, CD)
            distilled_main(args)
        
        elif args.model_type == "distilled_kd":
            args = update_args(args, CD)
            distill_kd_main(args)
        
        elif args.model_type == "distilled_con":
            args = update_args(args, CD)
            distill_con_main(args)
        """
         
    elif args.operation == "ood_detection":
        model_config = getattr(CO.model_config, args.model_type)
        configs = gen_configs(model_config)
       
        for cfg in configs:
            CO.update_config(cfg, args.ood_detector)
            args.ood = CO
          
            try:
                # metrics on contrastive models
                if args.ood_detector == "metrics":
                    metrics_ood_detector(args)

                # odin on finetuned models
                elif args.ood_detector == "odin":
                    odin_ood_detector(args)
            except:
                print(sys.exc_info())
                continue
        
        else:
            # ensemble detection on finetuned models
            if args.ood_detector == "ensemble":
                args.ood = CO
                setattr(args.ood.ensemble, "ensemble_configs", configs)
                ensemble_ood_detector(args)

  