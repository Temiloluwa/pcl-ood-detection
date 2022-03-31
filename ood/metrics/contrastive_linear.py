import os
import numpy as np
from utils import load_contrastive_model, cache_contrastive_features, save_features
from utils import compute_features, train_linear_model, Feats_n_targs, detach, load_finetuned_model, load_distilled_model
from datasets.get_data import get_contrastive_loaders, get_contrastive_targets
from torch.nn.functional import normalize
from torchvision import models 

def linear_evaluate(args, ood_dataset, result_path):
    
    
    def save_accuracy(path_, dataset, msg):
        output_layer = args.ood.metrics.output_layer
        with open(os.path.join(path_ , f'{dataset}_{output_layer}_accuracies.txt'), "w") as f:
            f.write(msg)
    
    cf = cache_contrastive_features
    save_path = os.path.join(result_path, "cache")
    test_accuracy = None
    args.ood.metrics.data = args.data
    args.rot_90 = args.ood.metrics.rot_90
    args.aug_num = args.ood.metrics.aug_num

    os.makedirs(save_path, exist_ok=True)
    print(f"iid dataset: {args.data}")

    if args.model_type == "contrastive":
        checkpoint_path = os.path.join(args.exp_root, "checkpoints")
        args.ood.metrics.arch = args.arch
        model = load_contrastive_model(checkpoint_path, args.ood.metrics, args.dim)
        #model = models.__dict__[args.arch](pretrained=True).cuda(args.ood.metrics.eval_gpu)
    
    model.eval()
    train_loader, test_loader = get_contrastive_loaders(args, args.data)
    train_features = compute_features(train_loader, model)
    test_features = compute_features(test_loader, model)
    train_targets, test_targets = get_contrastive_targets(args.data)
    linear_eval_res = None
    
    if args.ood.metrics.train_le:
        _, train_accuracy, test_accuracy = \
            train_linear_model(args, train_features, train_targets, test_features, test_targets)
        msg = f"train accuracy: {train_accuracy:.2f}% , test accuracy: {test_accuracy:.2f}%"
        save_accuracy(save_path, args.data, msg)
        linear_eval_res = {
            "data_set": args.data,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        }


    id_feats_n_targs = Feats_n_targs(train_features, test_features,\
                                     detach(train_targets), detach(test_targets))
 
    if args.ood.metrics.cache_le:
    # cache id features
        cf(args, args.data, save_path, train_features, test_features, train_targets, test_targets)

    print(f"ood dataset: {ood_dataset}")
 
    train_loader, test_loader = get_contrastive_loaders(args, ood_dataset)
    train_features = compute_features(train_loader, model)
    test_features = compute_features(test_loader, model)
    train_targets, test_targets = get_contrastive_targets(ood_dataset)
    

    # cache ood features
    if args.ood.metrics.cache_le:
        cf(args, ood_dataset, save_path, train_features, test_features, train_targets, test_targets)
    
    ood_feats_n_targs = Feats_n_targs(detach(train_features), detach(test_features),\
                                     detach(train_targets), detach(test_targets))
    
    return id_feats_n_targs, ood_feats_n_targs, linear_eval_res



def linear_evaluate_finetune(args, model, use_hook):
    ood_dataset = args.ood
    id_dataset = args.data
    print(f"Computing features for iid: {id_dataset}")
    train_loader, test_loader = get_contrastive_loaders(args, id_dataset)
    avg_train_features, fc_train_features = compute_features(train_loader, model, use_hook)
    avg_test_features, fc_test_features = compute_features(test_loader, model, use_hook)
    train_targets, test_targets = get_contrastive_targets(id_dataset)
    avg_id_feats_n_targs = Feats_n_targs(detach(normalize(avg_train_features)),\
                                     detach(normalize(avg_test_features)),\
                                    detach(train_targets), detach(test_targets))

    fc_id_feats_n_targs = Feats_n_targs(detach(normalize(fc_train_features)),\
                                     detach(normalize(fc_test_features)),\
                                    detach(train_targets), detach(test_targets))

    print(f"Computing features for ood: {ood_dataset}")
    _, ood_test_loader = get_contrastive_loaders(args, ood_dataset)
    avg_ood_test_features, fc_ood_test_features = compute_features(ood_test_loader, model, use_hook)
    avg_ood_test_features = detach(normalize(avg_ood_test_features))
    fc_ood_test_features = detach(normalize(fc_ood_test_features))
    output = [(avg_id_feats_n_targs, avg_ood_test_features), (fc_id_feats_n_targs, fc_ood_test_features)]
    return output
