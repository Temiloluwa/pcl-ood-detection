from operator import ge
from numpy.core.fromnumeric import all
import os
import pandas as pd
import json
import faiss
import numpy as np
from ood import linear_evaluate, metrics_evaluate, get_auroc, get_ensemble_data, prep_for_auroc, odin_main \
      , prep_odin_for_auroc, cache_predictions
from utils import normalize, prepare_finetune_paths, save_pickle, save_results_excel, load_finetuned_model
from itertools import product
from pprint import pprint
from tqdm import tqdm
from pathlib import Path


def metrics_ood_detector(args):
    """ ood detection on contrastive features"""
    id_dataset = args.data
    metrics = args.ood.metrics
    all_results = []
    all_props = []
    all_clus_result = []
    all_new_auroc_res = []
    checkpoint_path = os.path.join(args.exp_root, \
        f"checkpoint_{metrics.ckpt}_results", "ood_results", "metrics_result")
    metrics.result_path =  result_path = os.path.join(checkpoint_path, \
        f"{args.model_type}_enc_{metrics.encoder}_ckpt_{metrics.ckpt}_{metrics.output_layer}")
    
    print(f"Using checkpoint: {metrics.ckpt}")
    
    if args.model_type in ["finetuned", "distilled"]:
        raise ValueError("Wrong Model Type")
    
    os.makedirs(result_path, exist_ok=True)
    kwargs = {"means": None, "im2cluster": None, "clip": None}
    
    for num, ood_dataset in enumerate(args.ood_datasets):
        id_feats_n_targs, ood_feats_n_targs, linear_eval_results = linear_evaluate(args, ood_dataset, result_path)
        itrain_X, itest_X, itrain_y, itest_y = id_feats_n_targs.train_features, \
                                                id_feats_n_targs.test_features,\
                                                id_feats_n_targs.train_targets,\
                                                id_feats_n_targs.test_targets
        otest_X = ood_feats_n_targs[1]
        print(f"IID Shapes: train {itrain_X.shape}, test {itest_X.shape}")
        print(f"IID Shapes: train target {itrain_y.shape}, test target {itest_y.shape}")
        print(f"OOD Shapes: test {otest_X.shape}")
        itrain_X = normalize(itrain_X)
        itest_X = normalize(itest_X)
        otest_X = normalize(otest_X)

        dim =  itrain_X.shape[1]
        
        if num == 0 and metrics.num_prototype_clusters:
            print("Using Prototypes")
            im2cluster, prototypes, _ = run_clustering(itrain_X, metrics.num_prototype_clusters)
            kwargs = {"means": prototypes, "im2cluster": im2cluster, "clip": metrics.clip}

        
        for met, com, num_clus, clus_met, global_mal_cov in \
            product(metrics.metrics, metrics.pca_coms, metrics.clusters, metrics.cluster_method, [True, False]):

            #if com == 0 and clus_met=="gmm":
            #    continue

            if met != "mahalanobis" and global_mal_cov is True:
                continue
    
            print(f"Metric: {met}, pca: {com}, clusters: {num_clus}, ood: {ood_dataset}")
            _result, _props, _clus_result, _new_auroc_res = metrics_evaluate(itrain_X, itest_X, itrain_y, itest_y,\
                                         otest_X, met, num_clus, com, clus_met, global_mal_cov, **kwargs)
            
            for res in [_result, _props, _clus_result, _new_auroc_res]:
                res["ood"] = ood_dataset
                res["idd"] = id_dataset
            _result["linear_results"] = linear_eval_results
            all_results.append(_result)
            all_props.append(_props)
            all_clus_result.append(_clus_result)
            all_new_auroc_res.append(_new_auroc_res)

    save_pickle(result_path, "results", all_results)
    save_pickle(result_path, "props", all_props)
    save_pickle(result_path, "clus_result", all_clus_result)
    save_pickle(result_path, "new_auroc", all_new_auroc_res)
    #save_results_excel(args, result_path)



def ensemble_ood_detector(args):
    ens = args.ood.ensemble
    ensemble_configs = ens.ensemble_configs
    args.center_crop = ens.center_crop

    def process_ft_path(_path):
        _path = _path.split("/")
        _root = "/"+"/".join(_path[1:8])
        _model_no = _path[8]
        _ckpt_no = _path[9]
        return _root, _model_no, _ckpt_no

    if args.model_type == "finetuned":
        def prepare_path(dic):
            ckpt_name = "{}_ckpt_{}_mn_{}_lp_{}/{}".format(
                dic.encoder, args.ckpt, dic.model_no, int(dic.label_percent*100), \
                    f"checkpoint_{conf.ft_ckpt:04d}.pth.tar"
            )
            return ckpt_name
            
        ckpt_paths = []
        id_dataset = args.data
        
        for conf in ensemble_configs:
            ckpt_path = os.path.join(args.checkpoint_path, "finetune", prepare_path(conf))
            ckpt_paths.append(ckpt_path)
        
    elif args.model_type == "distilled_kd":
        print("Evaluating Distilled Model")
        ckpt_paths = []
        for conf in ens.ensemble_configs:
            conf.ckpt = args.ckpt
            conf.model_no = args.model_no
            model_path = os.path.join(args.checkpoint_path, 'distilled_kd', f'{prepare_finetune_paths(conf)}')
            ds_ckpt = "resnet18_best.pth" if conf.use_best_ckpt else f"ckpt_epoch_{conf.ds_ckpt}.pth"
            ckpt_paths.append(os.path.join(model_path, ds_ckpt))

    ckpt_paths = list(set([i for i in ckpt_paths if os.path.exists(i)]))

    ensemble_results = []
    ood_results = []
    id_dataset = args.data
    for ood_dataset in args.ood_datasets:
        ens_data =  get_ensemble_data(args, ood_dataset, ckpt_paths)
        for temp in ens.temperature:
            _all_ckpts = []
            
            for idx, ckpt in enumerate(ckpt_paths):
                _root, _model_no, _ckpt = process_ft_path(ckpt)
                prepped_data = prep_for_auroc(args, temp, *ens_data, idx)
                cache_predictions(prepped_data, os.path.join(_root, _model_no, f"temp_{temp}"))
                auroc, tnr_at_tpr95 = get_auroc(*prepped_data)
                
                ood_results.append({
                    "id": id_dataset,
                    "ood": ood_dataset,
                    "temperature": temp,
                    "model_no": _model_no,
                    "ckpt": _ckpt,
                    "auroc": auroc,
                    "tnr@tpr95": tnr_at_tpr95
                })
                _all_ckpts.append(os.path.join(_model_no,_ckpt))
                pprint(ood_results[-1])
            
            if len(ckpt_paths) > 1:
                prepped_data = prep_for_auroc(args, temp, *ens_data, -1)
                auroc, tnr_at_tpr95 = get_auroc(*prepped_data)
                ensemble_results.append({
                        "id": id_dataset,
                        "ood": ood_dataset,
                        "temperature": temp,
                        "model_no": "ensemble",
                        "ckpt": _all_ckpts,
                        "auroc": auroc,
                        "tnr@tpr95": tnr_at_tpr95
                    })
                _all_ckpts = []
                pprint(ensemble_results[-1])


    existing_res = [i for i in Path(os.path.join(_root, _model_no)).iterdir() if "pickle" in str(i)]
    if existing_res:
        for i in existing_res:
            os.remove(i)
    
    save_pickle(os.path.join(_root, _model_no), f"single_{_ckpt}", ood_results)
    save_pickle(os.path.join(_root, _model_no), "ens", ensemble_results)
    
    """
    save_path = os.path.join(args.checkpoint_path, "ood_results", "ensemble_result")
    ensf = "./cache/ens_ood_results.json"
    singf = "./cache/single_ood_results.json"

    with open(singf, "r") as f:
        temp = json.load(f)
        temp[f"{'-'.join(_all_ckpts)}"] = ood_results
    
    with open(singf, "w") as f:
        json.dump(temp, f)

        with open(os.path.join(_root, _model_no, "single_ood_results.json"), "w") as g:
            json.dump(temp, g)
    
    if ensemble_results:
        with open(ensf, "r") as f:
            temp = json.load(f)
            temp[f"{'-'.join(_all_ckpts)}"] = ensemble_results
        
        with open(ensf, "w") as f:
            json.dump(temp, f)
    """


def odin_ood_detector(args):
    with open("./cache/odin_ood_results.json", "r") as f:
        all_results = json.load(f)

    all_results = []

    if args.model_type == "finetuned":
        odin = args.ood.odin
        temperatures = odin.temperature
        odin.data, odin.ckpt, odin.batch_size = args.data, args.ckpt, args.batch_size
        ckpt_name = os.path.join(args.checkpoint_path, "finetune", prepare_finetune_paths(args.ood.odin))
        ckpt_path = os.path.join(ckpt_name, f"checkpoint_{args.ood.odin.ft_ckpt:04d}.pth.tar")
        model = load_finetuned_model(args.ood.odin, ckpt_path)
        for ood_dataset in args.ood_datasets:
            for temp in temperatures:
                odin.temperature = temp
                prepped_data = prep_odin_for_auroc(odin, ood_dataset, model)
                auroc, tnr_at_tpr95 = get_auroc(*prepped_data)
                print(f"ood_dataset: {ood_dataset}, temperature: {temp}, auroc: {auroc}, tnr@tpr95: {tnr_at_tpr95}")
                all_results.append({
                        "id": args.data,
                        "ood": ood_dataset,
                        "temperature": temp,
                        "model_no": ckpt_path,
                        "auroc": auroc,
                        "tnr@tpr95": tnr_at_tpr95
                    })
        save_pickle("./cache", "odin_results.pickle", all_results)

    else:
        arg = args.ood.odin
        args.model_path = os.path.join(args.checkpoint_path, 'distilled', 'finetuned')
        args.model_name = 's_{}_t_{}_{}_gamma_{}_alpha_{}_beta_{}_trial_{}'.format(arg.model_s, arg.model_t, arg.distill,
                                                                arg.gamma, arg.alpha, arg.beta, arg.trial_no)
        ds_ckpt = "resnet18_best.pth" if arg.use_best_ckpt else f"ckpt_epoch_{arg.ds_ckpt}.pth"
        ckpt_path = os.path.join(args.model_path, args.model_name, ds_ckpt)        
    
    """
    ood_results = []
    for ood_dataset in args.ood_datasets:
        for temp in args.temperature:
            print(f"Using temperature {temp}")
            args.temp = temp
            res = odin_main.main(args, ood_dataset, ckpt_path)
            res["ood"] = ood_dataset
            res["temperature"] = temp
            ood_results.append(res)
    
    all_results[f"{ckpt_name}"] = ood_results
    with open("./cache/odin_ood_results.json", "w") as f:
        json.dump(all_results, f)
"""


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


def run_clustering(data, num_cluster):
    clus_result = run_kmeans(data, num_cluster)
    im2cluster = np.array(clus_result['im2cluster']).flatten()
    prototypes = np.array(clus_result['centroids'][0])
    density = np.array(clus_result['density']).flatten()
    return im2cluster, prototypes, density