from easydict import EasyDict as edict

class ModelConfig():
    contrastive = None
    finetuned = None
    distilled = None

class OODConfig:
    model_config = ModelConfig()
    metrics = None
    ensemble = None
    odin = None

    def update_config(self, cfg, ood_detector):
        dic = getattr(self, ood_detector)
        for k, v in cfg.items():
            setattr(dic, k, v)
        setattr(self, ood_detector, dic)
    

CO = OODConfig()


CO.model_config.contrastive = edict({ 
    "encoder":["key"],
    "output_layer": ["avg_pool"],
    "ckpt": [199]
})


CO.model_config.finetuned = edict({
    "encoder":["key"],
    "output_layer": ["fc"],
    "model_no": [2],
    "label_percent": [0.1],
    "ft_ckpt":[600]
})


CO.model_config.distilled = edict({
    "encoder":["key", "query"],
    "output_layer": ["avg_pool"],
    "label_percent": [0.1],
    "model_t":["resnet50"],
    "model_s": ["resnet18"],
    "distill":["crd"],
    "gamma": [0],
    "alpha": [1],
    "beta": [0],
    "trial_no": [50],
    "use_best_ckpt":[False]
})


CO.model_config.distilled_kd = edict({
    "encoder":["key"],
    "use_best_ckpt":[True],
    "ds_ckpt":[800]
})


CO.metrics = edict({
    "eval_gpu": 2,
    "num_classes": 10,
    "train_le": True,
    "cache_le": False,
    "arch": "resnet50",
    "metrics": ["cosine", "mahalanobis"],
    "pca_coms": [0, 10, 100, 1000],
    "clusters": [0, 10, 20, 30, 40, 50],
    "cluster_method": ["kmeans"],
    "num_prototype_clusters": 0,
    "clip": 0,
    "rot_90": False,
    "aug_num": 3
})


CO.ensemble = edict({
    "num_classes": 100,
    "arch": "resnet50",
    "eval_gpu": 2,
    "num_thresholds":100,
    "center_crop": False,
    "temperature": [100, 50, 10, 5, 3, 2, 1]
})
