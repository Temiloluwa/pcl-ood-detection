import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import namedtuple
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from collections import Counter
from functools import partial
from pprint import pprint


class OodEvaluator:
    
    dist_mt = None
    train_pred_label = None
    test_pred_label = None  
    pred_test_label_type = None
    in_train_df = None
    cov_inv = None
    pca_com = None
    auroc = None
    aupr = None
    tnr_at_tpr95 = None
    cov_inv_0 = None
    cp = None
    gmm_results = {"gmm_default": None, "gmm_max_prob": None, "gmm_weighted_max_prob": None}
    cluster_eval_result = {}
    new_auroc_result = {}

    """
    OOD Evaluator performs ood detection on ID and OD features
    using distance metrics
    
    - Initialize OoD detector using ID train and test features and targets
    - Perform OOD detection using mahalanobis, euclidean or cosine measures
    - Ensure all Input features are normalized

    e.g

    # initialize
    oe = OodEvaluator(id_train_X, id_test_X, id_train_y, id_test_y, 30, 10)

    # compute distances
    oe(od_test_X, "mahalanobis").get_scores()

    # get AUROC and tnr@tpr95
    oe.get_auroc()

    """

    
    def __init__(self, 
                in_train_X,
                in_test_X,
                in_train_y,
                in_test_y,
                num_clusters = 30,
                pca_com = 10,
                cluster_method = "kmeans",
                means = None,
                im2cluster = None,
                clip = None,
                clip_metric = "cosine"
                ):
        
        """ Initialize Evaluator

        Args:
            in_train_X (np.ndarray): in distribution train features
            in_test_X (np.ndarray):  in distribution test features
            in_train_y (np.ndarray): in distribution train ground truths
            in_test_y (np.ndarray):  in distribution test ground truths
            num_clusters (int): num of clusters
            pca_com (int): number of pca components.
            cluster_method (string): kmeans or gmm
            means (np.ndarray, optional): in distribution prototypes/centriods 
            im2cluster (np.ndarray, optional): cluster assignment of indistribution data based on means (prototypes/centriods)
            clip (float, optional): value between 0 and 1 to clip means (prototypes/centroids)
            clip_metric (string, optional): cosine or euclidean to reassign samples whose clusters are clipped
        """
        if means is not None:
            # if means are supplied no PCA or clustering is perfomed
            print("setting pca and num_cluster to 0 because means are supplied")
            pca_com = 0
            num_clusters = 0

        # transform id features if PCA com is not 0
        self.in_train_X, self.in_test_X = \
            self._transform_feats(in_train_X, in_test_X, pca_com)
        self._ground_truth_train_targets = in_train_y.astype("int32")
        self.in_train_y = in_train_y.astype("int32")
        self.in_test_y = in_test_y.astype("int32")
        self.num_clusters = num_clusters
        self.gt_protoypes =  True if num_clusters == 0 else False
        self.pca_com = pca_com
        self.cluster_method = cluster_method

        if means is not None:
            self._means = means
            self._starting_means  = means
            print("using supplied means")
            self.in_train_y = im2cluster
        else:
            if not self.gt_protoypes:
                self._means = self._cluster()
                self._starting_means = self._means

        self._starting_X = self.in_train_X
        self._starting_y = self.in_train_y

        # clip means if clip is fraction is supplied  
        if not self.gt_protoypes and clip:
            print("clipping means")
            self.in_train_y, self._means, self.in_train_X =  \
                                    self.clip_means(clip, 
                                                self._means, 
                                                self.in_train_y, 
                                                self.in_train_X, 
                                                metric=clip_metric)
            print(f"new train data shape: {self.in_train_X.shape}")
        
        self.in_classes = np.unique(self.in_train_y).astype(int)
        self.class_idx = [np.where(self.in_train_y == i)[0] for i in self.in_classes]
        self.in_classes_X = [self.in_train_X[idx] for idx in self.class_idx]

        if self.gt_protoypes and means is None:
            self._means = np.stack([np.mean(X, axis=0) for X in self.in_classes_X])

        assert self._means.shape == (len(self.in_classes), self.in_train_X.shape[1])
            


    def reclip(self, clip, clip_metric=None):
        self.in_train_y, self._means, self.in_train_X =  \
                                    self.clip_means(clip, 
                                                    self._starting_means, 
                                                    self._starting_y, 
                                                    self._starting_X, 
                                                    metric=clip_metric)
        self.in_classes = np.unique(self.in_train_y).astype(int)
        self.class_idx = [np.where(self.in_train_y == i)[0] for i in self.in_classes]
        self.in_classes_X = [self.in_train_X[idx] for idx in self.class_idx]
        print(f"new train data shape: {self.in_train_X.shape}")
    


    def _transform_feats(self, in_train_X, in_test_X, pca_com):
        if pca_com:
            self.pca = PCA(n_components=pca_com)
            in_train_X = self.pca.fit_transform(in_train_X)
            in_test_X = self.pca.fit_transform(in_test_X)
        
        return in_train_X, in_test_X


    def _cluster(self):
        if self.cluster_method == "kmeans":
            print(f"Performing Kmeans clustering to {self.num_clusters} clusters")
            kmeans = KMeans(n_clusters=self.num_clusters, init="k-means++", random_state=5).fit(self.in_train_X)
            self.in_train_y = kmeans.labels_
            self.in_test_y = kmeans.predict(self.in_test_X)
            in_classes_mean = kmeans.cluster_centers_
        else:
            print(f"Performing clustering using GMM to {self.num_clusters} clusters")
            self.gmm = GaussianMixture(n_components=self.num_clusters, 
                                        n_init=3, covariance_type="full").fit(self.in_train_X)
            self.in_train_y = self.gmm.predict(self.in_train_X)
            self.in_test_y = self.gmm.predict(self.in_test_X)
            in_classes_mean = self.gmm.means_
    
        return in_classes_mean

    
    def __call__(self, out_X, distance_metric, **kwargs):

        self.dist_mt = distance_metric
        self._calc_distances(out_X, **kwargs)
        
        return self

    
    def euclidean(self, x, support_mean):
        n, d = x.shape
        m = len(support_mean)
        assert d == support_mean.shape[1], "feature dimensions don't match"
        x = np.expand_dims(x, 1)
        support_mean = np.expand_dims(support_mean, 0)
        dist = np.sqrt(np.sum(np.power(x - support_mean, 2), axis=2))   
        assert dist.shape == (n, m), "error occured"
        
        return  dist

    
    def cosine(self, x, support_mean):
        n = x.shape[0]
        m = support_mean.shape[0]
        sim = cosine_similarity(x, support_mean)

        assert sim.shape == (n,m), "error occured"

        return sim


    def mhl(self, u, v, icov):
        diff = u-v
        left = np.dot(diff, icov)
        return np.einsum('ij,ij -> i', left, diff)


    def rel_mahalanobis(self, x, support_mean):
        n, m = len(x), len(support_mean)
        dist = np.zeros((n, m))
        total_means = (np.mean(self.in_train_X, axis=0)).reshape((1, -1))
        
        if not np.all(self.cov_inv_0):
            self.cov_inv_0 = np.linalg.inv(np.cov(self.in_train_X.T))

        for i in tqdm(range(m)):
            _y = total_means
            _d = self.mhl(x, _y, self.cov_inv_0)
            dist[:, i] = _d

        mal_dist = self.mahalanobis(x, support_mean)
        rmd_dist = mal_dist - dist
        
        return rmd_dist

    def inverse(self, x, choice):
        if choice == "svd":
            u,s,v=np.linalg.svd(x)
            return np.dot(v.transpose(),np.dot(np.diag(s**-1),u.transpose()))
        
        elif choice == "identity":
            idd = np.identity(x.shape[0])
            return np.linalg.solve(x, idd)
        else:
            return np.linalg.inv(x)
            
    def mahalanobis(self, x, support_mean, global_cov=None, inv_choice="default", recal=False):
        n, m = len(x), len(support_mean)
        dist = np.zeros((n, m))
        inverse = partial(self.inverse, choice=inv_choice)
        
        if not np.all(self.cov_inv) or recal:
            recalculate = True
        else:
            recalculate = False

        if recalculate:
            if self.cluster_method == "kmeans":
                if global_cov:
                    print("global conv")
                    cov_inv = inverse(np.cov(self.in_train_X.T))
                    self.cov_inv = [cov_inv] * m
                else:
                    self.cov_inv = [inverse(np.cov(feat.T)) \
                            for feat in self.in_classes_X]
            else:
                self.cov_inv = self.gmm.precisions_
            
      
        for i in tqdm(range(m)):
            _y = support_mean[i].reshape((1, -1))
            _d = self.mhl(x, _y, self.cov_inv[i])
            dist[:, i] = _d
        
        return dist


    def _calc_distances(self, out_X, **kwargs):
        if self.pca_com:
            out_X = self.pca.fit_transform(out_X)

        if self.gt_protoypes:
            print("** Using Ground Truths **")
        
        if not self.pca_com:
            print(f"** No PCA **")

        dist_metric = getattr(self, self.dist_mt)
        print("calculating train distances")
        self.train_dist = dist_metric(self.in_train_X, self._means, **kwargs)
        print("calculating test distances")
        self.test_dist = dist_metric(self.in_test_X, self._means, **kwargs)
        print("calculating ood distances")
        self.out_dist = dist_metric(out_X, self._means, **kwargs)

        
        if self.cluster_method == "gmm" and not self.gt_protoypes:

            in_score = self.gmm_default_in_score = self.gmm.score_samples(self.in_test_X).reshape(-1, 1)
            out_dist_score = self.gmm_default_out_dist_score = self.gmm.score_samples(out_X).reshape(-1, 1)
            in_ood_samples = np.squeeze(np.vstack((in_score, out_dist_score)))
            labels = np.zeros(len(in_score) + len(out_dist_score), dtype=np.int32)
            labels[:len(in_score)] = 1
            self.gmm_results["gmm_default"] = roc_auc_score(labels, in_ood_samples)
            print(f"Default AUROC: {self.gmm_results['gmm_default']}")
        
            gmm = self.gmm
            max_cluster = np.argmax(gmm._estimate_log_weights())
            print('Max Cluster : ', max_cluster)
            in_score = np.max(gmm._estimate_log_prob(self.in_test_X), axis=1).reshape(-1, 1)
            out_dist_score = np.max(gmm._estimate_log_prob(out_X), axis=1).reshape(-1, 1)
            in_dist_lbl = np.argmax(gmm._estimate_log_prob(self.in_test_X), axis=1)
            out_dist_lbl = np.argmax(gmm._estimate_log_prob(out_X), axis=1)
            in_ood_samples = np.squeeze(np.vstack((in_score, out_dist_score)))
            labels = np.zeros(len(in_score) + len(out_dist_score), dtype=np.int32)
            labels[:len(in_score)] = 1
            self.gmm_results["gmm_max_prob"] = roc_auc_score(labels, in_ood_samples)
            
            print(f"Max Probability AUROC: {self.gmm_results['gmm_max_prob']}")
      
            in_score = np.max(gmm._estimate_weighted_log_prob(self.in_test_X), axis=1).reshape(-1, 1)
            out_dist_score = np.max(gmm._estimate_weighted_log_prob(out_X), axis=1).reshape(-1, 1)
            in_dist_lbl = np.argmax(gmm._estimate_weighted_log_prob(self.in_test_X), axis=1)
            out_dist_lbl = np.argmax(gmm._estimate_weighted_log_prob(out_X), axis=1)
            in_ood_samples = np.squeeze(np.vstack((in_score, out_dist_score)))
            labels = np.zeros(len(in_score) + len(out_dist_score), dtype=np.int32)
            labels[:len(in_score)] = 1
            self.gmm_results["gmm_weighted_max_prob"] = roc_auc_score(labels, in_ood_samples)
            print(f"Weighted max Probability  AUROC: {self.gmm_results['gmm_weighted_max_prob']}")    
    

    def get_scores(self, rescale=False):

        def scaled(x):
            ma = np.max(self.train_pred_score)
            mi = np.min(self.train_pred_score)
            return (x - mi)/(ma - mi)
        
        if self.dist_mt is None:
            raise NameError('Supply distance metric to instance')
        
        get_index = np.argmax if self.dist_mt == "cosine" else np.argmin
        self.train_pred_label = get_index(self.train_dist, axis=1)
        self.test_pred_label = get_index(self.test_dist, axis=1)
        self.out_pred_label = get_index(self.out_dist, axis=1)
        
        self.train_pred_score = self.train_dist[np.arange(len(self.train_dist)), self.train_pred_label]
        self.test_pred_score = self.test_dist[np.arange(len(self.test_dist)), self.test_pred_label]
        self.out_pred_score = self.out_dist[np.arange(len(self.out_dist)), self.out_pred_label]

        if rescale:
            print("rescaling")
            self.test_pred_score = scaled(self.test_pred_score)
            self.out_pred_score = scaled(self.out_pred_score)

        self.pred = np.concatenate([self.test_pred_label, self.out_pred_label])
        self.pred_scores = np.concatenate([self.test_pred_score, self.out_pred_score])
        self.ood_ground_truths = np.concatenate([np.zeros(len(self.test_pred_label)), np.ones(len(self.out_pred_label))])
                
        print(f"train ID accuracy: {(np.mean(self.train_pred_label == self.in_train_y) * 100):.2f}%")
        print(f"test ID accuracy: {(np.mean(self.test_pred_label == self.in_test_y) * 100):.2f}%")
        return self


    def get_auroc(self, num_thresholds=100):
        group_scores = [self.train_pred_score[i] for i in self.class_idx]
        n_min_max_scores = [(np.min(dat), np.max(dat)) for dat in group_scores]

        if self.dist_mt == "cosine":
            e_min_max_scores = [(0, 1) for dat in group_scores]
        else:
            e_min_max_scores = [(0, np.max(dat)) for dat in group_scores]
        
        aurocs = []
        tnrs = []
        for min_max_scores in [n_min_max_scores, e_min_max_scores]:
            metrics_results = []
            # create min to max thresholds for each class - shape (num_classes, num_thresholds)
            threshold_ranges = np.array([list(np.linspace(i[0], i[1], num_thresholds)) for i in min_max_scores])
            for thresh in range(num_thresholds):
                # select the appropriate threshold for the predicted class corresponding to the test and ood data
                threshold_scores = np.array([threshold_ranges[:, thresh][i] for i in self.pred])
                ood_prediction = (self.pred_scores > threshold_scores) if self.dist_mt != "cosine" else \
                                    (self.pred_scores < threshold_scores)
                tn, fp, fn, tp = metrics.confusion_matrix(self.ood_ground_truths , ood_prediction).ravel()
                metrics_results.append({"tpr": tp/(tp+fn), "fpr": fp/(fp+tn), "tnr": tn/(tn+fp)})


            confusion_df = pd.DataFrame(metrics_results)
            self.confusion_df = confusion_df
            df_auroc = confusion_df.sort_values(by=['fpr'], ascending=False)
            auroc = metrics.auc(df_auroc['fpr'], df_auroc['tpr'])
            auroc = auroc * 100
            aurocs.append(auroc)

            df_tpr95 = confusion_df.sort_values(by=['tpr'], ascending=False)
            if len(df_tpr95[df_tpr95.tpr >= 0.95]) > 0:
                tnr_at_tpr95 = df_tpr95[df_tpr95.tpr >= 0.95].iloc[-1]['tnr']
                tnr_at_tpr95 = tnr_at_tpr95 * 100
            else:
                tnr_at_tpr95 = 0
            
            tnrs.append(tnr_at_tpr95)
    
        gt = (self.ood_ground_truths == 1).astype(float) if self.dist_mt != "cosine" else \
            (self.ood_ground_truths == 0).astype(float)
        auroc = roc_auc_score(gt, self.pred_scores) * 100

        aurocs.append(auroc)
        print(f"n auroc: {aurocs[0]}, tnr@tpr95: {tnrs[0]}")
        print(f"e auroc: {aurocs[1]}, tnr@tpr95: {tnrs[1]}")
        print(f"sklearn-auroc: {aurocs[2]} %")
        self.auroc = aurocs
        self.tnr_at_tpr95 = tnrs


    def clip_means(self, clip, centroids, clus_assign, feat, metric=None):
        clip = int((1-clip) * len(np.unique(clus_assign)))
        self.assgn_before_clip = [k for i, (j, k)  in enumerate(Counter(clus_assign).most_common())]
        new_assign = {i:j for i, (j, k)  in enumerate(Counter(clus_assign).most_common(clip))}
        old2new = {v:k for k,v in new_assign.items()}
        print(f"old cluster number: {len(centroids)}, new cluster counts: {len(new_assign)}")
        new_centroids = np.zeros((len(new_assign), centroids.shape[1]))

        for k, v in new_assign.items():
            new_centroids[k] = centroids[v]

        new_clus_assign = np.full(len(clus_assign), -20)
        unclustered_idx = []
        clustered_idx = []
        clustered_targ = []
        for i, v in enumerate(clus_assign):
            if v in new_assign.values():
                new_clus_assign[i] = old2new[v]
                clustered_idx.append(i)
                clustered_targ.append(old2new[v])
            else:
                unclustered_idx.append(i)

        if metric is not None:
            print("keeping centriods")
            unclustered_feat = feat[unclustered_idx]

            assert len(np.where(np.array(new_clus_assign) == -20)[0]) == len(unclustered_idx), "error"
            self.assgn_just_before_clip = new_clus_assign.copy()
            self.assgn_just_before_clip = [v for k,v in Counter(self.assgn_just_before_clip).most_common() if k!= -20]

            new_clus_assign[unclustered_idx] = \
                        self.cosine(unclustered_feat, new_centroids).argmax(axis=1) \
                        if metric == "cosine" else self.euclidean(unclustered_feat, new_centroids).argmin(axis=1)\
            
            self.assgn_after_clip = [v for k,v in Counter(new_clus_assign).most_common()]
            assert np.sum(new_clus_assign == -20) == 0, "error"
        else:
            print("discarding centriods")
            feat = feat[clustered_idx]
            in_classes = np.unique(clustered_targ).astype(int)
            class_idx = [np.where(clustered_targ == i)[0] for i in in_classes]
            in_classes_X = [feat[idx] for idx in class_idx]
            new_centroids = np.stack([np.mean(X, axis=0) for X in in_classes_X])
            new_clus_assign = np.array(clustered_targ) 

        return new_clus_assign, new_centroids, feat

    
    def get_cluster_purity(self, clus_targets, gt_targets):
        k_classes = np.unique(clus_targets).astype(int)
        k_class_idx = [np.nonzero(np.equal(cls_, clus_targets)) for cls_ in k_classes]
        in_classes_in_k_clstr = [gt_targets[idx] for idx in k_class_idx]
        purity_list = []

        for cluster_k in in_classes_in_k_clstr:
            unique, counts = np.unique(cluster_k, return_counts=True)
            purity_list.append(np.round(np.asarray(counts).max()/len(cluster_k), 5))

        return purity_list
    
    
    def get_cluster_evaluation_metrics(self):
        gt_labels = self._ground_truth_train_targets 
        pred_labels = self.in_train_y
        print("cluster purity")
        cp = self.get_cluster_purity(pred_labels, gt_labels)
        print("homogeneity score")
        hm = metrics.homogeneity_score(gt_labels, pred_labels)
        print("completeness score")
        cmp = metrics.completeness_score(gt_labels, pred_labels)
        print("v measure score, beta = 1")
        vm = metrics.v_measure_score(gt_labels, pred_labels)
        print("adjusted rand score")
        ari = metrics.adjusted_rand_score(gt_labels, pred_labels)
        print("adjusted mutual info score")
        ami = metrics.adjusted_mutual_info_score(gt_labels, pred_labels)
        clus = len(np.unique(self.in_train_y))

        if clus > 1:
            print("silohuette - cosine")
            sil_cos = metrics.silhouette_score(self.in_train_X, pred_labels, metric='cosine')
            print("silohuette - mahalanobis")
            #sil_mal = metrics.silhouette_score(self.in_train_X, pred_labels, metric='mahalanobis')
            sil_mal = 0
            print("davis boulding")
            db_score = metrics.davies_bouldin_score(self.in_train_X, pred_labels)
            print("calinski harabasz")
            ch_score = metrics.calinski_harabasz_score(self.in_train_X, pred_labels)
        else:
            sil_cos = 0
            sil_mal = 0
            db_score = 0
            ch_score = 0

        self.cluster_eval_result = {
            "train_pred_label": self.train_pred_label,
            "test_pred_label": self.test_pred_label,
            "out_pred_label": self.out_pred_label,
            "train_pred_score": self.train_pred_score,
            "test_pred_score": self.test_pred_score,
            "out_pred_score": self.out_pred_score,
            "cp":cp,
            "hm": hm,
            "cmp": cmp,
            "vm": vm,
            "ari": ari,
            "ami": ami,
            "sil_cos": sil_cos,
            "sil_mal": sil_mal,
            "db_score": db_score,
            "ch_score": ch_score
        }
    

    def calculate_new_aurocs(self):

        def add_scores_to_df(df, df_ref) -> pd.DataFrame:
            add_global_scores_to_df2(df, df_ref)
            add_cluster_scores_to_df2(df, df_ref)
            return df

        def add_global_scores_to_df2(df, df_ref) -> pd.DataFrame:
            df['global_train'] = generate_global_scores2(df=df, df_ref=df_ref, ref_data='id_train')
            df['global_test'] = generate_global_scores2(df=df, df_ref=df_ref, ref_data='id_test')
            return df

        def add_cluster_scores_to_df2(df, df_ref) -> pd.DataFrame:
            # generate and add cluster scores based on training and test reference data
            df['cluster_train'] = generate_cluster_scores2(df_eval=df, df_ref=df_ref, ref_data='id_train')
            df['cluster_test'] = generate_cluster_scores2(df_eval=df, df_ref=df_ref, ref_data='id_test')
            return df

        def generate_global_scores2(df, df_ref, ref_data) -> pd.Series:
            # generate global score for an entire data frame (row by row) - more efficient
            d = np.sort(df_ref[df_ref.data==ref_data].distance)
            return df.apply(lambda row: inverse_quantile_sorted(d, row.distance), axis=1)


        def generate_cluster_scores2(df_eval : pd.DataFrame,
                             df_ref : pd.DataFrame,
                             ref_data : str) -> pd.Series:
            # prepare sorted distances for each cluster
            distances = {}
            for c in df_eval.label.unique():
                distances[c] = np.sort(df_ref[(df_ref.data == ref_data) & (df_ref.label == c)].distance)
            
            # now generate the scores
            return df_eval.apply(lambda row: inverse_quantile_sorted(distances[row.label], row.distance), 
                                axis=1)

        def inverse_quantile_sorted(data, sample):
            return np.searchsorted(data, sample, side='right') / len(data)

        def get_outlier_scores():
            dat_train = np.array([self.train_pred_score, self.train_pred_label]).T
            dat_train = pd.DataFrame(data=dat_train, columns=["distance", "label"])
            dat_train["data"] = "id_train"

            dat_test = np.array([self.test_pred_score, self.test_pred_label]).T
            dat_test = pd.DataFrame(data=dat_test, columns=["distance", "label"])
            dat_test["data"] = "id_test"

            dat_out = np.array([self.out_pred_score, self.out_pred_label]).T
            dat_out = pd.DataFrame(data=dat_out, columns=["distance", "label"])
            dat_out["data"] = "ood"

            da = pd.concat([dat_train, dat_test, dat_out], axis=0)
            return da

        def prepare_eval_set(df: pd.DataFrame) -> pd.DataFrame:
            df_eval = df[df.data!='id_train'].copy()
            df_eval['is_ood'] = np.where(df_eval.data=='ood', 1, 0)
            return df_eval
        

        def calculate_aurocs(df, true_col='is_ood', score_cols=['distance', 'global_train', 'global_test', 'cluster_train', 'cluster_test']) -> dict:
            result = {}
            for s in score_cols:
                result[s] = metrics.roc_auc_score(y_true=df[true_col], y_score=df[s])
            return result
        
        da = get_outlier_scores()
        da_eval = prepare_eval_set(da)
        da_eval = add_scores_to_df(da_eval, da)
        self.new_auroc_result = calculate_aurocs(da_eval)
        pprint(self.new_auroc_result)

        


def metrics_evaluate(itrain_X, itest_X, itrain_y, itest_y, otrain_X, met, prot, com, cluster_method, global_mal_cov, **kwargs):
      
    ood_evaluator = OodEvaluator(itrain_X,
                                 itest_X, 
                                 itrain_y, 
                                 itest_y, 
                                 prot, 
                                 com, 
                                 cluster_method, 
                                 means = kwargs["means"],
                                 im2cluster = kwargs["im2cluster"],
                                 clip = kwargs["clip"],
                                 clip_metric = "cosine")
    try:
        if met == "mahalanobis" and com == 0 and cluster_method=="kmeans" and global_mal_cov:
            ood_evaluator(otrain_X, met, global_cov=True, inv_choice="default", recal=True)
        else:
            ood_evaluator(otrain_X, met)
    except:
        pass       
    
    try:
        ood_evaluator.get_scores()
    except:
        pass
    
    try:
        ood_evaluator.get_auroc()
    except:
        pass

    try:
        ood_evaluator.get_cluster_evaluation_metrics()
    except:
        pass

    try:
        ood_evaluator.calculate_new_aurocs()
    except:
        pass

    print("\n")

    props_to_save = ['in_classes', 'class_idx', 'cov_inv', '_means', 'train_dist', 'in_train_y', 'in_test_y', \
                    'test_dist', 'out_dist', 'train_pred_label', 'test_pred_label', 'out_pred_label',\
                     'train_pred_score', 'test_pred_score', 'out_pred_score', 'pred', 'pred_scores']
    props = {}
    for k in props_to_save:
        if k in vars(ood_evaluator).keys():
            props[k] = vars(ood_evaluator)[k]

    aurocs = ood_evaluator.auroc
    tnrs = ood_evaluator.tnr_at_tpr95
    gmm_results = ood_evaluator.gmm_results

    result = {"metric": met,
            "pca": com,
            "clusters": prot,
            "n_auroc": aurocs[0],
            "n_tnr": tnrs[0],
            "e_auroc": aurocs[1],
            "e_tnr":  tnrs[1],
            "sklearn_auroc": aurocs[2],
            "cluster_method":cluster_method,
            "gmm_default": gmm_results["gmm_default"],
            "gmm_max_prob": gmm_results["gmm_max_prob"],
            "gmm_weighted_max_prob": gmm_results[ "gmm_weighted_max_prob"],
            "global_mal_cov": global_mal_cov
            }

    props_result = {"metric": met,
                    "pca": com,
                    "clusters": prot,
                    "cluster_method":cluster_method,
                    "ood_evaluator_props": props
                    }


    cluster_eval_result = ood_evaluator.cluster_eval_result
    new_auroc_result  = ood_evaluator.new_auroc_result

    for k,v in [("metric", met), ("pca", com), ("clusters", prot), ("cluster_method", cluster_method)]:
        cluster_eval_result[k] = v
        new_auroc_result[k] = v
    
    return result, props_result, cluster_eval_result, new_auroc_result


