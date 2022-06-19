# Investigation of OOD Detection using Contrastive Learning

This repo contains the codebase for OOD investigation using [Prototypical Contrastive Learning (PCL)] (https://github.com/salesforce/PCL.git).

Three types of representations are used in this work.
1. Contrastive representations trained using the KMeans-based PCL Framework
2. Finetuned representations trained on 10% of In-distribution training data
3. Distilled representations trained on 100% of unlabelled In-distribution training data
