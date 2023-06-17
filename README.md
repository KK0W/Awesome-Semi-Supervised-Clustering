# Awesome-Semi-Supervised-Clustering


This repository contains a list of resources related to semi-supervised clustering. Semi-supervised clustering is a type of clustering where some labels (or a partial set) are available, in addition to the data. The goal is to leverage these labels, along with the data, to perform better clustering. The problem is not well understood, and there are a variety of approaches to solve semi-supervised clustering.

# What is Semi-Supervised Clustering?
Semi-supervised clustering is a type of clustering algorithm that combines both labeled and unlabeled data to improve the clustering performance. In traditional clustering, the algorithm uses only the features of the data points to group them into clusters without any prior knowledge of their class labels. However, in semi-supervised clustering, a small portion of the data is labeled with the class information, and the algorithm uses this information along with the feature values to improve the clustering quality.

The labeled data can be used in different ways in semi-supervised clustering. For example, the algorithm can use the labeled data to guide the clustering process by assigning more weight to the labeled data points or by constraining the clustering to produce clusters that are consistent with the labeled data. Alternatively, the labeled data can be used to evaluate the quality of the clustering by comparing it to the true class labels.

Semi-supervised clustering has many applications in various fields such as image segmentation, text clustering, bioinformatics, and social network analysis. It can be particularly useful when the cost of obtaining labeled data is high or when the labeled data is limited, but there is a large amount of unlabeled data available.

More details can be found in the survey paper.[Link](https://www.sciencedirect.com/science/article/abs/pii/S0020025523002840)


## Papers

|Year|Paper| Venue |Code|
|:-----:|:------------------------:|:-----:|:---:|
|2021|&nbsp;&nbsp;&nbsp;[Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Cross-Domain_Adaptive_Clustering_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.html) &nbsp;&nbsp;&nbsp;|CVPR 2021 |[code](https://github.com/lijichang/CVPR2021-SSDA)|
|2019|[Semi-supervised deep embedded clustering](https://paperswithcode.com/paper/semi-supervised-deep-embedded-clustering)|Neurocomputing 2019 |[code](https://github.com/yongzx/SDEC-Keras)|
|2021|[SSSNET:Semi-Supervised Signed Network Clustering](https://paperswithcode.com/paper/sssnet-semi-supervised-signed-network)|| [code](https://github.com/sherylhyx/sssnet_signed_clustering)|
|2023|[Semi-supervised learning made simple with self-supervised clustering](https://openaccess.thecvf.com/content/CVPR2023/html/Fini_Semi-Supervised_Learning_Made_Simple_With_Self-Supervised_Clustering_CVPR_2023_paper.html)|CVPR2023|[code](https://github.com/pietroastolfi/suave-daino)|
|2020|[Poisson Learning: Graph Based Semi-Supervised Learning At Very Low Label Rates](https://arxiv.org/abs/2006.11184v2)| |[code](https://github.com/jwcalder/GraphLearning)|

## Reference

[1] Ren, Pengzhen, et al. "A survey of deep active learning." _ACM computing surveys_ (CSUR) 54.9 (2021): 1-40.  
[2] Awesome Active Learning. (2023). baifanxxx. https://github.com/baifanxxx/awesome-active-learning  
[3] 【论文汇总】人工智能顶会深度主动学习(Deep Active Learning)相关论文. (2022). 49号西瓜. https://blog.csdn.net/weixin_42126945/article/details/123418940?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-123418940-blog-125705082.235^v38^pc_relevant_sort_base2&spm=1001.2101.3001.4242.2&utm_relevant_index=4


### Python

- [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#semi-supervised-clustering)
- [scikit-multilearn](http://scikit.ml/api/skmultilearn.cluster.cobras.html)
- [spectral_clustering.py](https://github.com/harp/blob/master/algorithms/spectral_clustering.py)
- [sskmeans.py](https://github.com/pmtamayo/sskmeans)
- [sskmeans-scikit](https://github.com/timitsie/sskmeans_scikit)
- [higashi-cluster](https://github.com/iHilmi/higashi-cluster)
- [clustering-with-labels](https://github.com/MihaiBuda/Clustering-With-Labels)
- [kgcnn](https://github.com/ailabstw/kgcnn/tree/master/clustering/y-clustering)
- [semisup-semi-supervised-learning-for-python](https://github.com/tmadl/semisup)
- [RB-Sklearn](https://github.com/tmadl/RB-sklearn)
- [lssc_clustering](https://github.com/amoussavi/lssc_clustering)

### R

- [ssClust](https://www.rdocumentation.org/packages/ssClust/versions/0.1.3)

## Contribute

Contributions are always welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

## License

This repository is licensed under the [MIT License](LICENSE).
