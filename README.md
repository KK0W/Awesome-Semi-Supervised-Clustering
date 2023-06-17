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
|2023|[Semi-supervised learning made simple with self-supervised clustering](https://openaccess.thecvf.com/content/CVPR2023/html/Fini_Semi-Supervised_Learning_Made_Simple_With_Self-Supervised_Clustering_CVPR_2023_paper.html)|CVPR|[code](https://github.com/pietroastolfi/suave-daino)|
|2023|[Semi-Supervised Clustering Under a “Compact-Cluster” Assumption](https://ieeexplore.ieee.org/document/9693296)|IEEE||
|2022|[When Does Contrastive Visual Representation Learning Work?](https://arxiv.org/abs/2105.05837)| CVPR||
|2022|[Leverage Your Local and Global Representations: A New Self-Supervised Learning Strategy](https://arxiv.org/abs/2203.17205)|CVPR | |
|2022|[SimMatch: Semi-supervised Learning with Similarity Matching](https://arxiv.org/abs/2203.06915)| CVPR||
|2022|[Class-Aware Contrastive Semi-Supervised Learning](https://arxiv.org/abs/2203.02261)|  CVPR||
|2021|[A survey on Optimisation-based Semi-supervised Clustering Methods](https://ieeexplore.ieee.org/document/9667756)| IEEE ||
|2021|[Salp Swarm Algorithm based Semi-supervised Metric Fuzzy Clustering](https://ieeexplore.ieee.org/document/9512989)| IEEE ||
|2021|[Semi-Supervised Clustering with Inaccurate Pairwise Annotations](https://arxiv.org/abs/2104.02146v1)| |[code](https://github.com/danielgribel/SSC-IPA)|
|2021|&nbsp;&nbsp;&nbsp;[Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Cross-Domain_Adaptive_Clustering_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.html) &nbsp;&nbsp;&nbsp;|CVPR|[code](https://github.com/lijichang/CVPR2021-SSDA)|
|2021|[SSSNET:Semi-Supervised Signed Network Clustering](https://paperswithcode.com/paper/sssnet-semi-supervised-signed-network)|| [code](https://github.com/sherylhyx/sssnet_signed_clustering)|
|2020|[Poisson Learning: Graph Based Semi-Supervised Learning At Very Low Label Rates](https://arxiv.org/abs/2006.11184v2)| |[code](https://github.com/jwcalder/GraphLearning)|
|2020|[Semi-Supervised Clustering With Constraints of Different Types From Multiple Information Sources](https://ieeexplore.ieee.org/document/9031553)| IEEE||
|2020|[Semi-supervised Deep Embedded Clustering with Anomaly Detection for Semantic Frame Induction](https://aclanthology.org/2020.lrec-1.431/)| LREC|[code](https://github.com/yongzx/SDEC-AD)|
|2020|[Semi-Supervised Self-Training Feature Weighted Clustering Decision Tree and Random Forest](https://ieeexplore.ieee.org/document/9139499)| IEEE||
|2020|[Strongly local p-norm-cut algorithms for semi-supervised learning and local graph clustering](https://proceedings.neurips.cc//paper/2020/hash/3501672ebc68a5524629080e3ef60aef-Abstract.html)| NeurIPS |[code](https://github.com/MengLiuPurdue/SLQ)|
|2020|[A semi-supervised sparse K-Means algorithm](https://arxiv.org/abs/2003.06973)|PRL |[code](https://github.com/avouros/Code-PCSKM)|
|2020|[Poisson Learning: Graph Based Semi-Supervised Learning At Very Low Label Rates](https://arxiv.org/abs/2006.11184v2)| |[code](https://github.com/jwcalder/GraphLearning)|
|2019|[Semi-supervised deep embedded clustering](https://paperswithcode.com/paper/semi-supervised-deep-embedded-clustering)|Neurocomputing |[code](https://github.com/yongzx/SDEC-Keras)|

## Datasets
Datasets used in papers above, links lead to the homepage of each dataset.

| Dataset           | Link                                                                |
|-------------------|---------------------------------------------------------------------|
| vgmidi            | [Link](https://github.com/lucasnfe/vgmidi)                          |
| cifar-10          | [Link](https://www.cs.toronto.edu/~kriz/cifar.html)                 |
| cifar-100         | [Link](https://www.cs.toronto.edu/~kriz/cifar.html)                 |
| oxford-102-flower | [Link](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)          |
| caltech-101       | [Link](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)    |
| dtd               | [Link](https://www.robots.ox.ac.uk/~vgg/data/dtd/)                  |
| food-101          | [Link](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) |
| imagenet          | [Link](https://image-net.org/index.php)                             |
| banking77         | [Link](https://arxiv.org/abs/2003.04807)                            |
| clinc150          | [Link](https://github.com/clinc/oos-eval)                           |
| zinc              | [Link](https://zinc15.docking.org/)                                 |


## Reference

[1] CVPR 2022 最全整理：论文分方向汇总 / 代码 / 解读 / 直播 / 项目（更新中）【计算机视觉】
 https://www.cvmart.net/community/detail/6124

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
