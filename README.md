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

|2020|[A Survey of Active Learning for Text Classification using Deep Neural Networks](https://arxiv.org/abs/2008.07267)|Christopher Schröder et al.||
|2020|[A Survey of Deep Active Learning](https://arxiv.org/abs/2009.00236)|Pengzhen Ren et al.||

## Papers

### Image Recognition

|Year|Paper|Author|Code|
|:-----:|:------------------------:|:-----:|:---:|
|2022|&nbsp;&nbsp;&nbsp;[Budget-aware Few-shot Learning via Graph Convolutional Network](https://arxiv.org/abs/2201.02304) &nbsp;&nbsp;&nbsp;|Shipeng Yan et al.||
|2021|&nbsp;&nbsp;&nbsp;[Active Universal Domain Adaptation](https://openaccess.thecvf.com/content/ICCV2021/papers/Ma_Active_Universal_Domain_Adaptation_ICCV_2021_paper.pdf) &nbsp;&nbsp;&nbsp;|Xinhong Ma et al.||
|2021|&nbsp;&nbsp;&nbsp;[Contrastive Coding for Active Learning under Class Distribution Mismatch](https://openaccess.thecvf.com/content/ICCV2021/papers/Du_Contrastive_Coding_for_Active_Learning_Under_Class_Distribution_Mismatch_ICCV_2021_paper.pdf) &nbsp;&nbsp;&nbsp;|Pan Du et al.|[code](https://github.com/RUC-DWBI-ML/CCAL)|
|2021|&nbsp;&nbsp;&nbsp;[Multiple instance active learning for object detection](https://arxiv.org/abs/2104.02324) &nbsp;&nbsp;&nbsp;|Tianning Yuan et al.|[code](https://github.com/yuantn/MI-AOD)|
|2021|&nbsp;&nbsp;&nbsp;[S3VAADA: Submodular Subset Selection for Virtual Adversarial Active Domain Adaptation](https://openaccess.thecvf.com/content/ICCV2021/papers/Rangwani_S3VAADA_Submodular_Subset_Selection_for_Virtual_Adversarial_Active_Domain_Adaptation_ICCV_2021_paper.pdf) &nbsp;&nbsp;&nbsp;|Harsh Rangwani et al.|[code](https://github.com/val-iisc/s3vaada)|
|2021|&nbsp;&nbsp;&nbsp;[Semi-supervised Active Learning for Semi-supervised Models: Exploit Adversarial Examples with Graph-based Virtual Labels](https://openaccess.thecvf.com/content/ICCV2021/papers/Guo_Semi-Supervised_Active_Learning_for_Semi-Supervised_Models_Exploit_Adversarial_Examples_With_ICCV_2021_paper.pdf) &nbsp;&nbsp;&nbsp;|Jinnan Guo et al.||
|2021|&nbsp;&nbsp;&nbsp;[Sequential Graph Convolutional Network for Active Learning](https://arxiv.org/abs/2006.10219) &nbsp;&nbsp;&nbsp;|Razvan Caramalau et al.||
|2021|&nbsp;&nbsp;&nbsp;[Transferable Query Selection for Active Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2021/papers/Fu_Transferable_Query_Selection_for_Active_Domain_Adaptation_CVPR_2021_paper.pdf) &nbsp;&nbsp;&nbsp;|Bo Fu et al.|[code](https://github.com/thuml/Transferable-Query-Selection)|
|2021|&nbsp;&nbsp;&nbsp;[VaB-AL: Incorporating Class Imbalance and Difficulty with Variational Bayes for Active Learning](https://openaccess.thecvf.com/content/CVPR2021/papers/Choi_VaB-AL_Incorporating_Class_Imbalance_and_Difficulty_With_Variational_Bayes_for_CVPR_2021_paper.pdf) &nbsp;&nbsp;&nbsp;|Jongwon Choi et al.||
|2020|&nbsp;&nbsp;&nbsp;[Deep Active Learning for Biased Datasets via Fisher Kernel Self-Supervision](https://arxiv.org/abs/2003.00393) &nbsp;&nbsp;&nbsp;|Denis Gudovskiy et al.|[code](https://github.com/gudovskiy/al-fk-self-supervision)|
|2018|&nbsp;&nbsp;&nbsp;[The power of ensembles for active learning in image classification](https://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf) &nbsp;&nbsp;&nbsp;|William H. Beluch et al.||
|2017|&nbsp;&nbsp;&nbsp;[Active Decision Boundary Annotation with Deep Generative Models](https://arxiv.org/abs/1703.06971) &nbsp;&nbsp;&nbsp;|Miriam W. Huijser, Jan C. van Gemert|[code](https://github.com/MiriamHu/ActiveBoundary)|

### Text Classification

|Year|Paper|Author|Code|
|:-----:|:------------------------:|:-----:|:---:|
|2022|&nbsp;&nbsp;&nbsp;[PT4AL: Using Self-Supervised Pretext Tasks for Active Learning](https://arxiv.org/abs/2201.07459) &nbsp;&nbsp;&nbsp;|John Seon Keun Yi et al.|[code](https://github.com/johnsk95/pt4al)|
|2021|&nbsp;&nbsp;&nbsp;[Multi-Anchor Active Domain Adaptation for Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Ning_Multi-Anchor_Active_Domain_Adaptation_for_Semantic_Segmentation_ICCV_2021_paper.pdf) &nbsp;&nbsp;&nbsp;|Siyu Huang et al.|[code](https://github.com/munanning/MADA)|
|2021|&nbsp;&nbsp;&nbsp;[ReDAL: Region-based and Diversity-aware Active Learning for Point Cloud Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_ReDAL_Region-Based_and_Diversity-Aware_Active_Learning_for_Point_Cloud_Semantic_ICCV_2021_paper.pdf) &nbsp;&nbsp;&nbsp;|Tsung-Han Wu et al.||
|2021|&nbsp;&nbsp;&nbsp;[Revisiting Superpixels for Active Learning in Semantic Segmentation with Realistic Annotation Costs](https://openaccess.thecvf.com/content/CVPR2021/papers/Cai_Revisiting_Superpixels_for_Active_Learning_in_Semantic_Segmentation_With_Realistic_CVPR_2021_paper.pdf) &nbsp;&nbsp;&nbsp;|Lile Cai et al.||
|2021|&nbsp;&nbsp;&nbsp;[Semi-Supervised Active Learning with Temporal Output Discrepancy](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_Semi-Supervised_Active_Learning_With_Temporal_Output_Discrepancy_ICCV_2021_paper.pdf) &nbsp;&nbsp;&nbsp;|Munan Ning et al.|[code](https://github.com/siyuhuang/TOD)|
|2021|&nbsp;&nbsp;&nbsp;[Task-Aware Variational Adversarial Active Learning](https://arxiv.org/abs/2002.04709) &nbsp;&nbsp;&nbsp;|Kwanyoung Kim et al.|[code](https://github.com/cubeyoung/TA-VAAL)|
|2020|&nbsp;&nbsp;&nbsp;[State-Relabeling Adversarial Active Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_State-Relabeling_Adversarial_Active_Learning_CVPR_2020_paper.pdf) &nbsp;&nbsp;&nbsp;|Beichen Zhang et al.||
|2020|&nbsp;&nbsp;&nbsp;[ViewAL: Active Learning With Viewpoint Entropy for Semantic Segmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Siddiqui_ViewAL_Active_Learning_With_Viewpoint_Entropy_for_Semantic_Segmentation_CVPR_2020_paper.pdf) &nbsp;&nbsp;&nbsp;|Yawar Siddiqui et al.|[code](https://github.com/nihalsid/ViewAL)|
|2019|&nbsp;&nbsp;&nbsp;[Variational Adversarial Active Learning](https://arxiv.org/abs/1904.00370) &nbsp;&nbsp;&nbsp;|Samarth Sinha et al.|[code](https://github.com/sinhasam/vaal)|

### Object detection

|Year|Paper|Author|Code|
|:-----:|:------------------------:|:-----:|:---:|
|2021|&nbsp;&nbsp;&nbsp;[Active Learning for Deep Object Detection via Probabilistic Modeling](https://openaccess.thecvf.com/content/ICCV2021/papers/Choi_Active_Learning_for_Deep_Object_Detection_via_Probabilistic_Modeling_ICCV_2021_paper.pdf) &nbsp;&nbsp;&nbsp;|Jiwoong Choi et al.|[code](https://github.com/NVlabs/AL-MDN)|
|2021|&nbsp;&nbsp;&nbsp;[Active Learning for Lane Detection: A Knowledge Distillation Approach](https://openaccess.thecvf.com/content/ICCV2021/papers/Peng_Active_Learning_for_Lane_Detection_A_Knowledge_Distillation_Approach_ICCV_2021_paper.pdf) &nbsp;&nbsp;&nbsp;|Fengchao Peng et al.||
|2021|&nbsp;&nbsp;&nbsp;[Influence Selection for Active Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Influence_Selection_for_Active_Learning_ICCV_2021_paper.pdf)&nbsp;&nbsp;&nbsp;|Zhuoming Liu et al.||
|2021|&nbsp;&nbsp;&nbsp;[Multiple Instance Active Learning for Object Detection](https://arxiv.org/abs/2104.02324) &nbsp;&nbsp;&nbsp;|Tianning Yuan et al.|[code](https://github.com/yuantn/MI-AOD)|
|2019|&nbsp;&nbsp;&nbsp;[Learning Loss for Active Learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.pdf) &nbsp;&nbsp;&nbsp;|Donggeun Yoo et al.||
|2018|&nbsp;&nbsp;&nbsp;[Towards Human-Machine Cooperation: Self-supervised Sample Mining for Object Detection](https://arxiv.org/abs/1803.09867) &nbsp;&nbsp;&nbsp;|Keze Wang et al.||

### Others

|Year|Paper|Author|Code|
|:-----:|:------------------------:|:-----:|:---:|
|2023|&nbsp;&nbsp;&nbsp;[Low Budget Active Learning via Wasserstein Distance: An Integer Programming Approach](https://arxiv.org/abs/2106.02968) &nbsp;&nbsp;&nbsp;|Rafid Mahmood et al.||
|2022|&nbsp;&nbsp;&nbsp;[Active Learning Helps Pretrained Models Learn the Intended Task](https://arxiv.org/abs/2204.08491) &nbsp;&nbsp;&nbsp;|Alex Tamkin et al.||
|2022|&nbsp;&nbsp;&nbsp;[Meta-Query-Net: Resolving Purity-Informativeness Dilemma in Open-set Active Learning](https://arxiv.org/abs/2210.07805) &nbsp;&nbsp;&nbsp;|Binhui Xie et al.|[code](https://github.com/kaist-dmlab/mqnet)|
|2022|&nbsp;&nbsp;&nbsp;[Towards Fewer Annotations: Active Learning via Region Impurity and Prediction Uncertainty for Domain Adaptive Semantic Segmentation](https://arxiv.org/abs/2111.12940) &nbsp;&nbsp;&nbsp;|Binhui Xie et al.|[code](https://github.com/bit-da/ripu)|
|2021|&nbsp;&nbsp;&nbsp;[Ask&Confirm: Active Detail Enriching for Cross-Modal Retrieval with Partial Query](https://openaccess.thecvf.com/content/ICCV2021/papers/Cai_AskConfirm_Active_Detail_Enriching_for_Cross-Modal_Retrieval_With_Partial_Query_ICCV_2021_paper.pdf) &nbsp;&nbsp;&nbsp;|Guanyu Cai et al.|[code](https://github.com/CuthbertCai/Ask-Confirm)|

## Reference

[1] Ren, Pengzhen, et al. "A survey of deep active learning." _ACM computing surveys_ (CSUR) 54.9 (2021): 1-40.  
[2] Awesome Active Learning. (2023). baifanxxx. https://github.com/baifanxxx/awesome-active-learning  
[3] 【论文汇总】人工智能顶会深度主动学习(Deep Active Learning)相关论文. (2022). 49号西瓜. https://blog.csdn.net/weixin_42126945/article/details/123418940?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-123418940-blog-125705082.235^v38^pc_relevant_sort_base2&spm=1001.2101.3001.4242.2&utm_relevant_index=4




## Important Survey Papers

| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
| 2021 | Semi-Supervised Clustering: A Study on User-Guided and Active Approaches | IEEE Transactions on Neural Networks and Learning Systems | [Link](https://ieeexplore.ieee.org/abstract/document/9261803) | [Link](https://github.com/userguidedsemiSEMI) |
| 2020 | MixMatch: A Holistic Approach to Semi-Supervised Learning | NeurIPS 2019 | [Link](https://proceedings.neurips.cc/paper/2019/hash/1b5e3ef18d27ab1e91c3c2b99a5477f8-Abstract.html) | [Link](https://github.com/google-research/mixmatch)|
| 2019 | Deep Co-Clustering for Unsupervised and Semi-Supervised Learning | ICDM 2018 | [Link](https://ieeexplore.ieee.org/abstract/document/8595135) | - |
| 2018 | A Review of Semi-Supervised Clustering | IEEE Transactions on Knowledge and Data Engineering | [Link](https://ieeexplore.ieee.org/abstract/document/8428408) | - |
| 2017 | Semi-Supervised Clustering with Intercluster Discrimination | IEEE Transactions on Pattern Analysis and Machine Intelligence | [Link](https://ieeexplore.ieee.org/abstract/document/7962925) | - | 

## Papers
### Co-training based clustering
| Year | Title | Venue | Paper | Code |
|------|-------|-------|-------|------|
2023 | Co-Training-Based Outlier Detection for IoT Data Streams | IEEE/ACM Transactions on Networking | [Link](https://ieeexplore.ieee.org/document/9565368) | N/A
2022 | Multi-Modal Co-Training for Clustering | International Joint Conference on Neural Networks | [Link](https://ieeexplore.ieee.org/document/9662443) | N/A
2021 | Co-Training Multi-view Clustering with Enhanced Local-Global Consistency and Orthogonality | Neurocomputing | [Link](https://www.sciencedirect.com/science/article/pii/S0925231221010426) | N/A
| 2020 | Regularized Co-Training for Mapping Unlabeled Data between Different Domains | IEEE Transactions on Neural Networks and Learning Systems | [Link](https://ieeexplore.ieee.org/abstract/document/9096188) | - |
| 2019 | Co-Training Embeddings of Knowledge Graphs and Entity Descriptions for Cross-lingual Entity Alignment | ACL 2019 | [Link](https://www.aclweb.org/anthology/P19-1546/) | - |
2018 | Co-training over Unlabeled Data with Domain-Specific Information for Multi-domain Sentiment Analysis | IEEE International Symposium on Signal Processing and Information Technology | [Link](https://ieeexplore.ieee.org/document/8468805) | N/A
2017 | Co-Training Deep Convolutional Networks for Semi-Supervised Clustering | IEEE Transactions on Multimedia | [Link](https://ieeexplore.ieee.org/document/7838145) | [Code](https://github.com/Yangyangii/CoNet)
2016 | Multi-view Co-training for Semi-supervised Clustering | International Conference on Multimedia Modeling | [Link](https://link.springer.com/chapter/10.1007/978-3-319-27671-7_10) | N/A
2015 | Co-Training Ensemble Clustering for High-Dimensional Dat | Symposium on Applied Computing | [Link](https://dl.acm.org/doi/abs/10.1145/2695664.2695878) | N/A
| 2015 | A Co-Training Approach for Multi-View Spectral Clustering | IEEE Transactions on Image Processing | [Link](https://ieeexplore.ieee.org/abstract/document/7040050) | - |
| 2014 | Co-training for domain adaptation of sentiment classifiers | EMNLP 2014 | [Link](https://www.aclweb.org/anthology/D14-1081.pdf) | [Link](https://github.com/bluemonk482/co-training-for-domain-adaptation) |
### Self-training-based clustering
Year | Title | Venue | Paper | Code 
--- | --- | --- | --- | ---
2014 | A Self-training Approach to Cluster Unlabeled Documents | ACM Transactions on Speech and Language Processing | [Link](https://dl.acm.org/doi/abs/10.1145/2661529) | N/A
2015 | Self-Training Ensemble Clustering for High Dimensional Data | IEEE International Conference on Data Mining | [Link](https://ieeexplore.ieee.org/document/7373341) | [Code](https://github.com/luckiezhou/Self-Training-Ensemble-Clustering)
2016 | Self-Training Ensemble Clustering for High Dimensional Data | IEEE Transactions on Knowledge and Data Engineering | [Link](https://ieeexplore.ieee.org/document/7396918) | N/A
2018 | Self-Training-Based Clustering Ensemble for High Dimensional Data | IEEE Transactions on Neural Networks and Learning Systems | [Link](https://ieeexplore.ieee.org/document/8359461) | [Code](https://github.com/luckiezhou/Self-Training-Ensemble-Clustering)
2019 | Self-Training Ensemble Clustering for High Dimensional Data: A Multi-Objective Optimization Framework | Information Sciences | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0020025519301570) | [Code](https://github.com/luckiezhou/Self-Training-Ensemble-Clustering)
2020 | Self-Training Ensemble Clustering with Consistent Cluster Information | Proceedings of the AAAI Conference on Artificial Intelligence | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/6489/6347) | N/A
2021 | Self-Training Ensemble Clustering with Entropy Regularization | IEEE Transactions on Neural Networks and Learning Systems | [Link](https://ieeexplore.ieee.org/document/9448888) | [Code](https://github.com/mfuca/Self-Training-Ensemble-Clustering-with-Entropy-Regularization)
2022 | Self-Training-Based Ensemble Clustering with Dynamically Consistent Cluster Assignments | Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition | [Link](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_Self-Training-Based_Ensemble_Clustering_With_Dynamically_Consistent_Cluster_Assignments_ICCV_2021_paper.pdf) | [Code](https://github.com/luckiezhou/Self-Training-Ensemble-Clustering)
2023 | Self-Training Ensemble Clustering with Improved Diversity | Knowledge-Based Systems | [Link](https://www.sciencedirect.com/science/article/pii/S0950705122003100) | N/A



### Generate semi-supervised models

Year | Title | Venue | Paper | Code 
--- | --- | --- | --- | ---
2014 | Semi-Supervised Learning with Deep Generative Models | Advances in Neural Information Processing Systems | [Link](https://proceedings.neurips.cc/paper/2014/hash/6e8ba87b69b25a1a9b0cf1fe657f29d1-Abstract.html) | N/A
2015 | A Quality-Diversity Algorithm for Semi-Supervised Clustering with Generative Models | IEEE Transactions on Cybernetics | [Link](https://ieeexplore.ieee.org/document/7042659) | N/A
2016 | Improved Techniques for Training GANs | Advances in Neural Information Processing Systems | [Link](https://papers.nips.cc/paper/2016/hash/4a82bceae955be5e8f53c8c155fc32e3-Abstract.html) | [Code](https://github.com/openai/improved-gan)
2017 | A Unified Approach to Semi-Supervised Learning with Generative Adversarial Nets | IEEE Transactions on Pattern Analysis and Machine Intelligence | [Link](https://ieeexplore.ieee.org/document/8016922) | [Code](https://github.com/lzhbrian/SSGAN-Tensorflow)
2018 | Semi-Supervised Deep Generative Models for Improved Scene Understanding | IEEE Transactions on Image Processing | [Link](https://ieeexplore.ieee.org/document/8370201) | N/A
2019 | Semi-Supervised Clustering with DPGMM: The Role of Intrinsic Dimension | Knowledge-Based Systems | [Link](https://www.sciencedirect.com/science/article/pii/S0950705119300442) | N/A
2020 | ClusterGAN++: Learning Discrete Latent Codes for Unsupervised/Semi-Supervised Clustering | IEEE/CVF Conference on Computer Vision and Pattern Recognition | [Link](https://openaccess.thecvf.com/content_CVPR_2020/papers/Choi_ClusterGAN_Learning_Discrete_Latent_Codes_for_Unsupervised_Semi-Supervised_Clustering_CVPR_2020_paper.pdf) | [Code](https://github.com/chrischoy/ClusterGAN)
2021 | Semi-Supervised Feature Assignment using Multi-Assignment GANs | IEEE Journal of Selected Topics in Signal Processing | [Link](https://ieeexplore.ieee.org/document/9404369) | [Code](https://github.com/Bhavya-123/MAGAN)
2022 | Semi-Supervised Clustering via Deep Generative Models with Gumbel-Softmax Trick | Information Sciences | [Link](https://www.sciencedirect.com/science/article/pii/S0020025522006898) | N/A
2023 | Deep Semi-Supervised Learning for Anomaly Detection | Proceedings of the AAAI Conference on Artificial Intelligence | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/18284/18197) | N/A

### S3VMs

Year | Title | Venue | Paper | Code 
--- | --- | --- | --- | ---
2014 | Semisupervised overfitting Control by ℓ1,2-Regularized SVM | IEEE Transactions on Neural Networks and Learning Systems | [Link](https://ieeexplore.ieee.org/document/6784816) | N/A
2015 | Improving Semi-Supervised Learning Performance with Temporal Coherence | AAAI Conference on Artificial Intelligence | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/10816/10667) | N/A
2016 | Efficient Feature Selection for Semi-Supervised SVM with l2 Regularizer | Journal of Information and Computational Science | [Link](http://www.joics.com/uploadfile/2016/0519/20160519023200180.pdf) | N/A
2017 | A Novel Method for Semi-Supervised Classification Based on Adaptive Discrete Artificial Bee Colony Algorithm | Neurocomputing | [Link](https://www.sciencedirect.com/science/article/pii/S0925231217315438) | N/A
2018 | A Semi-Supervised Double Regularized SVM Based on Cliff Delta Score | International Conference on Intelligent Computing and Intelligent Systems | [Link](https://ieeexplore.ieee.org/document/8590859) | N/A
2019 | Semi-Supervised Fraud Detection with High Correlation Feature Selection and Soft Margin SVM | Journal of Information Security and Applications | [Link](https://www.sciencedirect.com/science/article/pii/S2214212618309652) | N/A
2020 | Semi-Supervised Classification of Hyperspectral Images based on Support Vector Machines | Sensors | [Link](https://www.mdpi.com/1424-8220/20/23/7033) | [Code](https://github.com/noelpy/Semi-Supervised-Hyperspectral-Image-Classification-Using-SVM)
2021 | Semi-Supervised Evo-SVM for Interval-Valued Data Classification with Differential Evolution | IEEE Transactions on Fuzzy Systems | [Link](https://ieeexplore.ieee.org/document/9314138) | N/A
2022 | Effect of Non-Normality on Semi-Supervised Support Vector Machines | Neurocomputing | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0925231222010750) | N/A
2023 | Large-Scale Consensus-Based Semi-Supervised Learning with Support Vector Machines | IEEE Transactions on Cybernetics | [Link](https://ieeexplore.ieee.org/document/9571790) | N/A

### Graph-Based Algorithms

Year | Title | Venue | Paper | Code 
--- | --- | --- | --- | ---
2014 | Graph-Based Semi-Supervised Learning with Convolutional Neural Networks | Conference on Computer Vision and Pattern Recognition | [Link](https://ieeexplore.ieee.org/document/6909639) | [Code](https://github.com/dhlee347/pyGSSL)
2015 | Non-Parametric Graph Construction for Semi-Supervised Learning on Manifolds | Advances in Neural Information Processing Systems | [Link](https://proceedings.neurips.cc/paper/2015/hash/2a1f385f7412f0dee5f7aa523f45a8ff-Abstract.html) | [Code](https://github.com/mmazuran/deep-ssl)
2016 | A Framework of Semi-Supervised Learning Based on Deep Generative Models | International Conference on Computer Vision | [Link](https://ieeexplore.ieee.org/document/7780681) | N/A
2017 | Learning Deep Representations with Probabilistic Knowledge Transfer | Ninth International Conference on Machine Learning and Data Mining | [Link](https://link.springer.com/chapter/10.1007/978-3-319-62401-3_2) | [Code](https://github.com/bharel/SSNMTL)
2018 | Combination of Edge and Node Information for Semi-Supervised Learning with Graph Convolutional Networks | IEEE Access | [Link](https://ieeexplore.ieee.org/document/8343583) | [Code](https://github.com/kimiyoung/planetoid)
2019 | Graphology: A Fast and Scalable Framework for Graph-Regularized Deep Learning-Based Drug Discovery | Journal of Chemical Information and Modeling | [Link](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00673) | N/A
2020 | Learning with Graphs: A Survey | Foundations and Trends in Machine Learning | [Link](https://www.nowpublishers.com/article/Details/MAL-084) | N/A
2021 | Semi-Supervised Learning with Graph Convolutional Networks: Methods, Analysis, and Applications | Journal of Signal Processing Systems | [Link](https://link.springer.com/article/10.1007/s11265-020-01767-5) | N/A
2022 | Graph Convolutional Networks with FishNet Graph Construction for Semi-Supervised Classification | IEEE Transactions on Neural Networks and Learning Systems | [Link](https://ieeexplore.ieee.org/document/9576872) | N/A
2023 | Graph Regularized Semi-Supervised Learning with Cross-Modal Representation | Information Sciences | [Link](https://www.sciencedirect.com/science/article/pii/S0020025522002593) | N/A



## Benchmark Datasets

Since semi-supervised clustering is primarily applied to graph-structured data, there are no ‘non-graph datasets’ available for this purpose. Typically, semi-supervised learning algorithms are only used in graph data because only graph data can naturally define neighborhood relationships.

#### Quick Start

- Step1: Download all datasets from \[[Google Drive](https://drive.google.com/drive/folders/1thSxtAexbvOyjx-bJre8D4OyFKsBe1bK?usp=sharing) | [Nutstore](https://www.jianguoyun.com/p/DfzK1pwQwdaSChjI2aME)]. Optionally, download some of them from URLs in the tables (Google Drive)
- Step2: Unzip them to **./dataset/**
- Step3: Change the type and the name of the dataset in **main.py**
- Step4: Run the **main.py**

- **utils.py**
  1. **load_data**: load graph datasets 
  2. **preprocess_data**:  performs additional preprocessing on the data
  3. **generate_labels**: generates pseudolabels or predicted labels
  4. **train_model**:  trains the semi-supervised clustering model on the data
  5. **predict_clusters**: predicts the cluster assignments for the unlabeled data
  6. **evaluate_model**: evaluates the performance of the model on test data
  7. **plot_clusters**: visualizes the cluster assignments and/or centroids in a scatter plot, heatmap, or other graphical representation
  8. **save_model and load_model**:   computes various clustering performance metrics
  9. **compute_cluster_metrics**: evaluate the performance of clustering
  10. **normalize_data**: normalizes the data for better clustering performance


#### Datasets Details

About the introduction of each dataset, please check [here](./dataset/README.md)
| Dataset | # Samples | # Dimension | # Edges | # Classes | URL |
|---------|-----------|-------------|---------|-------------|-----|
| MNIST   | 70,000    | 784         | N/A     | 10         | [Link](http://yann.lecun.com/exdb/mnist/) |
| Fashion-MNIST | 70,000 | 784 | N/A | 10 | [Link](https://github.com/zalandoresearch/fashion-mnist) |
| CIFAR-10 | 60,000    | 3 * 32 * 32 | N/A | 10 | [Link](https://www.cs.toronto.edu/~kriz/cifar.html) |
| CIFAR-100 | 60,000   | 3 * 32 * 32 | N/A | 100 | [Link](https://www.cs.toronto.edu/~kriz/cifar.html) |
| SVHN    | 600,000   | 32 * 32 * 3 | N/A | 10         | [Link](http://ufldl.stanford.edu/housenumbers/) |
| Reuters-21578 | 21,578 | 2,000 | N/A | 90 | [Link](https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection) |
| 20 Newsgroups | 20,000  | N/A         | N/A     | 20         | [Link](http://qwone.com/~jason/20Newsgroups/) |
| Olivetti faces | 400      | 64 * 64 | N/A     | 40         | [Link](https://scikit-learn.org/stable/datasets/) |
Citeseer | 3327 | 3703 | 4536 | 6 | [Link](https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz)
Cora | 2708 | 1433 | 5429 | 7 | [Link](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz)
DBLP | 1775 | 334 | 9005 | 4 | [Link](https://linqs-data.soe.ucsc.edu/public/dblp.tgz)
Pubmed | 19717 | 500 | 44327 | 3 | [Link](https://linqs-data.soe.ucsc.edu/public/lbc/pubmed.tgz)
NELL | 2389 | 542 | 2763 | 9 | [Link](https://www.dropbox.com/s/wi7xat1rrr8hq4j/ReadMe.txt?dl=0)
Wisconsin Breast Cancer | 569 | 30 | 1980 | 2 | [Link](https://www.cs.wisc.edu/~street/729/Project/WBCD.tgz)
USPS | 9298 | 256 | 180714 | 10 | [Link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps)
WebKB | 877 | 1703 | 1603 | 5 | [Link](https://linqs-data.soe.ucsc.edu/public/wcb.tgz)
BlogCatalog | 10312 | 3703 | 333983 | 39 | [Link](https://linqs-data.soe.ucsc.edu/public/lbc/BlogCatalog-dataset.rar)
Flickr | 89250 | 500 | 899756 | 195 | [Link](http://webdatacommons.org/hyperlinkgraph/2014-04/download.html)

## Code

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
