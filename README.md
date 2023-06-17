# Awesome-Semi-Supervised-Clustering


This repository contains a list of resources related to semi-supervised clustering. Semi-supervised clustering is a type of clustering where some labels (or a partial set) are available, in addition to the data. The goal is to leverage these labels, along with the data, to perform better clustering. The problem is not well understood, and there are a variety of approaches to solve semi-supervised clustering.

# What is Semi-Supervised Clustering?
Semi-supervised clustering is a type of clustering algorithm that combines both labeled and unlabeled data to improve the clustering performance. In traditional clustering, the algorithm uses only the features of the data points to group them into clusters without any prior knowledge of their class labels. However, in semi-supervised clustering, a small portion of the data is labeled with the class information, and the algorithm uses this information along with the feature values to improve the clustering quality.

The labeled data can be used in different ways in semi-supervised clustering. For example, the algorithm can use the labeled data to guide the clustering process by assigning more weight to the labeled data points or by constraining the clustering to produce clusters that are consistent with the labeled data. Alternatively, the labeled data can be used to evaluate the quality of the clustering by comparing it to the true class labels.

Semi-supervised clustering has many applications in various fields such as image segmentation, text clustering, bioinformatics, and social network analysis. It can be particularly useful when the cost of obtaining labeled data is high or when the labeled data is limited, but there is a large amount of unlabeled data available.


## Survey Papers

|Year|Paper|Author|Code|
|:-----:|:------------------------:|:-----:|:---:|
|2021|&nbsp;&nbsp;&nbsp;[Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Cross-Domain_Adaptive_Clustering_for_Semi-Supervised_Domain_Adaptation_CVPR_2021_paper.html) &nbsp;&nbsp;&nbsp;|Xueying Zhan et al.|[code](https://github.com/lijichang/CVPR2021-SSDA)|
|2021|[A Survey on Active Deep Learning: From Model-driven to Data-driven](https://arxiv.org/abs/2101.09933)|Peng Liu et al.||
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
