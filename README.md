# Awesome Object Detection

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![Update](https://img.shields.io/github/last-commit/whut09/Awesome-object-detection)

A curated list of object detection papers from 2021 to 2026, including
real-time detection, end-to-end detection, open-vocabulary detection, oriented
detection, small-object detection, and domain-robust detection.

## Table of Contents

- [Paper List](#paper-list)
- [Contributing](#contributing)
- [License](#license)

## Paper List

<!-- PAPER_TABLES:START -->
### Assignment, Loss, and Training
| Year | Pub | Title | Links | Main Institution |
|:---:|:---:|:---|:---:|:---:|
| 2023 | CVPR | **Bridging Precision and Confidence: A Train-Time Loss for Calibrating Object Detection** | [[paper](https://arxiv.org/abs/2303.14404)] [[code](https://github.com/akhtarvision/bpc_calibration)] | Mohamed bin Zayed University of Artificial Intelligence |
| 2023 | ICCV | **Bridging the Gap between Classification and Localization for Better Feature Alignment** | [[paper](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Bridging_the_Gap_between_Classification_and_Localization_for_Better_Feature_Alignment_ICCV_2023_paper.html)] [[code](https://github.com/TinyTigerPan/BCKD)] | See paper |
| 2023 | AAAI | **Correlation Loss: Enforcing Correlation between Classification and Localization** | [[paper](https://arxiv.org/abs/2301.01019)] | Middle East Technical University |
| 2022 | CVPR | **A Dual Weighting Label Assignment Scheme for Object Detection** | [[paper](https://arxiv.org/abs/2203.09730)] | See paper |
| 2022 | arXiv | **DSLA: Dynamic Smooth Label Assignment for Efficient Anchor-Free Object Detection** | [[paper](https://arxiv.org/abs/2208.00817)] [[code](https://github.com/MingboHong/DSLA)] | See paper |
| 2021 | arXiv | **Mutual Supervision for Dense Object Detection** | [[paper](https://arxiv.org/abs/2109.05986)] | See paper |
| 2021 | CVPR | **OTA: Optimal Transport Assignment for Object Detection** | [[paper](https://arxiv.org/abs/2103.14259)] [[code](https://github.com/Megvii-BaseDetection/OTA)] | Megvii |
| 2021 | CVPRW | **Pseudo-IoU: Improving Label Assignment in Anchor-Free Object Detection** | [[paper](https://arxiv.org/abs/2104.14082)] [[code](https://github.com/Barrett-python/Pseudo-IoU)] | See paper |
| 2021 | ICCV | **TOOD: Task-aligned One-stage Object Detection** | [[paper](https://arxiv.org/abs/2108.07755)] [[code](https://github.com/fcjian/TOOD)] | SenseTime / SCUT |

### DETR and End-to-End Detection
| Year | Pub | Title | Links | Main Institution |
|:---:|:---:|:---|:---:|:---:|
| 2026 | CVPR | **PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection** | [[paper](https://arxiv.org/abs/2603.06917)] | See paper |
| 2025 | ICCV | **Adversarial Attention Perturbations for Large Object Detection Transformers** | [[paper](https://arxiv.org/abs/2508.02987)] | See paper |
| 2025 | ICLR | **D-FINE: Redefine Regression Task in DETRs as Fine-Grained Distribution Refinement** | [[paper](https://arxiv.org/abs/2410.13842)] [[code](https://github.com/Peterande/D-FINE)] | University of Sydney / Baidu |
| 2025 | CVPR | **DEIM: DETR with Improved Matching for Fast Convergence** | [[paper](https://arxiv.org/abs/2412.04234)] [[code](https://github.com/ShihuaHuang95/DEIM)] | Chinese Academy of Sciences |
| 2025 | ICCV | **EvRT-DETR: Latent Space Adaptation of Image Detectors for Event-based Vision** | [[paper](https://arxiv.org/abs/2412.02890)] [[code](https://github.com/realtime-intelligence/evrt-detr)] | See paper |
| 2025 | CVPR | **MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism** | [[paper](https://arxiv.org/abs/2503.01463)] | See paper |
| 2025 | CVPR | **Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with MoE** | [[paper](https://arxiv.org/abs/2412.10028)] | See paper |
| 2025 | ICCV | **Sim-DETR: Unlock DETR for Temporal Sentence Grounding** | [[paper](https://arxiv.org/abs/2509.23867)] [[code](https://github.com/SooLab/Sim-DETR)] | See paper |
| 2023 | CVPR | **Dense Distinct Query for End-to-End Object Detection** | [[paper](https://arxiv.org/abs/2303.12776)] [[code](https://github.com/jshilong/DDQ)] | Shanghai AI Laboratory |
| 2023 | ICCV | **DETRs with Collaborative Hybrid Assignments Training** | [[paper](https://arxiv.org/abs/2211.12860)] [[code](https://github.com/Sense-X/Co-DETR)] | SenseTime |
| 2023 | ICLR | **DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection** | [[paper](https://arxiv.org/abs/2203.03605)] [[code](https://github.com/IDEA-Research/DINO)] | IDEA Research |
| 2023 | arXiv | **Focal-Stable-DINO: A Strong and Reproducible Object Detector with Public Resources** | [[paper](https://arxiv.org/abs/2304.13027)] | IDEA Research |
| 2023 | CVPR | **One-to-Few Label Assignment for End-to-End Dense Detection** | [[paper](https://arxiv.org/abs/2303.11567)] [[code](https://github.com/strongwolf/o2f)] | See paper |
| 2022 | ICLR | **DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR** | [[paper](https://arxiv.org/abs/2201.12329)] [[code](https://github.com/IDEA-Research/DAB-DETR)] | IDEA Research |
| 2022 | CVPR | **DN-DETR: Accelerate DETR Training by Introducing Query DeNoising** | [[paper](https://arxiv.org/abs/2203.01305)] [[code](https://github.com/IDEA-Research/DN-DETR)] | IDEA Research |
| 2021 | ICCV | **Conditional DETR for Fast Training Convergence** | [[paper](https://arxiv.org/abs/2108.06152)] [[code](https://github.com/Atten4Vis/ConditionalDETR)] | Microsoft Research Asia |
| 2021 | ICLR | **Deformable DETR: Deformable Transformers for End-to-End Object Detection** | [[paper](https://arxiv.org/abs/2010.04159)] [[code](https://github.com/fundamentalvision/Deformable-DETR)] | SenseTime / USTC |
| 2021 | CVPR | **Sparse R-CNN: End-to-End Object Detection with Learnable Proposals** | [[paper](https://arxiv.org/abs/2011.12450)] [[code](https://github.com/PeizeSun/SparseR-CNN)] | HKU / ByteDance |

### General Object Detection
| Year | Pub | Title | Links | Main Institution |
|:---:|:---:|:---|:---:|:---:|
| 2026 | CVPR | **A Closer Look at Cross-Domain Few-Shot Object Detection: Fine-Tuning Matters and Parallel Decoder Helps** | [[paper](https://arxiv.org/abs/2603.28182)] [[code](https://github.com/Intellindust-AI-Lab/FT-FSOD)] | See paper |
| 2026 | CVPR | **Beyond Duality: A Hybrid Framework of Leveraging Shared and Private Features for RGB-Event Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Beyond_Duality_A_Hybrid_Framework_of_Leveraging_Shared_and_Private_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Distribution-Aligned Multimodal Fusion for Robust Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Hao_Distribution-Aligned_Multimodal_Fusion_for_Robust_Object_Detection_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **DyFCLT: Dynamic Frequency-Decoupled Cross-Modal Learning Transformer for Multimodal Tiny Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Li_DyFCLT_Dynamic_Frequency-Decoupled_Cross-Modal_Learning_Transformer_for_Multimodal_Tiny_Object_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Explaining Object Detectors via Collective Contribution of Pixels** | [[paper](https://arxiv.org/abs/2412.00666)] | See paper |
| 2026 | CVPR | **Foundation Model Priors Enhance Object Focus in Feature Space for Source-Free Object Detection** | [[paper](https://arxiv.org/abs/2512.17514)] | See paper |
| 2026 | CVPR | **Heuristic-inspired Reasoning Priors Facilitate Data-Efficient Referring Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Heuristic-inspired_Reasoning_Priors_Facilitate_Data-Efficient_Referring_Object_Detection_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **InsCal: Calibrated Multi-Source Fully Test-Time Prompt Tuning for Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Que_InsCal_Calibrated_Multi-Source_Fully_Test-Time_Prompt_Tuning_for_Object_Detection_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Mind the Gap: Transferring Labels to Align Object Detection Datasets** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Kennerley_Mind_the_Gap_Transferring_Labels_to_Align_Object_Detection_Datasets_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Online Data Curation for Object Detection via Marginal Contributions to Dataset-level Average Precision** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_Online_Data_Curation_for_Object_Detection_via_Marginal_Contributions_to_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Partial Weakly-Supervised Oriented Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Partial_Weakly-Supervised_Oriented_Object_Detection_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Portable Active Learning for Object Detection** | [[paper](https://arxiv.org/abs/2605.10349)] | See paper |
| 2026 | CVPR | **RARE: Learn to RAnk and REtrieve for Monocular 3D Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Park_RARE_Learn_to_RAnk_and_REtrieve_for_Monocular_3D_Object_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Remedying Target-Domain Astigmatism for Cross-Domain Few-Shot Object Detection** | [[paper](https://arxiv.org/abs/2603.18541)] | See paper |
| 2026 | CVPR | **RHCNet: Residual-Guided Hierarchical Calibration Network for Robust Underwater Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_RHCNet_Residual-Guided_Hierarchical_Calibration_Network_for_Robust_Underwater_Object_Detection_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Spike-driven Discrete Aggregation for Event-based Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Spike-driven_Discrete_Aggregation_for_Event-based_Object_Detection_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **When Transformers Meet Mamba: A Hybrid Transformer-Mamba Network for Video Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Qi_When_Transformers_Meet_Mamba_A_Hybrid_Transformer-Mamba_Network_for_Video_CVPR_2026_paper.html)] | See paper |
| 2025 | ICCV | **Automated Model Evaluation for Object Detection via Prediction Consistency and Reliability** | [[paper](https://arxiv.org/abs/2508.12082)] [[code](https://github.com/YonseiML/autoeval-det)] | See paper |
| 2025 | CVPR | **Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention** | [[paper](https://arxiv.org/abs/2403.10173)] | See paper |
| 2025 | ICCV | **LMM-Det: Make Large Multimodal Models Excel in Object Detection** | [[paper](https://arxiv.org/abs/2507.18300)] [[code](https://github.com/360CVGroup/LMM-Det)] | See paper |
| 2025 | CVPR | **Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset** | [[paper](https://arxiv.org/abs/2412.06647)] [[code](https://github.com/Event-AHU/OpenEvDET)] | See paper |
| 2025 | ICCV | **Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights** | [[paper](https://arxiv.org/abs/2508.00649)] [[code](https://github.com/Gandolfczjh/APDE)] | See paper |
| 2025 | CVPR | **Test-Time Backdoor Detection for Object Detection Models** | [[paper](https://arxiv.org/abs/2503.15293)] | See paper |
| 2025 | CVPR | **Towards RAW Object Detection in Diverse Conditions** | [[paper](https://arxiv.org/abs/2411.15678)] [[code](https://github.com/lzyhha/AODRaw)] | See paper |
| 2025 | ICCV | **Visual Modality Prompt for Adapting Vision-Language Object Detectors** | [[paper](https://arxiv.org/abs/2412.00622)] [[code](https://github.com/heitorrapela/ModPrompt)] | See paper |
| 2024 | IJCV | **A Systematic Evaluation of Uncertainty Calibration in Pre-Trained Object Detectors** | [[paper](https://doi.org/10.1007/s11263-024-02219-z)] | University of Kassel |
| 2023 | ICCV | **AID: Pushing the Performance Boundary of Human-Aware Object Detection** | [[paper](https://arxiv.org/abs/2310.05666)] [[code](https://github.com/YilongLv/AID)] | See paper |
| 2023 | ICCV | **AlignDet: Aligning Pre-training and Fine-tuning in Object Detection** | [[paper](https://arxiv.org/abs/2307.11077)] | See paper |
| 2023 | ICCVW | **DatasetEquity: Are All Samples Created Equal? In The Quest For Equity Within Datasets** | [[paper](https://doi.org/10.1109/ICCVW60793.2023.00476)] | See paper |
| 2023 | CVPR | **DynamicDet: A Unified Dynamic Architecture for Object Detection** | [[paper](https://arxiv.org/abs/2304.05552)] [[code](https://github.com/VDIGPKU/DynamicDet)] | Peking University |
| 2023 | arXiv | **ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data** | [[paper](https://arxiv.org/abs/2309.00832)] | Cleanlab |
| 2023 | CVPRW | **Training Strategies for Object Detection with Vision Transformers** | [[paper](https://arxiv.org/abs/2304.02186)] | See paper |
| 2022 | arXiv | **BigDetection: A Large-scale Benchmark for Improved Object Detector Pre-training** | [[paper](https://arxiv.org/abs/2203.13249)] [[code](https://github.com/amazon-science/bigdetection)] | Amazon |
| 2022 | CVPR | **Proper Reuse of Image Classification Features Improves Object Detection** | [[paper](https://arxiv.org/abs/2204.00484)] | Google Research |
| 2021 | ACM MM | **Disentangle Your Dense Object Detector** | [[paper](https://arxiv.org/abs/2107.02963)] [[code](https://github.com/chenminghao3/DDOD)] | University of Science and Technology of China |

### Open-Vocabulary and Grounded Detection
| Year | Pub | Title | Links | Main Institution |
|:---:|:---:|:---|:---:|:---:|
| 2026 | CVPR | **Consistency Beyond Contrast: Enhancing Open-Vocabulary Object Detection Robustness via Contextual Consistency Learning** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Consistency_Beyond_Contrast_Enhancing_Open-Vocabulary_Object_Detection_Robustness_via_Contextual_CVPR_2026_paper.html)] | See paper |
| 2026 | arXiv | **DeCo-DETR: Decoupled Cognition DETR for Efficient Open-Vocabulary Object Detection** | [[paper](https://arxiv.org/abs/2604.02753)] | University of Science and Technology of China |
| 2026 | arXiv | **FlowOVD: Learning Generative Latent Flows for Zero-shot Open-vocabulary Detection** | [[paper](https://arxiv.org/abs/2606.00782)] | University of California, Merced |
| 2026 | CVPR | **NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection** | [[paper](https://arxiv.org/abs/2603.21069)] | See paper |
| 2026 | arXiv | **OV-DEIM: Real-time DETR-Style Open-Vocabulary Object Detection with GridSynthetic Augmentation** | [[paper](https://arxiv.org/abs/2603.07022)] [[code](https://github.com/wleilei/OV-DEIM)] | University of Electronic Science and Technology of China |
| 2026 | CVPR | **Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection** | [[paper](https://arxiv.org/abs/2604.04444)] | See paper |
| 2026 | arXiv | **ProCal: Inference-Time Proposal Calibration for Open-Vocabulary Object Detection** | [[paper](https://arxiv.org/abs/2607.01759)] | University of Maryland |
| 2026 | CVPR | **SRA-Det: Learning Omni-Grained Open-Vocabulary Detection Beyond Category Names** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_SRA-Det_Learning_Omni-Grained_Open-Vocabulary_Detection_Beyond_Category_Names_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Thermal-Det: Language-Guided Cross-Modal Distillation for Open-Vocabulary Thermal Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Ranasinghe_Thermal-Det_Language-Guided_Cross-Modal_Distillation_for_Open-Vocabulary_Thermal_Object_Detection_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **ViTPrompt: Training-Free Prompt Refinement with Visual Tokens for Open-Vocabulary Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Qin_ViTPrompt_Training-Free_Prompt_Refinement_with_Visual_Tokens_for_Open-Vocabulary_Detection_CVPR_2026_paper.html)] | See paper |
| 2026 | arXiv | **VL-DINO: Leveraging CLIP Vision-Language Knowledge for Open-Vocabulary Object Detection** | [[paper](https://arxiv.org/abs/2606.11546)] | University of Science and Technology of China |
| 2026 | arXiv | **VocaDet: Sample-Driven Open-Vocabulary Object Detection and Segmentation via Visual Tokenization and Vector Database Retrieval** | [[paper](https://arxiv.org/abs/2607.08541)] | University of Technology Sydney |
| 2026 | CVPR | **WeDetect: Fast Open-Vocabulary Object Detection as Retrieval** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Fu_WeDetect_Fast_Open-Vocabulary_Object_Detection_as_Retrieval_CVPR_2026_paper.html)] | See paper |
| 2025 | ICCV | **3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection** | [[paper](https://arxiv.org/abs/2507.23567)] | See paper |
| 2025 | CVPR | **ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection** | [[paper](https://arxiv.org/abs/2603.12409)] | See paper |
| 2025 | ICCV | **Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection** | [[paper](https://arxiv.org/abs/2507.17436)] [[code](https://github.com/wengminghe/Dynamic-DINO)] | See paper |
| 2025 | ICCV | **SFUOD: Source-Free Unknown Object Detection** | [[paper](https://arxiv.org/abs/2507.17373)] [[code](https://github.com/SFUOD)] | See paper |
| 2025 | CVPR | **YOLOE: Real-Time Seeing Anything** | [[paper](https://arxiv.org/abs/2503.07465)] [[code](https://github.com/THU-MIG/yoloe)] | Tsinghua University |
| 2024 | CVPR | **YOLO-World: Real-Time Open-Vocabulary Object Detection** | [[paper](https://arxiv.org/abs/2401.17270)] [[code](https://github.com/AILab-CVC/YOLO-World)] | Tencent AI Lab / SCUT |
| 2023 | arXiv | **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection** | [[paper](https://arxiv.org/abs/2303.05499)] [[code](https://github.com/IDEA-Research/GroundingDINO)] | IDEA Research |
| 2022 | ECCV | **Detecting Twenty-Thousand Classes Using Image-Level Supervision** | [[paper](https://arxiv.org/abs/2201.02605)] [[code](https://github.com/facebookresearch/Detic)] | Meta AI |
| 2022 | CVPR | **Grounded Language-Image Pre-training** | [[paper](https://arxiv.org/abs/2112.03857)] [[code](https://github.com/microsoft/GLIP)] | Microsoft |
| 2022 | ECCV | **Simple Open-Vocabulary Object Detection with Vision Transformers** | [[paper](https://arxiv.org/abs/2205.06230)] [[code](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)] | Google Research |

### Open-World and Domain-Robust Detection
| Year | Pub | Title | Links | Main Institution |
|:---:|:---:|:---|:---:|:---:|
| 2026 | CVPR | **Beyond Prompt Degradation: Prototype-Guided Dual-Pool Prompting for Incremental Object Detection** | [[paper](https://arxiv.org/abs/2603.02286)] [[code](https://github.com/zyt95579/PDP_IOD/tree/main)] | See paper |
| 2026 | CVPR | **Black-Box Domain Adaptation for Object Detection with Retention-Driven Knowledge Compression** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_Black-Box_Domain_Adaptation_for_Object_Detection_with_Retention-Driven_Knowledge_Compression_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection** | [[paper](https://arxiv.org/abs/2603.26092)] | See paper |
| 2026 | CVPR | **DA-Mamba: Learning Domain-Aware State Space Model for Global-Local Alignment in Domain Adaptive Object Detection** | [[paper](https://arxiv.org/abs/2603.18757)] | See paper |
| 2026 | arXiv | **Detecting Unknown Objects via Energy-based Separation for Open World Object Detection** | [[paper](https://arxiv.org/abs/2603.29954)] | Korea Advanced Institute of Science and Technology |
| 2026 | CVPR | **EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer** | [[paper](https://arxiv.org/abs/2602.20985)] | See paper |
| 2026 | CVPR | **Expert-Teacher-Student Collaborative Learning for Domain Adaptive Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Cui_Expert-Teacher-Student_Collaborative_Learning_for_Domain_Adaptive_Object_Detection_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Incremental Object Detection via Future-Aware Decoupled Cross-Head Distillation** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Yin_Incremental_Object_Detection_via_Future-Aware_Decoupled_Cross-Head_Distillation_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Parameterized Prompt for Incremental Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/An_Parameterized_Prompt_for_Incremental_Object_Detection_CVPR_2026_paper.html)] | See paper |
| 2026 | arXiv | **Why Domain Matters: Domain-Aware Benchmarking of Underwater Object Detection and Annotation Quality** | [[paper](https://arxiv.org/abs/2607.10575)] | University of Tasmania |
| 2025 | CVPR | **Efficient Test-Time Adaptive Object Detection via Sensitivity-Guided Pruning** | [[paper](https://arxiv.org/abs/2506.02462)] | See paper |
| 2025 | CVPR | **Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection** | [[paper](https://arxiv.org/abs/2503.23220)] [[code](https://github.com/TRAILab/DINO_Teacher)] | See paper |
| 2025 | CVPR | **SimLTD: Simple Supervised and Semi-Supervised Long-Tailed Object Detection** | [[paper](https://arxiv.org/abs/2412.20047)] | See paper |
| 2025 | ICCV | **UPRE: Zero-Shot Domain Adaptation for Object Detection via Unified Prompt and Representation Enhancement** | [[paper](https://arxiv.org/abs/2507.00721)] [[code](https://github.com/AMAP-ML/UPRE)] | See paper |
| 2023 | CVPR | **2PCNet: Two-Phase Consistency Training for Day-to-Night Unsupervised Domain Adaptive Object Detection** | [[paper](https://arxiv.org/abs/2303.13853)] | See paper |
| 2023 | ICCV | **Augmented Box Replay: Overcoming Foreground Shift for Incremental Object Detection** | [[paper](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Augmented_Box_Replay_Overcoming_Foreground_Shift_for_Incremental_Object_Detection_ICCV_2023_paper.html)] | See paper |
| 2023 | WACV | **ConfMix: Unsupervised Domain Adaptation for Object Detection via Confidence-Based Mixing** | [[paper](https://arxiv.org/abs/2210.11539)] [[code](https://github.com/giuliomattolin/ConfMix)] | University of Trento / FBK |
| 2023 | arXiv | **DINF: Dynamic Instance Noise Filter for Occluded Pedestrian Detection** | [[paper](https://arxiv.org/abs/2301.05565)] | See paper |
| 2023 | CVPRW | **Exploring the Role of Synthetic Data in Human Detection** | [[paper](https://arxiv.org/abs/2303.13221)] | See paper |
| 2023 | CVPR | **Generating Features with Increased Crop-Related Diversity for Few-Shot Object Detection** | [[paper](https://arxiv.org/abs/2304.05096)] | See paper |
| 2023 | ICCVW | **Tensor Factorization for Leveraging Cross-Modal Knowledge in Data-Constrained Object Detection** | [[paper](https://openaccess.thecvf.com/content/ICCV2023W/PAR/html/Sedunov_Tensor_Factorization_for_Leveraging_Cross-Modal_Knowledge_in_Data-Constrained_Object_Detection_ICCVW_2023_paper.html)] | Mitsubishi Electric Research Laboratories |
| 2022 | ECCV | **Exploring Resolution and Degradation Clues as Self-Supervised Signal for Low Quality Object Detection** | [[paper](https://arxiv.org/abs/2208.03062)] [[code](https://github.com/Chokyeol/AERIS)] | See paper |
| 2022 | arXiv | **Pedestrian Detection: Domain Generalization, CNNs, Transformers and Beyond** | [[paper](https://arxiv.org/abs/2201.03176)] [[code](https://github.com/hasanirtiza/Pedestron)] | See paper |
| 2022 | arXiv | **The Impact of Partial Occlusion on Pedestrian Detectability** | [[paper](https://arxiv.org/abs/2205.04812)] | See paper |

### Small, Aerial, and Oriented Detection
| Year | Pub | Title | Links | Main Institution |
|:---:|:---:|:---|:---:|:---:|
| 2026 | CVPR | **Balanced Hierarchical Contrastive Learning with Decoupled Queries for Fine-grained Object Detection in Remote Sensing Images** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Balanced_Hierarchical_Contrastive_Learning_with_Decoupled_Queries_for_Fine-grained_Object_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **BDNet: Bio-Inspired Dual-Backbone Small Object Detection Network** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Guan_BDNetBio-Inspired_Dual-Backbone_Small_Object_Detection_Network_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **CHAL: Causal-guided Hierarchical Anomaly-aware Learning for Moving Infrared Small Target Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Duan_CHAL_Causal-guided_Hierarchical_Anomaly-aware_Learning_for_Moving_Infrared_Small_Target_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Fourier Angle Alignment for Oriented Object Detection in Remote Sensing** | [[paper](https://arxiv.org/abs/2602.23790)] [[code](https://github.com/gcy0423/Fourier-Angle-Alignment)] | See paper |
| 2026 | arXiv | **LOGOS: Language-guided Oriented Object Detection in Aerial Scenes** | [[paper](https://arxiv.org/abs/2607.08004)] | University of California, Santa Cruz |
| 2026 | arXiv | **RiO-DETR: DETR for Real-time Oriented Object Detection** | [[paper](https://arxiv.org/abs/2603.09411)] | Nanjing University of Science and Technology |
| 2026 | CVPR | **Rotation Invariant and Symmetry Aware Pixel Difference Network for Remote Sensing Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Zhan_Rotation_Invariant_and_Symmetry_Aware_Pixel_Difference_Network_for_Remote_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Target-Aware Invertible Encoder with Reconstruction Guidance for Infrared Small Target Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Yan_Target-Aware_Invertible_Encoder_with_Reconstruction_Guidance_for_Infrared_Small_Target_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Towards Persistence: Learning Topological Constraints for Event-based Small Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/He_Towards_Persistence_Learning_Topological_Constraints_for_Event-based_Small_Object_Detection_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Tri-Modal Fusion Transformers for UAV-based Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Iaboni_Tri-Modal_Fusion_Transformers_for_UAV-based_Object_Detection_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Visual Prototype Conditioned Focal Region Generation for UAV-Based Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Visual_Prototype_Conditioned_Focal_Region_Generation_for_UAV-Based_Object_Detection_CVPR_2026_paper.html)] | See paper |
| 2025 | ICCV | **From Easy to Hard: Progressive Active Learning Framework for Infrared Small Target Detection with Single Point Supervision** | [[paper](https://arxiv.org/abs/2412.11154)] [[code](https://github.com/YuChuang1205/PAL)] | See paper |
| 2025 | ICCV | **Measuring the Impact of Rotation Equivariance on Aerial Object Detection** | [[paper](https://arxiv.org/abs/2507.09896)] [[code](https://github.com/Nu1sance/MessDet)] | See paper |
| 2025 | ICCV | **OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images** | [[paper](https://arxiv.org/abs/2503.06146)] | See paper |
| 2025 | CVPR | **Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images** | [[paper](https://arxiv.org/abs/2603.06925)] | See paper |
| 2025 | ICCV | **Uncertainty-Aware Gradient Stabilization for Small Object Detection** | [[paper](https://arxiv.org/abs/2303.01803)] | See paper |
| 2022 | ICIP | **Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection** | [[paper](https://arxiv.org/abs/2202.06934)] [[code](https://github.com/obss/sahi)] | OBSS |
| 2021 | CVPR | **ReDet: A Rotation-equivariant Detector for Aerial Object Detection** | [[paper](https://arxiv.org/abs/2103.07733)] [[code](https://github.com/csuhan/ReDet)] | Wuhan University |

### YOLO and Real-Time Detection
| Year | Pub | Title | Links | Main Institution |
|:---:|:---:|:---|:---:|:---:|
| 2026 | CVPR | **AKCMamba-YOLO: Selective State Space Models For Real-Time Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_AKCMamba-YOLO_Selective_State_Space_Models_For_Real-Time_Object_Detection_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **Does YOLO Really Need to See Every Training Image in Every Epoch?** | [[paper](https://arxiv.org/abs/2603.17684)] | See paper |
| 2026 | arXiv | **MambaPSA: A Mamba-based Replacement for C2PSA in YOLO26** | [[paper](https://arxiv.org/abs/2607.12681)] | National Taiwan University of Science and Technology |
| 2026 | arXiv | **No Attention, No Problem: DPU-Aware Attention Approximation in Modern YOLO on FPGA** | [[paper](https://arxiv.org/abs/2607.13106)] | Bremen University of Applied Sciences |
| 2026 | arXiv | **Ultralytics YOLO26: An End-to-End Edge-Optimized Vision Model** | [[paper](https://arxiv.org/abs/2606.03748)] [[code](https://github.com/ultralytics/ultralytics)] | Ultralytics |
| 2026 | CVPR | **YOLO-Master: MOE-Accelerated with Specialized Transformers for Enhanced Real-time Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_YOLO-Master_MOE-Accelerated_with_Specialized_Transformers_for_Enhanced_Real-time_Detection_CVPR_2026_paper.html)] | See paper |
| 2026 | CVPR | **YOLO-ULM: Ultra-Lightweight Models for Real-Time Object Detection** | [[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Han_YOLO-ULM_Ultra-Lightweight_Models_for_Real-Time_Object_Detection_CVPR_2026_paper.html)] | See paper |
| 2025 | arXiv | **YOLOv12: Attention-Centric Real-Time Object Detectors** | [[paper](https://arxiv.org/abs/2502.12524)] [[code](https://github.com/sunsmarterjie/yolov12)] | University at Buffalo / UCAS |
| 2025 | arXiv | **YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception** | [[paper](https://arxiv.org/abs/2506.17733)] [[code](https://github.com/iMoonLab/yolov13)] | See paper |
| 2024 | CVPR | **DETRs Beat YOLOs on Real-Time Object Detection** | [[paper](https://arxiv.org/abs/2304.08069)] [[code](https://github.com/lyuwenyu/RT-DETR)] | Baidu |
| 2024 | arXiv | **LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection** | [[paper](https://arxiv.org/abs/2406.03459)] [[code](https://github.com/Atten4Vis/LW-DETR)] | Microsoft Research Asia |
| 2024 | NeurIPS | **YOLOv10: Real-Time End-to-End Object Detection** | [[paper](https://arxiv.org/abs/2405.14458)] [[code](https://github.com/THU-MIG/yolov10)] | Tsinghua University |
| 2024 | arXiv | **YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information** | [[paper](https://arxiv.org/abs/2402.13616)] [[code](https://github.com/WongKinYiu/yolov9)] | Academia Sinica |
| 2023 | NeurIPS | **Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism** | [[paper](https://arxiv.org/abs/2309.11331)] [[code](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO)] | Huawei Noah's Ark Lab |
| 2023 | arXiv | **Real-Time Flying Object Detection with YOLOv8** | [[paper](https://arxiv.org/abs/2305.09972)] | See paper |
| 2023 | AAAI | **YOLOV: Making Still Image Object Detectors Great at Video Object Detection** | [[paper](https://arxiv.org/abs/2208.09686)] [[code](https://github.com/YuHengsss/YOLOV)] | See paper |
| 2022 | arXiv | **DAMO-YOLO: A Report on Real-Time Object Detection Design** | [[paper](https://arxiv.org/abs/2211.15444)] [[code](https://github.com/tinyvision/DAMO-YOLO)] | Alibaba Group |
| 2022 | arXiv | **PP-YOLOE: An Evolved Version of YOLO** | [[paper](https://arxiv.org/abs/2203.16250)] [[code](https://github.com/PaddlePaddle/PaddleDetection)] | Baidu |
| 2022 | arXiv | **RTMDet: An Empirical Study of Designing Real-Time Object Detectors** | [[paper](https://arxiv.org/abs/2212.07784)] [[code](https://github.com/open-mmlab/mmyolo)] | OpenMMLab |
| 2022 | arXiv | **YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors** | [[paper](https://arxiv.org/abs/2207.02696)] [[code](https://github.com/WongKinYiu/yolov7)] | Academia Sinica |
| 2021 | arXiv | **YOLO5Face: Why Reinventing a Face Detector** | [[paper](https://arxiv.org/abs/2105.12931)] [[code](https://github.com/deepcam-cn/yolov5-face)] | Deepcam |
| 2021 | arXiv | **YOLOX: Exceeding YOLO Series in 2021** | [[paper](https://arxiv.org/abs/2107.08430)] [[code](https://github.com/Megvii-BaseDetection/YOLOX)] | Megvii |
<!-- PAPER_TABLES:END -->

## Contributing

Contributions are welcome. Please add papers that are relevant to object
detection and follow these guidelines:

- Use a stable paper identifier, preferably an arXiv ID or DOI.
- Include the title, publication year, venue, paper link, and main institution.
- Add the official code or project link when one is available.
- Place the paper in the most specific existing category; add a category only when necessary.
- Keep titles and links accurate, and do not add duplicate papers or duplicate versions.
- For recent arXiv papers, use the arXiv publication year and arXiv as the venue.
- Add the entry to data/papers.json, run python scripts/catalog.py validate, and regenerate the README with python scripts/catalog.py render.
- Keep paper-reported results separate from independently reproduced results.

Please open a pull request with a short explanation of why the paper belongs in
the list. Corrections to titles, venues, links, and categories are also welcome.

## License

Released under the [MIT License](LICENSE).
