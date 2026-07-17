# Awesome Object Detection

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![Update](https://img.shields.io/github/last-commit/whut09/Awesome-object-detection)

A curated list of object detection papers from 2021 to 2026, including
real-time detection, end-to-end detection, open-vocabulary detection, oriented
detection, small-object detection, and domain-robust detection.

## Table of Contents

- [Paper List](#paper-list)
- [Screenshot Source Entries](#screenshot-source-entries)
- [Contributing](#contributing)
- [License](#license)

## Paper List

<!-- PAPER_TABLES:START -->
### Assignment, Loss, and Training
| Year | Pub | Title | Links | Main Institution |
|:---:|:---:|:---|:---:|:---:|
| 2021 | CVPR | **OTA: Optimal Transport Assignment for Object Detection** | [[paper](https://arxiv.org/abs/2103.14259)] [[code](https://github.com/Megvii-BaseDetection/OTA)] | Megvii |
| 2021 | ICCV | **TOOD: Task-aligned One-stage Object Detection** | [[paper](https://arxiv.org/abs/2108.07755)] [[code](https://github.com/fcjian/TOOD)] | SenseTime / SCUT |

### DETR and End-to-End Detection
| Year | Pub | Title | Links | Main Institution |
|:---:|:---:|:---|:---:|:---:|
| 2025 | ICLR | **D-FINE: Redefine Regression Task in DETRs as Fine-Grained Distribution Refinement** | [[paper](https://arxiv.org/abs/2410.13842)] [[code](https://github.com/Peterande/D-FINE)] | University of Sydney / Baidu |
| 2025 | CVPR | **DEIM: DETR with Improved Matching for Fast Convergence** | [[paper](https://arxiv.org/abs/2412.04234)] [[code](https://github.com/ShihuaHuang95/DEIM)] | Chinese Academy of Sciences |
| 2023 | ICCV | **DETRs with Collaborative Hybrid Assignments Training** | [[paper](https://arxiv.org/abs/2211.12860)] [[code](https://github.com/Sense-X/Co-DETR)] | SenseTime |
| 2023 | ICLR | **DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection** | [[paper](https://arxiv.org/abs/2203.03605)] [[code](https://github.com/IDEA-Research/DINO)] | IDEA Research |
| 2022 | ICLR | **DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR** | [[paper](https://arxiv.org/abs/2201.12329)] [[code](https://github.com/IDEA-Research/DAB-DETR)] | IDEA Research |
| 2022 | CVPR | **DN-DETR: Accelerate DETR Training by Introducing Query DeNoising** | [[paper](https://arxiv.org/abs/2203.01305)] [[code](https://github.com/IDEA-Research/DN-DETR)] | IDEA Research |
| 2021 | ICCV | **Conditional DETR for Fast Training Convergence** | [[paper](https://arxiv.org/abs/2108.06152)] [[code](https://github.com/Atten4Vis/ConditionalDETR)] | Microsoft Research Asia |
| 2021 | ICLR | **Deformable DETR: Deformable Transformers for End-to-End Object Detection** | [[paper](https://arxiv.org/abs/2010.04159)] [[code](https://github.com/fundamentalvision/Deformable-DETR)] | SenseTime / USTC |
| 2021 | CVPR | **Sparse R-CNN: End-to-End Object Detection with Learnable Proposals** | [[paper](https://arxiv.org/abs/2011.12450)] [[code](https://github.com/PeizeSun/SparseR-CNN)] | HKU / ByteDance |

### Open-Vocabulary and Grounded Detection
| Year | Pub | Title | Links | Main Institution |
|:---:|:---:|:---|:---:|:---:|
| 2026 | arXiv | **DeCo-DETR: Decoupled Cognition DETR for Efficient Open-Vocabulary Object Detection** | [[paper](https://arxiv.org/abs/2604.02753)] | University of Science and Technology of China |
| 2026 | arXiv | **FlowOVD: Learning Generative Latent Flows for Zero-shot Open-vocabulary Detection** | [[paper](https://arxiv.org/abs/2606.00782)] | University of California, Merced |
| 2026 | arXiv | **OV-DEIM: Real-time DETR-Style Open-Vocabulary Object Detection with GridSynthetic Augmentation** | [[paper](https://arxiv.org/abs/2603.07022)] [[code](https://github.com/wleilei/OV-DEIM)] | University of Electronic Science and Technology of China |
| 2026 | arXiv | **ProCal: Inference-Time Proposal Calibration for Open-Vocabulary Object Detection** | [[paper](https://arxiv.org/abs/2607.01759)] | University of Maryland |
| 2026 | arXiv | **VL-DINO: Leveraging CLIP Vision-Language Knowledge for Open-Vocabulary Object Detection** | [[paper](https://arxiv.org/abs/2606.11546)] | University of Science and Technology of China |
| 2026 | arXiv | **VocaDet: Sample-Driven Open-Vocabulary Object Detection and Segmentation via Visual Tokenization and Vector Database Retrieval** | [[paper](https://arxiv.org/abs/2607.08541)] | University of Technology Sydney |
| 2025 | CVPR | **YOLOE: Real-Time Seeing Anything** | [[paper](https://arxiv.org/abs/2503.07465)] [[code](https://github.com/THU-MIG/yoloe)] | Tsinghua University |
| 2024 | CVPR | **YOLO-World: Real-Time Open-Vocabulary Object Detection** | [[paper](https://arxiv.org/abs/2401.17270)] [[code](https://github.com/AILab-CVC/YOLO-World)] | Tencent AI Lab / SCUT |
| 2023 | arXiv | **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection** | [[paper](https://arxiv.org/abs/2303.05499)] [[code](https://github.com/IDEA-Research/GroundingDINO)] | IDEA Research |
| 2022 | ECCV | **Detecting Twenty-Thousand Classes Using Image-Level Supervision** | [[paper](https://arxiv.org/abs/2201.02605)] [[code](https://github.com/facebookresearch/Detic)] | Meta AI |
| 2022 | CVPR | **Grounded Language-Image Pre-training** | [[paper](https://arxiv.org/abs/2112.03857)] [[code](https://github.com/microsoft/GLIP)] | Microsoft |
| 2022 | ECCV | **Simple Open-Vocabulary Object Detection with Vision Transformers** | [[paper](https://arxiv.org/abs/2205.06230)] [[code](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)] | Google Research |

### Open-World and Domain-Robust Detection
| Year | Pub | Title | Links | Main Institution |
|:---:|:---:|:---|:---:|:---:|
| 2026 | arXiv | **Detecting Unknown Objects via Energy-based Separation for Open World Object Detection** | [[paper](https://arxiv.org/abs/2603.29954)] | Korea Advanced Institute of Science and Technology |
| 2026 | arXiv | **Why Domain Matters: Domain-Aware Benchmarking of Underwater Object Detection and Annotation Quality** | [[paper](https://arxiv.org/abs/2607.10575)] | University of Tasmania |

### Small, Aerial, and Oriented Detection
| Year | Pub | Title | Links | Main Institution |
|:---:|:---:|:---|:---:|:---:|
| 2026 | arXiv | **LOGOS: Language-guided Oriented Object Detection in Aerial Scenes** | [[paper](https://arxiv.org/abs/2607.08004)] | University of California, Santa Cruz |
| 2026 | arXiv | **RiO-DETR: DETR for Real-time Oriented Object Detection** | [[paper](https://arxiv.org/abs/2603.09411)] | Nanjing University of Science and Technology |
| 2022 | ICIP | **Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection** | [[paper](https://arxiv.org/abs/2202.06934)] [[code](https://github.com/obss/sahi)] | OBSS |
| 2021 | CVPR | **ReDet: A Rotation-equivariant Detector for Aerial Object Detection** | [[paper](https://arxiv.org/abs/2103.07733)] [[code](https://github.com/csuhan/ReDet)] | Wuhan University |

### YOLO and Real-Time Detection
| Year | Pub | Title | Links | Main Institution |
|:---:|:---:|:---|:---:|:---:|
| 2026 | arXiv | **MambaPSA: A Mamba-based Replacement for C2PSA in YOLO26** | [[paper](https://arxiv.org/abs/2607.12681)] | National Taiwan University of Science and Technology |
| 2026 | arXiv | **No Attention, No Problem: DPU-Aware Attention Approximation in Modern YOLO on FPGA** | [[paper](https://arxiv.org/abs/2607.13106)] | Bremen University of Applied Sciences |
| 2025 | arXiv | **YOLOv12: Attention-Centric Real-Time Object Detectors** | [[paper](https://arxiv.org/abs/2502.12524)] [[code](https://github.com/sunsmarterjie/yolov12)] | University at Buffalo / UCAS |
| 2024 | CVPR | **DETRs Beat YOLOs on Real-Time Object Detection** | [[paper](https://arxiv.org/abs/2304.08069)] [[code](https://github.com/lyuwenyu/RT-DETR)] | Baidu |
| 2024 | arXiv | **LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection** | [[paper](https://arxiv.org/abs/2406.03459)] [[code](https://github.com/Atten4Vis/LW-DETR)] | Microsoft Research Asia |
| 2024 | NeurIPS | **YOLOv10: Real-Time End-to-End Object Detection** | [[paper](https://arxiv.org/abs/2405.14458)] [[code](https://github.com/THU-MIG/yolov10)] | Tsinghua University |
| 2024 | arXiv | **YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information** | [[paper](https://arxiv.org/abs/2402.13616)] [[code](https://github.com/WongKinYiu/yolov9)] | Academia Sinica |
| 2023 | NeurIPS | **Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism** | [[paper](https://arxiv.org/abs/2309.11331)] [[code](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO)] | Huawei Noah's Ark Lab |
| 2022 | arXiv | **PP-YOLOE: An Evolved Version of YOLO** | [[paper](https://arxiv.org/abs/2203.16250)] [[code](https://github.com/PaddlePaddle/PaddleDetection)] | Baidu |
| 2022 | arXiv | **RTMDet: An Empirical Study of Designing Real-Time Object Detectors** | [[paper](https://arxiv.org/abs/2212.07784)] [[code](https://github.com/open-mmlab/mmyolo)] | OpenMMLab |
| 2022 | arXiv | **YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors** | [[paper](https://arxiv.org/abs/2207.02696)] [[code](https://github.com/WongKinYiu/yolov7)] | Academia Sinica |
| 2021 | arXiv | **YOLOX: Exceeding YOLO Series in 2021** | [[paper](https://arxiv.org/abs/2107.08430)] [[code](https://github.com/Megvii-BaseDetection/YOLOX)] | Megvii |
<!-- PAPER_TABLES:END -->

## Screenshot Source Entries

The supplied screenshots contain a multi-label source list. All 98 visible
entries, including entries that may belong to adjacent vision tasks, are
preserved in [the transcription](docs/wechat-screenshot-entries.md) for later
paper-link verification, deduplication, and task classification.

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
