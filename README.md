# Awesome Object Detection

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![Update](https://img.shields.io/github/last-commit/wzhforgithub/Awesome-object-detection)
![Papers](https://img.shields.io/badge/curated_papers-29-blue)

A curated and machine-readable collection of object-detection papers from the
five completed publication years **2021-2025**, plus rolling 2026 discovery
(last checked **July 17, 2026**).
The table style follows
[`njulj/Awesome-Agent-Based-Low-Level-Vision`](https://github.com/njulj/Awesome-Agent-Based-Low-Level-Vision),
and structured records plug into
[`whut09/YOLO-Agent`](https://github.com/whut09/YOLO-Agent).

> A manually reviewed list cannot guarantee every paper. This first version is a
> high-signal seed catalog plus an arXiv discovery pipeline, so coverage can grow
> without feeding unverified metadata directly into the training harness.

## Table of Contents

- [Paper List](#paper-list)
- [YOLO-Agent Integration](#yolo-agent-integration)
- [Catalog Maintenance](#catalog-maintenance)
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
| 2025 | CVPR | **YOLOE: Real-Time Seeing Anything** | [[paper](https://arxiv.org/abs/2503.07465)] [[code](https://github.com/THU-MIG/yoloe)] | Tsinghua University |
| 2024 | CVPR | **YOLO-World: Real-Time Open-Vocabulary Object Detection** | [[paper](https://arxiv.org/abs/2401.17270)] [[code](https://github.com/AILab-CVC/YOLO-World)] | Tencent AI Lab / SCUT |
| 2023 | arXiv | **Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection** | [[paper](https://arxiv.org/abs/2303.05499)] [[code](https://github.com/IDEA-Research/GroundingDINO)] | IDEA Research |
| 2022 | ECCV | **Detecting Twenty-Thousand Classes Using Image-Level Supervision** | [[paper](https://arxiv.org/abs/2201.02605)] [[code](https://github.com/facebookresearch/Detic)] | Meta AI |
| 2022 | CVPR | **Grounded Language-Image Pre-training** | [[paper](https://arxiv.org/abs/2112.03857)] [[code](https://github.com/microsoft/GLIP)] | Microsoft |
| 2022 | ECCV | **Simple Open-Vocabulary Object Detection with Vision Transformers** | [[paper](https://arxiv.org/abs/2205.06230)] [[code](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)] | Google Research |

### Small, Aerial, and Oriented Detection
| Year | Pub | Title | Links | Main Institution |
|:---:|:---:|:---|:---:|:---:|
| 2022 | ICIP | **Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection** | [[paper](https://arxiv.org/abs/2202.06934)] [[code](https://github.com/obss/sahi)] | OBSS |
| 2021 | CVPR | **ReDet: A Rotation-equivariant Detector for Aerial Object Detection** | [[paper](https://arxiv.org/abs/2103.07733)] [[code](https://github.com/csuhan/ReDet)] | Wuhan University |

### YOLO and Real-Time Detection
| Year | Pub | Title | Links | Main Institution |
|:---:|:---:|:---|:---:|:---:|
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

## YOLO-Agent Integration

The catalog exports YOLO-Agent `PaperRecord` JSONL and a separate harness-hint
stream. Paper claims remain `paper_prior` until local protocol-matched reproduction.

```bash
python scripts/catalog.py validate
python scripts/catalog.py export-yolo
python scripts/catalog.py export-hints
python scripts/catalog.py sync-yolo --yolo-root ../YOLO-Agent
```

See [`docs/YOLO_AGENT_INTEGRATION.md`](docs/YOLO_AGENT_INTEGRATION.md).

## Catalog Maintenance

`data/papers.json` is the reviewed source of truth.

```bash
python scripts/discover_arxiv.py --start-year 2021 --max-results 100
python scripts/catalog.py validate
python scripts/catalog.py render
python scripts/catalog.py stats
```

## Contributing

Add a stable paper identifier, official links, YOLO-Agent component tags,
applicability, and testable harness hints. Run validation and tests before a PR.

## License

Released under the [MIT License](LICENSE).
