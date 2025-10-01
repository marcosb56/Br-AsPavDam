BR-AsPavDam — Reproducible Training & Evaluation
Automated detection of urban asphalt defects in Brazil using YOLO.
Dataset: BR-AsPavDam (2,167 images, 3,918 boxes) • Code: fully reproducible pipelines • Results & weights released.

Dataset (Zenodo): https://zenodo.org/records/16291115
Repository: https://github.com/marcosb56/Br-AsPavDam

Environment & Hardware
Platform: Google Colab • GPU: NVIDIA L4 • RAM: ~15 GB • CUDA: 12.2
OS: Ubuntu 22.04 • Frameworks: PyTorch + Ultralytics (YOLOv5)
Rationale for 640 px: we train at 640×640 to reduce compute and approximate real-time / low-resource deployment, while preserving sufficient spatial detail for pavement defects.
Data

We use BR-AsPavDam (public, academic use) with YOLO‐format labels.
DOI (Dataset): https://zenodo.org/records/16291115
Classes (final 5): Fissures, Shoving (corrugation/shoving/rutting merged), Ravelling, Pothole, Patch
Splits/organization: images/ and labels/ (homonymous TXT per image)
Image size: 1152×2048 (PNG originals); training pipelines resize to 640×640
Pretrained Weights (RDDC2020 – IMSC)
We initialize from RDDC2020/IMSC public weights (Maeda et al., 2020).
The helper script download_IMSC_grddc2020_weights.sh fetches and stores the checkpoint under:
yolov5/weights/IMSC/last_100_100_640_16.pt
These weights provide domain priors (asphalt textures and defect patterns), speeding up convergence and improving generalization.
Training Configuration
Base model: YOLOv5 (Ultralytics, 2020)
Resolution: 640×640
Batch size: 16
Epochs:
Approach: 100 (patience=20), bootstrap on validation (100 resamples)
Augmentations: horizontal flip; small rotations (±5°); HSV/brightness/contrast jitter; Gaussian noise; motion blur; gamma correction; CLAHE; realistic shadows (all label-preserving)
Class imbalance: targeted oversampling (synthetic variants) until desired per-class frequencies (Approaches 2–3)
Eval thresholds: --conf 0.25, --iou 0.50 (unless otherwise stated)

# 1) IMSC weights
bash scripts/download_IMSC_grddc2020_weights.sh  # puts last_100_100_640_16.pt under yolov5/weights/IMSC/

# 2) Train (repro settings)
python scripts/train.py \
  --data configs/br-aspavdam.yaml \
  --weights yolov5/weights/IMSC/last_100_100_640_16.pt \
  --imgsz 640 --batch 16 --epochs 100 

# 3) Evaluate
python scripts/val.py --weights runs/train/exp/weights/best.pt --conf 0.25 --iou 0.50
Results (Summary)
100 epochs (Approach 3): mAP@50 = 84.3%
Notes: Reported metrics follow YOLOv5’s standard evaluation (P/R, mAP@50, mAP@50:95). Be mindful of the data imbalance (very few Deformations) and consider class-aware weighting/augmentations for further improvements.

How to Cite

Dataset
@dataset{Borges_BR-AsPavDam_2025,
  title     = {BR-AsPavDam: Urban asphalt defects dataset (Brazil)},
  author    = {Borges, Marcos A. and Spanhol, Fabio and collaborators},
  year      = {2025},
  publisher = {Zenodo},
  url       = {https://zenodo.org/records/16291115}
}
Repository (GitHub)
@misc{Borges_Spanhol_BrAsPavDam_Code_2025,
  title        = {BR-AsPavDam: Training and evaluation code for urban asphalt defect detection},
  author       = {Borges, Marcos A. and Spanhol, Fabio},
  year         = {2025},
  howpublished = {\url{https://github.com/marcosb56/Br-AsPavDam}},
  urldate      = {2025-10-01}
}
