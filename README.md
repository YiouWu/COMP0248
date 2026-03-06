# COMP0248
This repository contains my individual COMP0248 Coursework 1 implementation for RGB-D hand gesture perception with **multi-task learning**:
- **Segmentation**: hand mask (binary)
- **Detection**: hand bounding box
- **Classification**: 10 gesture classes

It supports:
- A **baseline** RGB-D model (shared encoder + U-Net style decoder + cls/box heads)
- An **innovation** that reuses the predicted mask to improve localisation and recognition:
  - mask-guided pooling for cls/box features
  - inference-time mask-to-box refinement for detection


---

## 1. Environment (Linux + GPU)

### 1.1 Check GPU

```bash
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

### 1.2 Create a conda environment

```bash
conda create -n comp0248 python=3.10 -y
conda activate comp0248
```

### 1.3 Install dependencies

Install a CUDA-enabled PyTorch build that matches your machine, then common packages:

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install numpy pillow matplotlib scikit-learn tqdm
```

---

## 2. Data

Expected collated dataset layout:

```
collated_dataset/RGB_depth_annotations/
  <subject_name_or_id>/
    G01_call/clip01/{rgb,depth,depth_raw,annotation}/...
    ...
```

Depth frames are read from `depth/` (PNG). Masks are read from `annotation/` (PNG) for keyframes only.

---

## 3. Create train/val split (subject-level)

From the repository root:

```bash
python -m src.make_split --data_root "..collated_dataset/RGB_depth_annotations" --out "splits_rgbd.json" --val_frac 0.15 --seed 42
```

---

## 4. Train (early stopping)

Early stopping monitors a composite validation score:

(score) = (seg_dice + det_acc@0.5 + cls_acc) / 3

### 4.1 Baseline

```bash
python -m src.train --data_root "..collated_dataset/RGB_depth_annotations" --split_json "splits_rgbd.json" --variant baseline --out_dir "results_baseline" --weights_dir "weights_baseline" --epochs 100 --patience 8 --min_delta 0.002 --batch_size 16 --num_workers 2 --lr 0.001 --weight_decay 0.0001 --target_size 256 --device cuda
```

### 4.2 Innovation

```bash
python -m src.train --data_root "..collated_dataset/RGB_depth_annotations" --split_json "splits_rgbd.json" --variant innovation --out_dir "results_innovation" --weights_dir "weights_innovation" --epochs 100 --patience 8 --min_delta 0.002 --batch_size 16 --num_workers 2 --lr 0.001 --weight_decay 0.0001 --target_size 256 --device cuda
```

Outputs:
- `weights_*/best.pt`, `weights_*/last.pt`
- `results_*/log.csv`

---

## 5. Evaluate (val)

### 5.1 Baseline (val)

```bash
python -m src.evaluate --data_root "..collated_dataset/RGB_depth_annotations" --split_json "splits_rgbd.json" --variant baseline --weights "weights_baseline/best.pt" --split val --batch_size 16 --num_workers 2 --target_size 256 --device cuda --out_dir "results_baseline"
```

### 5.2 Innovation (val)

```bash
python -m src.evaluate --data_root "..collated_dataset/RGB_depth_annotations" --split_json "splits_rgbd.json" --variant innovation --weights "weights_innovation/best.pt" --split val --batch_size 16 --num_workers 2 --target_size 256 --device cuda --out_dir "results_innovation"
```

Outputs:
- `results_*/metrics_val.json`

---

## 6. Independent test set (test-only split)

If you have an independent test directory structured as:

```
test/
  G01_call/clip*/...
  ...
```

Create a test-only split file:

```bash
python -m src.make_split --data_root "test" --out "splits_test.json" --test_only
```

Then evaluate:

### 6.1 Baseline (test)

```bash
python -m src.evaluate --data_root "test" --split_json "splits_test.json" --variant baseline --weights "weights_baseline/best.pt" --split test --batch_size 16 --num_workers 2 --target_size 256 --device cuda --out_dir "results_baseline"
```

### 6.2 Innovation (test)

```bash
python -m src.evaluate --data_root "test" --split_json "splits_test.json" --variant innovation --weights "weights_innovation/best.pt" --split test --batch_size 16 --num_workers 2 --target_size 256 --device cuda --out_dir "results_innovation"
```

Outputs:
- `results_*/metrics_test.json`

---

## 7. Visualisation

This script saves overlays (RGB, depth, GT overlay, prediction overlay), confusion matrices, and optional curves.

### 7.1 Validation visualisation

Baseline:

```bash
python -m src.visualise --data_root "..collated_dataset/RGB_depth_annotations" --split_json "splits_rgbd.json" --variant baseline --weights "weights_baseline/best.pt" --split val --out_dir "results_baseline" --num_samples 24 --batch_size 16 --num_workers 2 --target_size 256 --device cuda --plot_curves
```

Innovation:

```bash
python -m src.visualise --data_root "..collated_dataset/RGB_depth_annotations" --split_json "splits_rgbd.json" --variant innovation --weights "weights_innovation/best.pt" --split val --out_dir "results_innovation" --num_samples 24 --batch_size 16 --num_workers 2 --target_size 256 --device cuda --plot_curves
```

### 7.2 Test visualisation (innovation)

```bash
python -m src.visualise --data_root "test" --split_json "splits_test.json" --variant innovation --weights "weights_innovation/best.pt" --split test --out_dir "results_innovation" --num_samples 24 --batch_size 16 --num_workers 2 --target_size 256 --device cuda
```

Outputs:
- `results_*/visuals/<split>/overlay_<split>_*.png`
- `results_*/confusion_<split>.png`
- `results_*/curves.png` (when `--plot_curves` is set)

---


## Notes

- All experiments are reproduced on **Linux + GPU**.
- For a fair comparison, baseline and innovation use the same split file, preprocessing, optimiser settings, and early stopping rule.
- If dataloading is slow, try `--num_workers 0` or reduce `--batch_size`.
