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

Install PyTorch that matches the CUDA version on the remote GPU machine.

Example for CUDA 12.1 wheels:
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
``` 
Other packages:

```bash
pip install numpy pillow matplotlib scikit-learn tqdm
```
## 2. Data layout

The dataset is not included in the repository.

### 2.1 Collated RGB-D dataset

Expected structure:

collated_dataset/
  RGB_depth_annotations/
    SubjectA/
      G01_call/clip01/{rgb,depth,annotation}/frame_*.png
      ...
    SubjectB/
      ...
2.2 Independent test set

A common structure is:

test/
  G01_call/clip*/...
  G02_dislike/clip*/...
  ...
3. Train and evaluate (Baseline)

Run the following commands from the repository root.

3.1 Create a subject split (train and val)
python -m src.make_split \
  --data_root "../collated_dataset/RGB_depth_annotations" \
  --out "splits_rgbd.json" \
  --val_frac 0.15 \
  --seed 42
3.2 Train baseline (early stopping)
python -m src.train \
  --data_root "../collated_dataset/RGB_depth_annotations" \
  --split_json "splits_rgbd.json" \
  --variant baseline \
  --out_dir "results_baseline" \
  --weights_dir "weights_baseline" \
  --epochs 100 \
  --patience 8 \
  --min_delta 0.002 \
  --batch_size 16 \
  --num_workers 2 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --target_size 256 \
  --device cuda
3.3 Evaluate baseline (val)
python -m src.evaluate \
  --data_root "../collated_dataset/RGB_depth_annotations" \
  --split_json "splits_rgbd.json" \
  --variant baseline \
  --weights "weights_baseline/best.pt" \
  --split val \
  --batch_size 16 \
  --num_workers 2 \
  --target_size 256 \
  --device cuda \
  --out_dir "results_baseline"
3.4 Visualise baseline (val)
python -m src.visualise \
  --data_root "../collated_dataset/RGB_depth_annotations" \
  --split_json "splits_rgbd.json" \
  --variant baseline \
  --weights "weights_baseline/best.pt" \
  --split val \
  --out_dir "results_baseline" \
  --num_samples 24 \
  --batch_size 16 \
  --num_workers 2 \
  --target_size 256 \
  --device cuda \
  --plot_curves
4. Train and evaluate (Innovation)

Select the innovation by setting --variant innovation.

4.1 Train innovation
python -m src.train \
  --data_root "../collated_dataset/RGB_depth_annotations" \
  --split_json "splits_rgbd.json" \
  --variant innovation \
  --out_dir "results_innovation" \
  --weights_dir "weights_innovation" \
  --epochs 100 \
  --patience 8 \
  --min_delta 0.002 \
  --batch_size 16 \
  --num_workers 2 \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --target_size 256 \
  --device cuda
4.2 Evaluate innovation (val)
python -m src.evaluate \
  --data_root "../collated_dataset/RGB_depth_annotations" \
  --split_json "splits_rgbd.json" \
  --variant innovation \
  --weights "weights_innovation/best.pt" \
  --split val \
  --batch_size 16 \
  --num_workers 2 \
  --target_size 256 \
  --device cuda \
  --out_dir "results_innovation"
4.3 Visualise innovation (val)
python -m src.visualise \
  --data_root "../collated_dataset/RGB_depth_annotations" \
  --split_json "splits_rgbd.json" \
  --variant innovation \
  --weights "weights_innovation/best.pt" \
  --split val \
  --out_dir "results_innovation" \
  --num_samples 24 \
  --batch_size 16 \
  --num_workers 2 \
  --target_size 256 \
  --device cuda \
  --plot_curves
5. Independent test set
5.1 Create a test-only split file
python -m src.make_split \
  --data_root "../test" \
  --out "splits_test.json" \
  --test_only
5.2 Evaluate on test

Baseline:

python -m src.evaluate \
  --data_root "../test" \
  --split_json "splits_test.json" \
  --variant baseline \
  --weights "weights_baseline/best.pt" \
  --split test \
  --batch_size 16 \
  --num_workers 2 \
  --target_size 256 \
  --device cuda \
  --out_dir "results_baseline"

Innovation:

python -m src.evaluate \
  --data_root "../test" \
  --split_json "splits_test.json" \
  --variant innovation \
  --weights "weights_innovation/best.pt" \
  --split test \
  --batch_size 16 \
  --num_workers 2 \
  --target_size 256 \
  --device cuda \
  --out_dir "results_innovation"
5.3 Visualise on test

Innovation:

python -m src.visualise \
  --data_root "../test" \
  --split_json "splits_test.json" \
  --variant innovation \
  --weights "weights_innovation/best.pt" \
  --split test \
  --out_dir "results_innovation" \
  --num_samples 24 \
  --batch_size 16 \
  --num_workers 2 \
  --target_size 256 \
  --device cuda
