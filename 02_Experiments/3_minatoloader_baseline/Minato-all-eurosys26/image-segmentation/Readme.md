# Workload: 3D U-Net on KITS19 Dataset
This is an overview of how to run this image segmentation workload. 


This benchmark represents a **3D medical image segmentation task** using the [2019 Kidney Tumor Segmentation Challenge (KiTS19)](https://kits19.grand-challenge.org/) dataset.  
The model is a [U-Net3D](https://arxiv.org/pdf/1606.06650.pdf) variant based on [No New-Net](https://arxiv.org/pdf/1809.10483.pdf).

---

## 1. Dataset

- **Name:** KITS19  
- **Description:** 3D CT scans of kidneys with tumor segmentation labels.  
- **Source:** [https://kits19.grand-challenge.org/](https://kits19.grand-challenge.org/)  

### 2. Preprocessing Pipeline  
To improve model generalization, the training pipeline applies the following transformations:

- **Random Flip** – randomly flips images to introduce spatial variability.  
- **Cast** – converts images to `float32` and labels to `uint8`.  
- **Random Brightness Augmentation** – adjusts brightness by up to ±30% with a 10% probability.  
- **Gaussian Noise** – injects Gaussian noise (mean = 0.0, std = 0.1) with a 10% probability.  

---

## 3. Workload Description

The workload evaluates segmentation accuracy on 3D volumes using:

- **Baselines:**
  1. `PyTorch DataLoader`  
  2. `NVIDIA DALI`

- **Proposed System:**
  - `SpeedyLoader`

---

## Folder Structure

- **Async/** → SpeedyLoader (our system)  
- **imseg/** → PyTorch default DataLoader  
- **nvidia-dali/** → NVIDIA DALI  

All three folders share the same execution process.  
You can build and run the workloads either with **Docker** or **Singularity**.


## 4. Directions

### Steps to Configure Machine

1. Go to the repo:
  ```bash
  cd image-segmentation
  ```

2. Build the U-Net3D container.
  - Option A – Docker
    ```bash
    docker build -t unet3d .
    ```
  - Option B – Singularity
    ```bash
    singularity build unet3d.sif singularity.def
    ```

## 5. Steps to Download and Verify Data

Download the KiTS19 dataset:
```bash 
mkdir raw-data-dir && cd raw-data-dir
git clone https://github.com/neheller/kits19
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
```

Data will be downloaded to raw-data-dir/kits19/data.

## 6.Start an interactive session in the container:

Docker example:
```bash
mkdir data results
docker run --ipc=host -it --rm --runtime=nvidia \
    -v RAW-DATA-DIR:/raw_data \
    -v PREPROCESSED-DATA-DIR:/data \
    -v RESULTS-DIR:/results \
    unet3d:latest /bin/bash
```

Singularity example:
```bash
mkdir data results
singularity exec --nv \
    -B RAW-DATA-DIR:/raw_data \
    -B PREPROCESSED-DATA-DIR:/data \
    -B RESULTS-DIR:/results \
    unet3d.sif /bin/bash
```

Preprocess the dataset:
```bash 
python3 preprocess_dataset.py --data_dir /raw_data --results_dir /data
```
Run training with:
```bash 
bash run_and_time.sh <SEED>
```