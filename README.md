# 🧠 Patch-based Time Series Anomaly Detection

This is a learning-oriented project for implementing patch-based time series anomaly detection models. It supports training and inference for multiple models including AutoEncoder, USAD, PatchTST-AE, TranAD, AnomalyTransformer, DCdetector, and more — all under a unified framework with consistent evaluation and visualization.

---
## 🚀 Dataset
This project uses the Server Machine Dataset (SMD) from the OmniAnomaly project and is used here for research and educational purposes only.  
🔗 Download link:
https://github.com/NetManAIOps/OmniAnomaly/tree/master/data  

To use this dataset:  

Download the raw data from the above link (specifically the server_machine folder)  

Place it under a new folder:  
```bash
dataset/raw_data/SMD/  
```
Run the preprocessing script to convert it into standardized .csv files:  
```bash
python cleandata_smd.py
```
This will generate:  
```bash
dataset/
  └── clean_data/
      └── SMD/
          ├── train/
          ├── test/
          └── label/
```
You can now train and test models using this clean dataset.


## 🚀 Getting Started

### 1. Install basic dependencies

torch>=1.10  
numpy>=1.24  
pandas>=2.0  
scikit-learn>=1.2  
matplotlib>=3.5  
joblib  
tqdm  


### 2. Train a model
Example using AutoEncoder:
```bash
python main.py --model autoencoder --machine machine-1-1 --patch_size 16
```
This command will train an AutoEncoder model using `machine-1-1` data with a patch size of 16.  
Model checkpoints will be saved in the `checkpoints/` directory.

### 3. Run inference
To evaluate the trained model and generate anomaly scores:
```bash
python inference.py --model autoencoder --machine machine-1-1 --patch_size 16  
```

This will:  
- Calculate anomaly scores on the test set  
- Apply a 95th percentile threshold  
- Use point_adjust to align predictions with anomaly segments  
- Print Precision, Recall, F1, AUC metrics  
- Save visualization plot and result JSON in the vis/ directory


### 4. Batch Run on Multiple Machines
To train and evaluate models on a batch of machines (e.g., `machine-1-1`, `machine-1-2`, ...), use:

```bash
python batch_run.py --model autoencoder --patch_size 16 --patch_mode sliding
```
```bash
machine_ids = [
  "machine-1-1", "machine-1-2", ..., "machine-3-11"
]
```
This script will:  
- Loop through all specified machine IDs
- Call main.py for training each one
- Call inference.py for evaluation
- Save per-machine JSON results in vis/
- Aggregate and compute macro-average metrics (Precision, Recall, F1, AUC)
- Export summary CSV to:
