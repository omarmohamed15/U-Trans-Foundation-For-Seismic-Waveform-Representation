# 🧪 Training (Self-Supervised Reconstruction)

This folder contains the **foundation training pipeline** used to train U-Trans via self-supervised waveform reconstruction.

During training, input waveforms from the combined dataset (`DataCollected`) are deliberately corrupted in both the **time domain** and **frequency domain**, and the model is trained to reconstruct the original clean signal.

---

# 📂 Folder Contents

The `train/` directory includes:

- `U-Trans_Train.ipynb`  
  Main notebook used to run foundation training.

- `EqT_utils_Recon.py`  
  Utility functions for:
  - Waveform preprocessing  
  - Time-domain corruption (masking, noise injection, gaps)  
  - Frequency-domain corruption (spectral masking / perturbation)  
  - Data generator logic  
  - Learning rate scheduling  
  - Training callbacks  

- `IDS_Collected_Data_train.npy`  
  Trace IDs used for training.

- `IDS_Collected_Data_valid.npy`  
  Trace IDs used for validation.

- `IDS_Collected_Data_test.npy`  
  Trace IDs reserved for evaluation/testing.

---

# 📊 Dataset Dependency

Training requires the unified dataset:

```
DataCollected
```

This HDF5 file is generated using the `data/` folder scripts by combining:

- STEAD
- INSTANCE
- TXED

Each trace must be stored as:

```
<Trace_ID>/
    └── data  (6000 × 3 waveform array)
```

Waveforms are standardized to:

```
(6000, 3)
```

---

# 🧠 Training Objective

The U-Trans foundation model is trained using a **self-supervised reconstruction objective**:

1. Load clean waveform from `DataCollected`
2. Apply corruption:
   - Time-domain corruption
   - Frequency-domain corruption
3. Feed corrupted waveform into U-Trans
4. Reconstruct original waveform
5. Optimize reconstruction loss

This encourages the model to learn:

- Robust waveform representations  
- Multi-scale temporal structure  
- Spectral characteristics  
- Transferable latent embeddings  

---

# 🚀 Running Training

Open and execute:

```
train/U-Trans_Train.ipynb
```

The notebook:

- Loads ID splits from `.npy` files
- Reads waveforms from `DataCollected`
- Applies corruption operators
- Trains the U-Trans reconstruction model
- Saves trained foundation weights

---

# ✅ Outputs

Training produces:

- Model checkpoints (foundation weights)
- Training history logs
- Optional reconstruction visualizations

The resulting weights are later used for:

- Latent feature extraction
- Downstream task fine-tuning
- Seismic transfer learning experiments

---

# 📌 Notes

- Ensure `DataCollected` exists before training.
- ID split files must match trace IDs inside the HDF5 file.
- GPU acceleration is recommended.
- Batch size should be adjusted based on GPU memory.
