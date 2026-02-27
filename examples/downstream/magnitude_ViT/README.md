# U-Trans + ViT  
## California STEAD Example

This repository provides a complete example of training and testing an **earthquake magnitude estimation model** using:

- **U-Trans foundation backbone**
- **Vision Transformer (ViT)-based magnitude regression model**
- **California subset of the STEAD dataset**

---

# 📚 Dataset

## STEAD (Stanford Earthquake Dataset)

STEAD is a large-scale seismic waveform dataset containing:

- 3-component waveform recordings  
- Earthquake and noise traces  
- Metadata including arrival times and event properties  

In this example:

- Only the **California region subset** is used.
- Each waveform has a fixed length of **6000 samples**.
- Input shape is **(6000, 3)** representing three-component seismic data.
- Event metadata includes ground-truth earthquake magnitude values.

---

# 🎯 Purpose of This Example

This pipeline demonstrates how to:

- Train an earthquake magnitude regression model on California STEAD traces  
- Predict earthquake magnitudes  
- Evaluate regression performance  
- Test the trained model on a held-out test set  

This setup reproduces the California STEAD experiment using the **U-Trans + ViT architecture**.

---

# 📂 Required Files

The following files must be prepared before running the example:

### 1️⃣ Dataset File

`DataCollected`  
→ HDF5 file containing STEAD California traces  

Each trace must be stored under its trace ID as a group:

```
<Trace_ID>/
    └── data  (6000 × 3 waveform array)
```

The HDF5 metadata must also include true event magnitude values.

---

### 2️⃣ Train / Validation / Test Splits

- `train_Events.npy` → Training trace IDs  
- `valid_Events.npy` → Validation trace IDs  
- `test_Events.npy` → Test trace IDs  

Each `.npy` file should contain trace IDs that match keys inside the HDF5 file.

---

# 🏗 Project Structure

```
01_train/
    Trainer script (trainer1) for model training

02_test/
    Tester script (tester1) for model inference and evaluation

03_read/
    Post-processing and evaluation utilities

EqT_utils_Mag_California.py
    DataGenerator
    Learning rate scheduler
    Dataset utilities for the California magnitude task
```

---

# 🧠 Model Architecture

This experiment combines:

## 🔹 U-Trans Foundation Backbone

- U-Net encoder–decoder structure  
- Transformer bottleneck representation  
- Learns generalized seismic waveform representations  

## 🔹 ViT Magnitude Model

- Patch tokenization  
- Transformer encoder layers  
- Global feature aggregation  
- Fully connected regression output  

Final output:

```
Relative Event magnitude value (continuous regression)
```

---

# 🔄 Workflow

## Step 1 — Train Model

Run training from:

```
01_train/
```

This will:

- Load training and validation IDs  
- Train the regression model  
- Save best weights in:

```
<output_name>_outputs/models/
```

---

## Step 2 — Run Testing

Run inference from:

```
02_test/
```

Tester parameters include:

- `input_model` → path to trained weights  
- `input_hdf5` → dataset path  
- `input_testset` → test IDs  

Outputs include:

- Predicted event magnitudes  
- Magnitude error metrics  

---

## Step 3 — Post-Processing & Evaluation

Use:

```
03_read/
```

This module:

- Reads model predictions  
- Computes regression metrics  
- Calculates error statistics  
- Produces summary evaluation results  

---

# ⚙️ Configuration

### Input Shape
```
(6000, 3)
```

### Normalization
```
std   # per-trace standard deviation normalization
```

### Task Type
```
Regression (earthquake magnitude prediction)
```

---

# 📊 Evaluation

Typical regression metrics include:

- Mean Absolute Error (MAE)  

---

# 🖥 Hardware Notes

- GPU recommended  
- Batch size depends on GPU memory  
- Generator mode allows large datasets without full RAM loading  

---

# 📌 Summary

This repository demonstrates a full end-to-end experiment for:

✔ Training  
✔ Validation  
✔ Testing  
✔ Earthquake magnitude regression  
✔ Performance evaluation  

Using the **U-Trans foundation model combined with a Vision Transformer (ViT) regression model** on the **California subset of STEAD**.

---

# 📎 Citation

If you use this implementation, please cite:

**U-Trans: a foundation model for seismic waveform representation and enhanced downstream earthquake tasks**  
DOI: 10.1038/s41598-026-41454-x
