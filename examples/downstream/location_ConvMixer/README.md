# U-Trans + ConvMixer  
## California STEAD Example

This repository provides a complete example of training and testing an **earthquake location model** using:

- **U-Trans foundation backbone**
- **ConvMixer-based location regression model**
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
- Event metadata includes ground-truth earthquake location values.

---

# 🎯 Purpose of This Example

This pipeline demonstrates how to:

- Train an earthquake location regression model on California STEAD traces  
- Predict event hypocenter (e.g., latitude, longitude, depth)  
- Evaluate spatial prediction performance  
- Test the trained model on a held-out test set  

This setup reproduces the California STEAD experiment using the **U-Trans + ConvMixer architecture**.

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

The HDF5 metadata must also include true event location values.

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

EqT_utils_Loc_California.py
    DataGenerator
    Learning rate scheduler
    Dataset utilities for California location task
```

---

# 🧠 Model Architecture

This experiment combines:

## 🔹 U-Trans Foundation Backbone

- U-Net encoder–decoder structure  
- Transformer bottleneck representation  
- Learns generalized seismic waveform representations  

## 🔹 ConvMixer Location Model

- Patch-based convolutional mixing  
- Depthwise spatial mixing  
- Channel mixing layers  
- Global feature aggregation  
- Fully connected regression output  

Final output:

```
Relative Event location vector (e.g., latitude, longitude, depth)
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
- Depth normalization range (`max_d`, `min_d`)  

Outputs include:

- Predicted event coordinates  
- Location error metrics  

---

## Step 3 — Post-Processing & Evaluation

Use:

```
03_read/
```

This module:

- Reads model predictions  
- Computes regression metrics  
- Calculates spatial error statistics  
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
Regression (earthquake location prediction)
```

### Depth Normalization (California STEAD)

```
max_d = 114.5
min_d = 1e-5
```

These values must match the dataset depth range used during training.

---

# 📊 Evaluation

Typical regression metrics include:

- Mean Absolute Error (MAE)  
- Root Mean Square Error (RMSE)  
- Epicentral distance error  

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
✔ Earthquake location regression  
✔ Spatial evaluation  

Using the **U-Trans foundation model combined with a ConvMixer regression model** on the **California subset of STEAD**.

---

# 📎 Citation

If you use this implementation, please cite:

**U-Trans: a foundation model for seismic waveform representation and enhanced downstream earthquake tasks**  
DOI: 10.1038/s41598-026-41454-x

---
