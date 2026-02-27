# U-Trans + CCT  
## Texas Example (Polarity Classification)

This repository provides a complete example of training and testing a **seismic polarity classification model** using:

- **U-Trans foundation backbone**
- **Compact Convolutional Transformer (CCT)-based polarity classification model**
- **Texas subset of the Texas dataset**

---

# 📚 Dataset

## TXED (Texas Seismic Dataset)

Texas is a large-scale seismic waveform dataset containing:

- 3-component waveform recordings  
- Earthquake and noise traces  
- Metadata including arrival times and event properties  

In this example:

- Only the **Texas region subset** is used.
- Each waveform has a fixed length of **6000 samples**.
- Input shape is **(6000, 3)** representing three-component seismic data.
- Event metadata includes ground-truth first-motion polarity labels.

---

# 🎯 Purpose of This Example

This pipeline demonstrates how to:

- Train a polarity classification model on Texas Texas traces  
- Classify first-motion polarity (e.g., Up / Down)  
- Evaluate classification performance  
- Test the trained model on a held-out test set  

This setup reproduces the Texas Texas experiment using the **U-Trans + CCT architecture**.

---

# 📂 Required Files

The following files must be prepared before running the example:

### 1️⃣ Dataset File

`DataCollected`  
→ HDF5 file containing Texas Texas traces  

Each trace must be stored under its trace ID as a group:

```
<Trace_ID>/
    └── data  (6000 × 3 waveform array)
```

The HDF5 metadata must also include true polarity labels.

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

EqT_utils_Polarity_Texas.py
    DataGenerator
    Learning rate scheduler
    Dataset utilities for Texas polarity task
```

---

# 🧠 Model Architecture

This experiment combines:

## 🔹 U-Trans Foundation Backbone

- U-Net encoder–decoder structure  
- Transformer bottleneck representation  
- Learns generalized seismic waveform representations  

## 🔹 CCT Polarity Classification Model

- Convolutional tokenization  
- Compact transformer encoder layers  
- Sequence pooling  
- Fully connected classification output  

Final output:

```
Polarity class label (e.g., Up / Down)
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
- Train the classification model  
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

- Predicted polarity labels  
- Classification performance metrics  

---

## Step 3 — Post-Processing & Evaluation

Use:

```
03_read/
```

This module:

- Reads model predictions  
- Computes classification metrics  
- Calculates accuracy and confusion matrix  
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
Classification (seismic polarity prediction)
```

---

# 📊 Evaluation

Typical classification metrics include:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrix  

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
✔ Seismic polarity classification  
✔ Performance evaluation  

Using the **U-Trans foundation model combined with a Compact Convolutional Transformer (CCT) classification model** on the **Texas subset of Texas**.

---

# 📎 Citation

If you use this implementation, please cite:

**U-Trans: a foundation model for seismic waveform representation and enhanced downstream earthquake tasks**  
DOI: 10.1038/s41598-026-41454-x
