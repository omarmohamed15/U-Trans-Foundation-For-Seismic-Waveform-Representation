# U-Trans + EQCCT  
## California STEAD Example

This repository provides a complete example of training and testing a **P-wave picking model** using:

- **U-Trans foundation backbone**
- **EQCCT transformer-based picking head**
- **California subset of the STEAD dataset**

---

#  Dataset

## STEAD (Stanford Earthquake Dataset)

STEAD is a large-scale seismic waveform dataset containing:

- 3-component waveform recordings  
- Earthquake and noise traces  
- Metadata including arrival times and event properties  

In this example:

- Only the **California region subset** is used.
- Each waveform has a fixed length of **6000 samples**.
- Input shape is **(6000, 3)** representing three-component seismic data.

---

# ? Purpose of This Example

This pipeline demonstrates how to:

- Train a P-wave picking model on California STEAD traces  
- Evaluate model performance  
- Test on a held-out test set  
- Extract predicted P picks  
- Optionally estimate prediction uncertainty  

This setup reproduces the California STEAD experiment using the **U-Trans + EQCCT architecture**.

---

# ? Required Files

The following files must be prepared before running the example:

### 1?? Dataset File

`DataCollected`  
? HDF5 file containing STEAD California traces  

Each trace must be stored under its trace ID as a group:

```
<Trace_ID>/
    ??? data  (6000 × 3 waveform array)
```

---

### 2?? Train / Validation / Test Splits

- `train_Events.npy` ? Training trace IDs  
- `valid_Events.npy` ? Validation trace IDs  
- `test_Events.npy` ? Test trace IDs  

Each `.npy` file should contain a list/array of trace IDs that match the keys inside the HDF5 file.

---

# ? Project Structure

```
01_train/
    Trainer script (trainer1) for model training

02_test/
    Tester script (tester1) for model inference and evaluation

03_read_picks/
    Peak detection and P-pick extraction utilities

EQCCT_P_utils.py
    DataGenerator
    Learning rate scheduler
    Dataset utilities
```

---

# ? Model Architecture

This experiment combines:

## ? U-Trans Foundation Backbone

- U-Net encoder?decoder structure  
- Transformer bottleneck representation  
- Learns generalized seismic waveform representations  

## ? EQCCT P-wave Head

- Convolutional preprocessing  
- Patch tokenization  
- Transformer layers  
- Sample-wise P probability output (6000 × 1)  

---

# ? Workflow

## Step 1  Train Model

Run training from:

```
01_train/
```

This will:

- Load training and validation IDs  
- Train the model  
- Save best weights in:

```
<output_name>_outputs/models/
```

---

## Step 2  Run Testing

Run inference from:

```
02_test/
```

Tester parameters include:

- `input_model` ? path to trained weights  
- `input_hdf5` ? dataset path  
- `input_testset` ? test IDs  
- `P_threshold` ? minimum probability for P-pick  
- `detection_threshold` ? event detection threshold  
- `estimate_uncertainty` ? enable Monte Carlo sampling (optional)  

Outputs include:

- Prediction probabilities  
- Extracted P picks  
- Evaluation metrics  
- Optional diagnostic plots  

---

## Step 3  Extract P Picks

Use:

```
03_read_picks/
```

This module:

- Detects peaks in probability curves  
- Applies probability thresholds  
- Selects highest-confidence pick per trace  
- Outputs final P-pick arrays  

---

# ?? Configuration

### Input Shape
```
(6000, 3)
```

### Normalization
```
std   # per-trace standard deviation normalization
```

### Mode
```
generator   # memory-efficient batch processing
```

---

# ? Threshold Selection

- `P_threshold` controls pick sensitivity  
- Lower ? more picks (higher recall, lower precision)  
- Higher ? fewer picks (higher precision, lower recall)  

Recommended range:

```
0.05 ? 0.20
```

---

# ? Uncertainty Estimation (Optional)

If enabled:

```
estimate_uncertainty=True
number_of_sampling=5+
```

The model performs multiple stochastic forward passes and computes prediction variance.

This increases computation time.

---

#  Hardware Notes

- GPU recommended  
- Batch size depends on GPU memory  
- Generator mode allows large datasets without full RAM loading  

---

#  Summary

This repository demonstrates a full end-to-end experiment for:

? Training  
? Validation  
? Testing  
? P-wave picking  
? Evaluation  

Using the **U-Trans foundation model combined with an EQCCT transformer head** on the **California subset of STEAD**.

---

#  Citation

If you use this implementation, please cite:

**U-Trans: a foundation model for seismic waveform representation and enhanced downstream earthquake tasks**  
DOI: 10.1038/s41598-026-41454-x

---
