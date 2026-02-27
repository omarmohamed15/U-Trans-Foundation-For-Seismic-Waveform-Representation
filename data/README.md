# Data Preparation  
## Combined Seismic Dataset (STEAD + INSTANCE + TXED)

This folder contains the data preparation pipeline used to construct a unified seismic dataset for training U-Trans foundation and downstream models.

---

# 📚 Source Datasets

The combined dataset is built from the following publicly available seismic waveform datasets:

- **STEAD**  
  https://github.com/smousavi05/STEAD  

- **INSTANCE**  
  https://github.com/INGV/instance  

- **TXED**  
  https://github.com/chenyk1990/txed  

Each dataset contains 3-component seismic waveform recordings along with associated metadata.

---

# 📥 Download Instructions

1. Download the three datasets from their official repositories.
2. Create a folder in your project directory called:

```
Data_Seismic
```

3. Place all downloaded dataset files inside the `Data_Seismic` folder.

Your structure should look like:

```
Data_Seismic/
    ├── STEAD/
    ├── INSTANCE/
    └── TXED/
```

Make sure the internal dataset structures remain unchanged.

---

# 🎯 Purpose

The goal of the `data` script is to:

- Load waveform data from STEAD, INSTANCE, and TXED  
- Standardize waveform length and format  
- Harmonize metadata fields  
- Merge all datasets into a single HDF5 file  

The final combined dataset is saved as:

```
DataCollected
```

This unified file is used across all downstream tasks, including:

- P-wave picking  
- S-wave picking  
- Event location  
- Magnitude estimation  
- Polarity classification  

---

# 📂 Output File Structure

The generated HDF5 file:

```
DataCollected
```

Each seismic trace is stored under its trace ID as a group:

```
<Trace_ID>/
    ├── data  (6000 × 3 waveform array)
    └── attrs (metadata including arrivals, magnitude, location, polarity, etc.)
```

Waveforms are standardized to:

```
(6000, 3)
```

Where:

- 6000 = number of time samples  
- 3 = three-component waveform  

---

# 🔄 Processing Steps

The `data` script performs the following:

1. Read waveform files from `Data_Seismic/`.
2. Standardize sampling and format.
3. Resize or trim waveforms to 6000 samples.
4. Extract and unify metadata fields.
5. Assign consistent trace IDs.
6. Merge all traces into one HDF5 structure.
7. Save final dataset as `DataCollected`.

---

# ⚙️ Requirements

Before running the script:

- Ensure all three datasets are downloaded.
- Place them inside the `Data_Seismic` folder.
- Update file paths inside the script if needed.
- Install required packages:
  - numpy
  - h5py
  - tqdm
  - pandas (if metadata processing is used)

---

# 📌 Notes

- Train/validation/test splits are created separately using `.npy` ID files.
- Dataset balancing (if required) should be handled independently.
- This combined dataset enables foundation pretraining and all downstream experiments from a unified source.

---

# 📎 References

If you use the original datasets, please cite:

- STEAD: Mousavi et al., 2019  
- INSTANCE: Michelini et al., 2021  
- TXED: Chen et al., 2021  

For the foundation model, please cite:

**U-Trans: a foundation model for seismic waveform representation and enhanced downstream earthquake tasks**  
DOI: 10.1038/s41598-026-41454-x
