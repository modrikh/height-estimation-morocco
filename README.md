## Building Height Estimation in Morocco Using Sentinel-1, Sentinel-2, and DEM

This project implements a deep learning model to estimate per-pixel **building height maps** from satellite imagery. It combines **Sentinel-1 (SAR)**, **Sentinel-2 (MSI)**, and **DEM** inputs in a **ResNet-based multi-branch U-Net** (MBHR-Net), using precomputed height labels from DSM − DEM.

---

## Project Structure

```
├── data/
│   ├── raw/
│   │   ├── sentinel 1/        # Full city SAR images
│   │   ├── sentinel 2/        # Full city MSI images
│   ├── ref/
│   │   ├── dem/               # FABDEM
│   │   ├── dsm/               # AW3D30
│   │   └── heigth/            # Any availabe reference dataset
│   └── splits/
│       └── train_clean.csv    # CSV of matched tiles
├── code/
│   ├── models/
│   │   └── architectures.py   # MBHR-ResNet model
│   ├── training/
│   │   └── train.py           # Training script
│   ├── utils.py               # Helpers for DEM, CSV, resizing
│   ├── augment.py             # S1/S2/DEM loading + scaling
│   ├── generators.py          # Custom tf.keras generator
├── outputs/
│   └── checkpoints, logs, etc.
```

---

## Data Preparation

### 1. **Download Source Data**

Use [Google Earth Engine](https://code.earthengine.google.com/) to export:

* **Sentinel-1 (VV, VH)** – GRD monthly average
* **Sentinel-2 (B2, B3, B4, B8)** – monthly median cloud-free
* **AW3D30** (DSM) and **FABDEM** (DEM)

### 2. **Precompute Building Height**

In GEE or Python:

```
building_height = DSM - DEM
```

Export at 10m resolution as GeoTIFF.

### 3. **Place Files**

For each city, place:

* `City_S1_2023.tif` in `data/raw/sentinel 1/`
* `City_S2_2023.tif` in `data/raw/sentinel 2/`
* `City_DEM_2023.tif` in `data/ref/dem/`
* `City_Building_Height_10m.tif` in `data/ref/heigth/`

---

## Generate CSV for Training

Run this script to generate the file `train_clean.csv`:

```bash
python csv_gen.py
```

It will output:

```csv
label_file,s1_file,s2_file,dem_file
Casablanca_Building_Height_10m.tif,Casablanca_S1_2023.tif,Casablanca_S2_2023.tif,Casablanca_DEM_2023.tif
...
```

---

## Model: MBHR-ResNet

* Multi-branch encoder:

  * **ResNet50** for S1 (2 bands) and S2 (4 bands)
  * **Shallow CNN** for DEM (1 band)
* U-Net-style decoder
* Final output: **128×128×1 regression map**

---

## Train the Model ( using a virtual enviroment ) 

```bash
python -m code.training.train.py
```

This will:

* Load file paths from `train_clean.csv`
* Train on S1, S2, DEM → label (building height)
* Save the best model to `outputs/checkpoints/best_model.h5`
* Plot training curves
* Print validation metrics (RMSE, MAE, R²)

---

## Evaluation Metrics

Evaluated per-pixel on the validation set:

* **RMSE** – root mean squared error
* **MAE** – mean absolute error
* **R²** – coefficient of determination

After training, final validation metrics are shown and saved.

---

## Dependencies

* Python 3.8+
* TensorFlow ≥ 2.9
* NumPy, Rasterio, OpenCV, Matplotlib, Pandas

Install dependencies:

```bash
pip install -r requirements.txt
```
