import csv
import random
import numpy as np
import rasterio
import cv2  # Required for resizing
from pathlib import Path
from code.config import IMG_HEIGHT, IMG_WIDTH

# === Load label/S1/S2/DEM filenames from CSV ===
def load_data(csv_path, split_ratio=0.2):
    """
    Load training file tuples from a CSV and split into train/val.
    Each row must contain: label_file, s1_file, s2_file, dem_file
    """
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) 
        for row in reader:
            if len(row) != 4:
                print(f"⚠️ Skipping invalid row: {row}")
                continue
            rows.append(tuple(row))

    random.shuffle(rows)
    split_index = int(len(rows) * (1 - split_ratio))
    return rows[:split_index], rows[split_index:]

# === Normalize DEM to [0, 1] ===
def normalize_dem(dem_array, min_val=0, max_val=3000):
    """
    Normalize raw DEM values to range [0, 1]
    """
    return np.clip((dem_array - min_val) / (max_val - min_val), 0, 1)

# === Load and resize DEM ===
def load_resized_dem(dem_path, target_shape=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Reads a DEM raster, resizes to (IMG_HEIGHT, IMG_WIDTH), and normalizes
    """
    with rasterio.open(dem_path) as src:
        dem = src.read(1)  
    dem = np.nan_to_num(dem)
    dem_resized = cv2.resize(dem, target_shape, interpolation=cv2.INTER_AREA)
    return normalize_dem(dem_resized).reshape(target_shape + (1,))
