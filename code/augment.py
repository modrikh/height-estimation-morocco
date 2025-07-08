# augment.py

import cv2 as cv
import numpy as np
import rasterio
from code.config import IMG_HEIGHT, IMG_WIDTH, s1_ch, s2_ch, S2_MAX

# === Sentinel-1: Load + Scale ===
def scale_img_s1(matrix):
    min_values = np.array([-23, -28])  # VV, VH
    max_values = np.array([0, -5])
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float32)
    matrix = np.nan_to_num((matrix - min_values) / (max_values - min_values))
    return matrix.reshape(w, h, d).clip(0, 1)

def GRD_toRGB_S1(S1_PATH, fname):
    path_img = S1_PATH / fname
    with rasterio.open(path_img) as sar:
        sar_img = sar.read((1, 2))  # VV, VH
    sar_img = np.moveaxis(sar_img, 0, -1)  # (H, W, C)
    vv = cv.resize(sar_img[:, :, 0], (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_AREA)
    vh = cv.resize(sar_img[:, :, 1], (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_AREA)
    x_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, s1_ch), dtype=np.float32)
    x_img[:, :, 0] = vv
    x_img[:, :, 1] = vh
    return scale_img_s1(x_img)

# === Sentinel-2: Load + Scale ===
def scale_img_s2(matrix, max_vis):
    min_values = np.array([0, 0, 0, 0])  # B4, B3, B2, B8
    max_values = np.array([max_vis] * 4)
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float32)
    matrix = np.nan_to_num((matrix - min_values) / (max_values - min_values))
    return matrix.reshape(w, h, d).clip(0, 1)

def GRD_toRGB_S2(S2_PATH, fname, max_vis):
    path_img = S2_PATH / fname
    with rasterio.open(path_img) as src:
        s2_img = src.read((1, 2, 3, 4))  # B4, B3, B2, B8
    s2_img = np.moveaxis(s2_img, 0, -1)
    x_img = np.zeros((IMG_HEIGHT, IMG_WIDTH, s2_ch), dtype=np.float32)
    for i in range(s2_ch):
        x_img[:, :, i] = cv.resize(s2_img[:, :, i], (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_AREA)
    return scale_img_s2(x_img, max_vis)

# === DEM: Load + Normalize ===
def load_dem(DEM_PATH, fname, max_elevation=1000.0):
    """
    Loads and normalizes a DEM GeoTIFF image.
    """
    path_img = DEM_PATH / fname
    with rasterio.open(path_img) as src:
        dem = src.read(1)  # single band
    dem = cv.resize(dem, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv.INTER_AREA)
    dem = np.nan_to_num(dem.astype(np.float32))
    dem = np.clip(dem / max_elevation, 0, 1)
    return dem[..., np.newaxis]  # shape: (H, W, 1)
