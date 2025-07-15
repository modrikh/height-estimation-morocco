import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

# === Config ===
CSV_PATH = Path("data/splits/train_clean.csv")  # Update if needed
DATA_ROOT = Path("data")                    # Where your .tif files are stored
LABEL_PATH = DATA_ROOT / "ref" / "height"
S1_PATH = DATA_ROOT /"raw" / "sentinel 1"
S2_PATH = DATA_ROOT /"raw" / "sentinel 2"
DEM_PATH = DATA_ROOT /"raw" / "dsm"

# === Load CSV ===
df = pd.read_csv(CSV_PATH)

# === Pick a City ===
row = df.iloc[12]  # e.g. Agadir
label_file, s1_file, s2_file, dem_file = row

# === Read Images ===
def read_tif(path):
    with rasterio.open(path) as src:
        img = src.read()
        img = np.moveaxis(img, 0, -1)  # CHW -> HWC
    return np.nan_to_num(img)

label = read_tif(LABEL_PATH / label_file)
s1 = read_tif(S1_PATH / s1_file)
s2 = read_tif(S2_PATH / s2_file)
dem = read_tif(DEM_PATH / dem_file)

# === Normalize helpers ===
def norm(x):
    x = np.nan_to_num(x)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)

# === Plot ===
plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
plt.title("Sentinel-1 VV")
plt.imshow(norm(s1[:, :, 0]), cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Sentinel-1 VH")
plt.imshow(norm(s1[:, :, 1]), cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("DEM")
plt.imshow(norm(dem[:, :, 0]), cmap='terrain')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Sentinel-2 RGB")
plt.imshow(norm(s2[:, :, :3]))  # B4, B3, B2
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Sentinel-2 NIR (B8)")
plt.imshow(norm(s2[:, :, 3]), cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("Building Height (label)")
plt.imshow(label[:, :, 0], cmap='viridis')
plt.colorbar(fraction=0.03)
plt.axis('off')

plt.tight_layout()
plt.show()
