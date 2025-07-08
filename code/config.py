from pathlib import Path

# === Image Shape ===
IMG_HEIGHT = 128
IMG_WIDTH = 128

# === Channel Counts ===
s1_ch = 2   # Sentinel-1: VV, VH
s2_ch = 4   # Sentinel-2: B4, B3, B2, B8
dem_ch = 1  # Single band elevation

# === Sentinel-2 Normalization ===
S2_MAX = 3000

# === File Paths ===
ROOT_DIR = Path(r"C:\Users\pivo\Project_GEP\height-estimation-morocco")

S1_PATH = ROOT_DIR / "data/raw/sentinel 1"
S2_PATH = ROOT_DIR / "data/raw/sentinel 2"
DEM_PATH = ROOT_DIR / "data/ref/dem"
LABEL_PATH = ROOT_DIR / "data/ref/heigth"  # or /height if renamed
PATCHES_PATH = ROOT_DIR / "data/patches"

CSV_PATH = ROOT_DIR / "data/splits/train_clean.csv"
OUTPUT_PATH = ROOT_DIR / "outputs"

# === Training Config ===
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
