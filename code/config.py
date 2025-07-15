
from pathlib import Path

# === Définition du chemin racine ===
ROOT_DIR = Path(r"C:\Users\pivo\Project_GEP\height-estimation-morocco")

# === Forme des Images et Canaux ===
IMG_HEIGHT = 256
IMG_WIDTH = 256
s1_ch = 2
s2_ch = 4
dem_ch = 1
# === Normalisation ===
S2_MAX = 3000

# === Chemins des Fichiers (EXPLICITES ET CLAIRS) ===
S1_PATH = ROOT_DIR / "data/raw/sentinel 1"
S2_PATH = ROOT_DIR / "data/raw/sentinel 2"
DEM_PATH = ROOT_DIR / "data/raw/dsm"

# Le chemin unique vers vos fichiers WorldCover
LABEL_PATH = ROOT_DIR / "data/ref/height" 

CSV_PATH = ROOT_DIR / "data/splits/train_clean.csv"
OUTPUT_PATH = ROOT_DIR / "outputs"

# === Configuration de l'Entraînement ===
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4

# === Configuration Multi-Tâches ===
NUM_CLASSES = 2
BUILT_UP_CLASS_ID = 50
DEFAULT_BUILDING_HEIGHT = 10.0
LOSS_WEIGHT_REGRESSION = 0.6
LOSS_WEIGHT_CLASSIFICATION = 0.4