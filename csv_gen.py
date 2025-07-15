# code/csv_gen.py (Version finale pour CLASSIFICATION SEULEMENT)
import csv
from pathlib import Path

# === File Paths ===
ROOT_DIR = Path(r"C:\Users\pivo\Project_GEP\height-estimation-morocco")

S1_PATH = ROOT_DIR / "data/raw/sentinel 1"
S2_PATH = ROOT_DIR / "data/raw/sentinel 2"
DEM_PATH = ROOT_DIR / "data/raw/dsm"
LABEL_PATH = ROOT_DIR / "data/ref/height"
CSV_PATH = ROOT_DIR / "data/splits/train_clean.csv"
rows = []

# Le dossier LABEL_PATH contient maintenant les fichiers WorldCover
print(f"Analyse des fichiers de labels WorldCover dans : {LABEL_PATH}")
for label_path in LABEL_PATH.glob("*_WorldCover_10m.tif"):
    city = label_path.stem.replace("_WorldCover_10m", "")
    
    s1_file = S1_PATH / f"{city}_S1.tif"
    s2_file = S2_PATH / f"{city}_S2.tif"
    dem_file = DEM_PATH / f"{city}_DSM.tif"

    if all([s1_file.exists(), s2_file.exists(), dem_file.exists()]):
        # Ordre : (label_classe, s1, s2, dem)
        rows.append([label_path.name, s1_file.name, s2_file.name, dem_file.name])
        print(f"✅ Correspondance trouvée pour : {city}")
    else:
        print(f"⚠️ {city} ignoré : Fichiers d'entrée manquants.")

with open(CSV_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["label_fname", "s1_fname", "s2_fname", "dem_fname"])
    writer.writerows(rows)

print(f"\n✅ CSV généré. Sauvegardé dans {CSV_PATH} avec {len(rows)} entrées.")