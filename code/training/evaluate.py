
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from keras.models import load_model
import rasterio
import cv2

# === Imports depuis votre projet ===
from code.config import *
from code.metrics import rmse_metric, mae_metric, r2_metric_buildings_only
from code.losses import combined_mse_cosine_loss
from code.augment import GRD_toRGB_S1, GRD_toRGB_S2, load_dem

# === Fonctions Utilitaires (Helpers) ===
def extract_center_patch(array, size=128):
    if array.ndim == 3: h, w, _ = array.shape
    else: h, w = array.shape
    center_h, center_w = h // 2, w // 2
    half = size // 2
    if array.ndim == 3:
        return array[center_h - half:center_h + half, center_w - half:center_w + half, :]
    else:
        return array[center_h - half:center_h + half, center_w - half:center_w + half]

# === CONFIGURATION DE L'ÉVALUATION ===
city = "Casablanca"
max_height_denorm = 100.0
patch_size_to_extract = 256
model_path = OUTPUT_PATH / "checkpoints_multitask" / "best_model.keras"

# === Chargement du Modèle Multi-Tâches ===
print(f"Chargement du modèle depuis : {model_path}")
model = load_model(
    model_path,
    custom_objects={
        "combined_mse_cosine_loss": combined_mse_cosine_loss,
        "rmse_metric": rmse_metric,
        "mae_metric": mae_metric,
        "r2_metric_buildings_only": r2_metric_buildings_only
    }
)
print("Modèle chargé avec succès.")

# === 1. Lire et Pré-traiter les Données ===
print("Lecture et pré-traitement des données...")
s1_full = GRD_toRGB_S1(S1_PATH, f"{city}_S1.tif")
s2_full = GRD_toRGB_S2(S2_PATH, f"{city}_S2.tif", S2_MAX)
dem_full = load_dem(DEM_PATH, f"{city}_DSM.tif")


from code.generators import MultiTaskPatchGenerator
temp_gen = MultiTaskPatchGenerator(file_tuples=[], batch_size=1, patch_size=1)

worldcover_fname = f"{city}_WorldCover_10m.tif"
worldcover_map_full = temp_gen.read_worldcover_map(worldcover_fname)

true_height_full = temp_gen.create_height_label_from_map(worldcover_map_full)

# === 2. Extraire le Patch Central de Toutes les Données ===
print(f"Extraction du patch central de {patch_size_to_extract}x{patch_size_to_extract}...")
s1_patch = extract_center_patch(s1_full, size=patch_size_to_extract)
s2_patch = extract_center_patch(s2_full, size=patch_size_to_extract)
dem_patch = extract_center_patch(dem_full, size=patch_size_to_extract)
true_height_patch = extract_center_patch(true_height_full, size=patch_size_to_extract)
s2_rgb_patch = s2_patch[:, :, [0, 1, 2]]

# === 3. Faire la Prédiction ===
print("Prédiction avec le modèle...")
inputs = [s1_patch[np.newaxis, ...], s2_patch[np.newaxis, ...], dem_patch[np.newaxis, ...]]
pred_list = model.predict(inputs)
pred_height_patch_normalized = pred_list[0]

# === 4. Post-traitement et Calcul de l'Erreur ===
pred_map = pred_height_patch_normalized.squeeze() * max_height_denorm
true_map = true_height_patch.squeeze() * max_height_denorm
error_map = np.abs(pred_map - true_map)
vmax_error = np.percentile(error_map, 98)

# === 5. Créer la Figure Comparative à 4 Panneaux ===
print("Création de la visualisation comparative...")
fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
fig.suptitle(f'Analyse Qualitative des Prédictions - {city}', fontsize=16)
vmax_height = max(1, np.max(true_map)) # Assurer que vmax n'est pas 0


# (A) Image Optique
axes[0, 0].imshow(s2_rgb_patch); axes[0, 0].set_title('(A) Image Optique (Sentinel-2)'); axes[0, 0].axis('off')
# (B) Vérité Terrain
im_b = axes[0, 1].imshow(true_map, cmap='viridis', vmin=0, vmax=vmax_height); axes[0, 1].set_title('(B) Label de Hauteur (Généré)'); axes[0, 1].axis('off')
# (C) Prédiction
im_c = axes[1, 0].imshow(pred_map, cmap='viridis', vmin=0, vmax=vmax_height); axes[1, 0].set_title('(C) Prédiction de Hauteur (Modèle)'); axes[1, 0].axis('off')
# (D) Erreur
im_d = axes[1, 1].imshow(error_map, cmap='hot', vmin=0, vmax=vmax_error); axes[1, 1].set_title('(D) Carte des Erreurs Absolues'); axes[1, 1].axis('off')

# Barres de couleur
fig.colorbar(im_b, ax=axes[0, 1], label='Hauteur (m)', fraction=0.046, pad=0.04)
fig.colorbar(im_c, ax=axes[1, 0], label='Hauteur (m)', fraction=0.046, pad=0.04)
fig.colorbar(im_d, ax=axes[1, 1], label='Erreur Absolue (m)', fraction=0.046, pad=0.04)

out_path = OUTPUT_PATH / f"{city}_qualitative_evaluation.png"
plt.savefig(out_path, dpi=300)
plt.show()

print(f"✅ Image d'évaluation sauvegardée dans : {out_path}")