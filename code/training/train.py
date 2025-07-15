# code/training/train.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import tensorflow as tf
import numpy as np
import pandas as pd
from keras.saving import save_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from code.config import *
from code.generators import MultiTaskPatchGenerator
from code.losses import combined_mse_cosine_loss # Assurez-vous que ce fichier existe et contient la fonction
from code.metrics import rmse_metric, mae_metric, r2_metric_buildings_only
from code.utils import load_data
from code.models.architectures import build_multitask_mbhr_resnet
from code.training.history_plot import plot_training_history

# === Chemins ===
checkpoint_dir = OUTPUT_PATH / "checkpoints_multitask"
final_model_path = OUTPUT_PATH / "final_model_multitask.keras"
checkpoint_path = checkpoint_dir / "best_model.keras"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# === Callbacks ===
callbacks = [
    ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
]

# === 1. Charger les Données ===
train_files, valid_files = load_data(CSV_PATH, split_ratio=0.2)
print(f"Trouvé {len(train_files)} fichiers d'entraînement et {len(valid_files)} de validation.")

# === 2. Générateurs de Données ===
PATCH_SIZE = 128
train_gen = MultiTaskPatchGenerator(train_files, batch_size=BATCH_SIZE, patch_size=PATCH_SIZE, training=True)
valid_gen = MultiTaskPatchGenerator(valid_files, batch_size=BATCH_SIZE, patch_size=PATCH_SIZE, training=False)

# === 3. Construire le Modèle ===
print("Construction du modèle multi-tâches...")
model = build_multitask_mbhr_resnet(
    input_shape_s1=(PATCH_SIZE, PATCH_SIZE, s1_ch),
    input_shape_s2=(PATCH_SIZE, PATCH_SIZE, s2_ch),
    input_shape_dem=(PATCH_SIZE, PATCH_SIZE, dem_ch),
    num_classes=NUM_CLASSES
)
model.summary(line_length=150)

# === 4. Compiler le Modèle ===
print("Compilation du modèle...")
losses = {
    "output_height": combined_mse_cosine_loss,
    "output_segmentation": "categorical_crossentropy"
}
loss_weights = {
    "output_height": LOSS_WEIGHT_REGRESSION,
    "output_segmentation": LOSS_WEIGHT_CLASSIFICATION
}
metrics = {
    "output_height": [rmse_metric, mae_metric, r2_metric_buildings_only],
    "output_segmentation": ['accuracy', tf.keras.metrics.MeanIoU(num_classes=NUM_CLASSES)]
}
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=losses, loss_weights=loss_weights, metrics=metrics)

# === 5. Entraîner le Modèle ===
print("\nDébut de l'entraînement...")
history = model.fit(train_gen, validation_data=valid_gen, epochs=EPOCHS, callbacks=callbacks, verbose=1)

# === 6. Évaluation et Sauvegarde ===
print("\nÉvaluation finale sur le jeu de validation :")
results = model.evaluate(valid_gen, verbose=1)
print("Métriques finales de validation (depuis l'historique):")
for key, value in history.history.items():
    if key.startswith('val_'):
        print(f"  {key}: {value[-1]:.4f}")

hist_df = pd.DataFrame(history.history)
hist_csv_path = OUTPUT_PATH / "multitask_training_log.csv"
hist_df.to_csv(hist_csv_path, index=False)
print(f"Historique d'entraînement sauvegardé dans {hist_csv_path}")

plot_training_history(history)

print(f"\nSauvegarde du modèle final dans : {final_model_path}")
save_model(model, str(final_model_path))
print("✅ Modèle final sauvegardé.")