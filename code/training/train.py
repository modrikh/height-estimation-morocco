#train.py

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import tensorflow as tf
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from code.config import *
from code.generators import Cust_DatasetGenerator
from code.losses import combined_mse_cosine_loss
from code.metrics import rmse_metric, mae_metric, r2_metric
from code.utils import load_data
from code.models.architectures import build_mbhr_resnet
from code.training.history_plot import plot_training_history

# === Callback Setup ===
checkpoint_path = OUTPUT_PATH / "checkpoints" / "best_model.h5"
checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

callbacks = [
    ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
]

# === 1. Load Data CSV ===
csv_path = CSV_PATH
train_data, val_data = load_data(csv_path, split_ratio=0.2)

# === 2. Data Generators ===
train_gen = Cust_DatasetGenerator(train_data, batch_size=BATCH_SIZE, training=True)
val_gen = Cust_DatasetGenerator(val_data, batch_size=BATCH_SIZE, training=False)

# === 3. Build Model ===
print("Building MBHR-ResNet model...")
model = build_mbhr_resnet(
    input_shape_s1=(IMG_HEIGHT, IMG_WIDTH, s1_ch),
    input_shape_s2=(IMG_HEIGHT, IMG_WIDTH, s2_ch),
    input_shape_dem=(IMG_HEIGHT, IMG_WIDTH, dem_ch)
)


# === 4. Compile Model ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=combined_mse_cosine_loss,
    metrics=[rmse_metric, mae_metric, r2_metric]
)

model.summary()

# === 5. Train Model ===
print("Starting training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

# === Evaluate on validation set
print("\nFinal evaluation on validation set:")
results = model.evaluate(val_gen, verbose=1)
for name, val in zip(model.metrics_names, results):
    print(f"{name}: {val:.4f}")

# === 6. Plot Training History ===
plot_training_history(history)

# === 7. Sanity Check on One Batch ===
sample = train_gen[0]
inputs, labels = sample
print("Label stats:", np.min(labels), np.max(labels), "NaNs:", np.isnan(labels).any())

# === 8. Save Final Model ===
final_model_path = OUTPUT_PATH / "final_model.h5"
model.save(str(final_model_path))
print(f"Final model saved to: {final_model_path}")
