# evaluate.py

import numpy as np
import rasterio
import cv2
from code.metrics import rmse_metric, mae_metric, r2_metric
import tensorflow as tf

def evaluate_model(model, test_files, generator_fn):
    preds = []
    trues = []

    for (label_file, s1_file, s2_file) in test_files:
        (s1, s2), label = generator_fn(label_file, s1_file, s2_file)
        pred = model.predict([np.expand_dims(s1, 0), np.expand_dims(s2, 0)])
        preds.append(pred[0].squeeze())
        trues.append(label.squeeze())

    preds = np.array(preds)
    trues = np.array(trues)

    rmse = tf.keras.metrics.Mean()(rmse_metric(trues, preds)).numpy()
    mae = tf.keras.metrics.Mean()(mae_metric(trues, preds)).numpy()
    r2 = tf.keras.metrics.Mean()(r2_metric(trues, preds)).numpy()

    print(f"Evaluation - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")