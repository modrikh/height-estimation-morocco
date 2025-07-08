#metrics.py

import tensorflow as tf
import tensorflow.keras.backend as K

# === RMSE (Root Mean Squared Error) ===
def rmse_metric(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# === MAE (Mean Absolute Error) ===
def mae_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred - y_true))

# === R² (Coefficient of Determination) ===
def r2_metric(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - (ss_res / (ss_tot + K.epsilon()))
    return r2
