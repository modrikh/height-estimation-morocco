#metrics.py
import tensorflow as tf
import tensorflow.keras.backend as K

# === RMSE (Root Mean Squared Error) ===
def rmse_metric(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# === MAE (Mean Absolute Error) ===
def mae_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred - y_true))

# === Original R² (Calculated on ALL pixels) ===
def r2_metric(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - ss_res / (ss_tot + K.epsilon())
    return tf.clip_by_value(r2, -1.0, 1.0)

# === IMPROVED: R² calculated ONLY on building pixels ===
def r2_metric_buildings_only(y_true, y_pred):
    """
    Calculates the R-squared (R²) metric only for pixels where the
    ground truth (y_true) is greater than zero (i.e., on buildings).
    """
   
    building_mask = tf.greater(y_true, 0)

    y_true_buildings = tf.boolean_mask(y_true, building_mask)
    y_pred_buildings = tf.boolean_mask(y_pred, building_mask)

    ss_res = tf.reduce_sum(tf.square(y_true_buildings - y_pred_buildings))
    ss_tot = tf.reduce_sum(tf.square(y_true_buildings - tf.reduce_mean(y_true_buildings)))
    r2 = 1 - ss_res / (ss_tot + K.epsilon())

    return tf.where(tf.equal(ss_tot, 0), 0.0, tf.clip_by_value(r2, -1.0, 1.0))