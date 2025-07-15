#metrics.py
import tensorflow as tf
import tensorflow.keras.backend as K

# === RMSE (Root Mean Squared Error) ===
# This metric is fine as is.
def rmse_metric(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# === MAE (Mean Absolute Error) ===
# This metric is fine as is.
def mae_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_pred - y_true))

# === Original R² (Calculated on ALL pixels) ===
# Keep this for comparison if you want, but don't use it as your main success metric.
def r2_metric(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    r2 = 1 - ss_res / (ss_tot + K.epsilon())
    return tf.clip_by_value(r2, -1.0, 1.0)

# === NEW & IMPROVED: R² calculated ONLY on building pixels ===
# This is the metric you should use to track model performance.
def r2_metric_buildings_only(y_true, y_pred):
    """
    Calculates the R-squared (R²) metric only for pixels where the
    ground truth (y_true) is greater than zero (i.e., on buildings).
    """
    # 1. Create a boolean mask for pixels that are part of a building.
    # The condition `y_true > 0` identifies these pixels.
    building_mask = tf.greater(y_true, 0)

    # 2. Apply the mask to both the true and predicted values.
    # This filters the tensors, keeping only the values for building pixels.
    y_true_buildings = tf.boolean_mask(y_true, building_mask)
    y_pred_buildings = tf.boolean_mask(y_pred, building_mask)

    # 3. Calculate R² only on the filtered (building) data.
    ss_res = tf.reduce_sum(tf.square(y_true_buildings - y_pred_buildings))
    ss_tot = tf.reduce_sum(tf.square(y_true_buildings - tf.reduce_mean(y_true_buildings)))
    r2 = 1 - ss_res / (ss_tot + K.epsilon())

    # 4. Handle the edge case where a batch has no buildings.
    # If ss_tot is zero (no variance), R² is meaningless, so we return 0.0.
    # This avoids division-by-zero (NaN) errors.
    # We also clip to ensure the value stays within a reasonable range.
    return tf.where(tf.equal(ss_tot, 0), 0.0, tf.clip_by_value(r2, -1.0, 1.0))