import tensorflow as tf
import tensorflow.keras.backend as K

# === Recommended: Combined MSE + Cosine Similarity ===
def combined_mse_cosine_loss(y_true, y_pred):

    y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    y_pred = tf.where(tf.math.is_nan(y_pred), tf.zeros_like(y_pred), y_pred)

    # MSE
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Cosine similarity
    flat_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    flat_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    dot_product = tf.reduce_sum(flat_true * flat_pred, axis=1)
    norm_true = tf.norm(flat_true, axis=1)
    norm_pred = tf.norm(flat_pred, axis=1)

    cosine_sim = dot_product / (norm_true * norm_pred + 1e-6)
    cosine_loss = tf.reduce_mean(1.0 - cosine_sim)

    # Weighted sum
    return 0.8 * mse + 0.2 * cosine_loss


# === Pure MSE (for reference) ===
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))


# === Optional: Smooth L1 Loss ===
def smooth_l1_loss(y_true, y_pred, delta=1.0):
    diff = tf.abs(y_true - y_pred)
    less = 0.5 * tf.square(diff)
    more = delta * (diff - 0.5 * delta)
    return tf.reduce_mean(tf.where(diff < delta, less, more))


# === Optional: SSIM + MSE + RMSE Combo ===
def SSIM_loss_graph(target, pred):
    target = tf.where(tf.math.is_nan(target), tf.zeros_like(target), target)
    pred = tf.where(tf.math.is_nan(pred), tf.zeros_like(pred), pred)

    ssim_loss_weight = 0.3
    mse_loss_weight = 0.5
    rmse_loss_weight = 0.2
    max_val = 100.0  

    ssim = tf.image.ssim(target, pred, max_val=max_val)
    ssim_loss = tf.reduce_mean(1.0 - ssim)
    mse = tf.reduce_mean(tf.square(target - pred))
    rmse = tf.sqrt(mse)

    return ssim_loss_weight * ssim_loss + mse_loss_weight * mse + rmse_loss_weight * rmse
