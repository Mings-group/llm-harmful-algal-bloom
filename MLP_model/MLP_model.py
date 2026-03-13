"""
Paper-faithful MLP model training for DSP toxins (Cruz et al. 2022)
-------------------------------------------------------------------
- Uses 12-week input window to predict 4-week horizon (t+1..t+4)
- Uses Adam optimizer and masked Huber loss
- One model per variant with fixed architecture and hyperparameters:
    Univariate   → 1 layer [256], L1=1e-3, L2=0,      lr=0.01,  δ=0.05
    Bivariate    → 1 layer [32],  L1=0,    L2=1e-4,   lr=0.001, δ=2
    Multivariate → 2 layers [16,8], L1=1e-3, L2=1e-5, lr=0.01,  δ=2
"""

import os
import json
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import regularizers
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------- CONFIG ----------------
DATA_DIR = "../preprocessed"
OUTPUT_DIR = "results_mlp_windowed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VARIANTS = ["univariate", "bivariate", "multivariate"]
WINDOW_SIZE = 12
HORIZON = 4
PAPER_EPOCHS = 75
PAPER_BATCH = 8
PATIENCE = 30

# ---------------- Fixed Configs per Variant ----------------
CONFIGS = {
    "univariate":  {"layers": [256],     "l1": 1e-3, "l2": 0.0,   "lr": 0.01,  "delta": 0.05},
    "bivariate":   {"layers": [32],      "l1": 0.0,  "l2": 1e-4,  "lr": 0.001, "delta": 2.0},
    "multivariate":{"layers": [16, 8],   "l1": 1e-3, "l2": 1e-5,  "lr": 0.01,  "delta": 2.0},
}

# ---------------- Load scalers ----------------
with open(os.path.join(DATA_DIR, "scalers.json")) as f:
    SCALERS = json.load(f)

def inverse_scale_array(arr, feature_name="dsp_toxins"):
    sc = SCALERS[feature_name]
    minv, maxv = sc["min"], sc["max"]
    arr = np.array(arr, dtype=float)
    return np.where(arr == -1, -1.0, arr * (maxv - minv) + minv)

# ---------------- Masked Huber ----------------
def masked_huber(delta=1.0):
    def loss_fn(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
        error = y_true - y_pred
        abs_error = tf.abs(error)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        huber = 0.5 * tf.square(quadratic) + delta * linear
        masked = huber * mask
        denom = tf.reduce_sum(mask)
        return tf.reduce_sum(masked) / (denom + 1e-8)
    return loss_fn

# ---------------- Metrics ----------------
def per_horizon_metrics_masked(y_true_inv, y_pred_inv):
    """Compute per-horizon RMSE/MAE ignoring -1 values."""
    # Ensure both are 2D: (samples, horizons)
    if y_true_inv.ndim == 3:
        y_true_inv = y_true_inv.squeeze(-1)
    if y_pred_inv.ndim == 3:
        y_pred_inv = y_pred_inv.squeeze(-1)

    H = y_true_inv.shape[1]
    rmse_list, mae_list = [], []

    for h in range(H):
        mask = y_true_inv[:, h] != -1
        if np.sum(mask) == 0:
            rmse_list.append(np.nan)
            mae_list.append(np.nan)
            continue
        yt, yp = y_true_inv[mask, h], y_pred_inv[mask, h]
        rmse_list.append(math.sqrt(mean_squared_error(yt, yp)))
        mae_list.append(mean_absolute_error(yt, yp))
    return rmse_list, mae_list


# ---------------- Build Model ----------------
def build_mlp(input_shape, output_steps, cfg):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    for n in cfg["layers"]:
        model.add(Dense(
            n, activation="relu",
            kernel_regularizer=regularizers.L1L2(l1=cfg["l1"], l2=cfg["l2"])
        ))
    model.add(Dense(output_steps, activation="linear"))
    return model

# ---------------- Main Loop ----------------
metrics_records = []

for variant in VARIANTS:
    print(f"\n=== TRAINING {variant.upper()} MLP ===")
    cfg = CONFIGS[variant]

    variant_path = os.path.join(DATA_DIR, variant)
    X_train = np.load(os.path.join(variant_path, "X_train.npy"))
    y_train = np.load(os.path.join(variant_path, "y_train.npy"))
    X_val   = np.load(os.path.join(variant_path, "X_val.npy"))
    y_val   = np.load(os.path.join(variant_path, "y_val.npy"))
    X_test  = np.load(os.path.join(variant_path, "X_test.npy"))
    y_test  = np.load(os.path.join(variant_path, "y_test.npy"))

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_mlp(input_shape, HORIZON, cfg)
    model.compile(
        optimizer=Adam(learning_rate=cfg["lr"]),
        loss=masked_huber(cfg["delta"])
    )

    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=PAPER_EPOCHS,
        batch_size=PAPER_BATCH,
        callbacks=[callback],
        verbose=0
    )

    # Predict all splits
    preds = {}
    for split, (X, y) in zip(
        ["train", "val", "test"], [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    ):
        y_pred = model.predict(X, verbose=0)
        y_true_inv = inverse_scale_array(y)
        y_pred_inv = inverse_scale_array(y_pred)
        rmse_list, mae_list = per_horizon_metrics_masked(y_true_inv, y_pred_inv)

        rec = {"variant": variant, "split": split}
        for h in range(HORIZON):
            rec[f"MAE_t+{h+1}"] = mae_list[h]
            rec[f"RMSE_t+{h+1}"] = rmse_list[h]
        metrics_records.append(rec)

    model.save(os.path.join(OUTPUT_DIR, f"{variant}_mlp_final.keras"))
    print(f"✅ Saved model for {variant}")

# Save metrics
metrics_df = pd.DataFrame(metrics_records)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "mlp_metrics_summary.csv"), index=False)
print("\n✅ All MLP training complete.")
print("📊 Results saved in:", OUTPUT_DIR)
