#!/usr/bin/env python3
"""
Faithful LSTM training — MSE-optimized variant
----------------------------------------------
Identical to the original faithful script EXCEPT:
- Training loss = masked MSE (instead of masked MAE)

All evaluation, grid, and reporting logic is unchanged.
"""

import os
import json
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------- CONFIG ----------------
DATA_DIR = "../preprocessing_scripts/preprocessed_bi_uni_proper_split/w12"
OUTPUT_DIR = "results_lstm_faithful_mse_proper_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VARIANTS = ["univariate", "bivariate","multivariate"]
WINDOW_SIZE = 12
HORIZON = 4
PAPER_EPOCHS = 75
PAPER_BATCH = 8

GRID = [
    {"neurons": [4],        "l1": 0,     "l2": 0,     "lr": 0.001, "delta": 0.5},
    {"neurons": [64],       "l1": 1e-4,  "l2": 1e-4,  "lr": 0.001, "delta": 2},
    {"neurons": [32, 16],   "l1": 1e-4,  "l2": 1e-3,  "lr": 0.01,  "delta": 0.5},
]

# ---------------- Load scalers ----------------
scalers_path = os.path.join(DATA_DIR, '..',"scalers.json")
if os.path.exists(scalers_path):
    with open(scalers_path) as f:
        SCALERS = json.load(f)
else:
    SCALERS = None

def inverse_scale_array(arr, feature_name="dsp_toxins"):
    """Inverse min–max scaling, preserving -1 placeholders."""
    if SCALERS is None:
        return arr
    sc = SCALERS[feature_name]
    minv, maxv = sc["min"], sc["max"]
    a = np.array(arr, dtype=float)
    return np.where(a == -1, -1.0, a * (maxv - minv) + minv)

# ---------------- Shape helpers ----------------
def load_split_files(variant_dir):
    X_splits, y_splits = {}, {}
    for s in ["train", "val", "test"]:
        xp = os.path.join(variant_dir, f"X_{s}.npy")
        yp = os.path.join(variant_dir, f"y_{s}.npy")
        if os.path.exists(xp) and os.path.exists(yp):
            X_splits[s] = np.load(xp)
            y_splits[s] = np.load(yp)
        else:
            return None, None
    return X_splits, y_splits

def fix_X_shape(X):
    if X.ndim == 3:
        return X
    if X.ndim == 4 and X.shape[-1] == 1:
        return np.squeeze(X, axis=-1)
    if X.ndim == 2 and X.shape[1] % WINDOW_SIZE == 0:
        n_feat = X.shape[1] // WINDOW_SIZE
        return X.reshape(X.shape[0], WINDOW_SIZE, n_feat)
    raise ValueError(f"Bad X shape {X.shape}")

def fix_y_shape(y):
    y = np.array(y)
    if y.ndim == 3 and y.shape[2] == 1:
        return y
    if y.ndim == 2:
        return y[..., np.newaxis]
    if y.ndim == 1:
        return y.reshape(-1, 1, 1)
    raise ValueError(f"Unexpected y shape {y.shape}")

# ---------------- Loss ----------------
def masked_mse():
    """Masked MSE loss ignoring -1 placeholders."""
    def loss_fn(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
        err = tf.square(y_true - y_pred)
        masked = err * mask
        denom = tf.reduce_sum(mask)
        return tf.reduce_sum(masked) / (denom + 1e-8)
    return loss_fn

# ---------------- Metrics ----------------
def per_horizon_metrics_masked(y_true_inv, y_pred_inv):
    H = y_true_inv.shape[1]
    rmse_list, mae_list = [], []
    for h in range(H):
        mask = (y_true_inv[:, h] != -1)
        if np.sum(mask) == 0:
            rmse_list.append(np.nan)
            mae_list.append(np.nan)
            continue
        yt = y_true_inv[mask, h]
        yp = y_pred_inv[mask, h]
        rmse_list.append(math.sqrt(mean_squared_error(yt, yp)))
        mae_list.append(mean_absolute_error(yt, yp))
    return rmse_list, mae_list

# ---------------- Model builder ----------------
def build_model(input_shape, output_steps, neurons, l1=0.0, l2=0.0):
    model = Sequential()
    for i, n in enumerate(neurons):
        model.add(LSTM(
            n,
            input_shape=input_shape if i == 0 else None,
            return_sequences=(i < len(neurons) - 1),
            kernel_regularizer=l1_l2(l1=l1, l2=l2)
        ))
    model.add(Dense(output_steps, activation="linear"))
    return model

# ---------------- Training Loop ----------------
records = []

for vid, variant in enumerate(VARIANTS):
    print(f"\n=== VARIANT: {variant} ===")
    variant_dir = os.path.join(DATA_DIR, variant)
    X_splits, y_splits = load_split_files(variant_dir)
    if X_splits is None:
        raise FileNotFoundError(f"No split files found for {variant}")

    X_train, X_val, X_test = [fix_X_shape(X_splits[k]) for k in ["train", "val", "test"]]
    y_train, y_val, y_test = [fix_y_shape(y_splits[k]) for k in ["train", "val", "test"]]

    input_shape = (WINDOW_SIZE, X_train.shape[2])
    params = GRID[vid]  # Use the corresponding grid entry
    print(f"\nUsing grid config for {variant}: {params}")

    model = build_model(input_shape, HORIZON,
                        params["neurons"], params["l1"], params["l2"])
    model.compile(
        optimizer=RMSprop(learning_rate=params["lr"]),
        loss=masked_mse()
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=PAPER_EPOCHS,
        batch_size=PAPER_BATCH,
        verbose=0
    )

    # Evaluate on validation
    y_val_pred = model.predict(X_val)
    y_val_inv = inverse_scale_array(y_val)
    y_val_pred_inv = inverse_scale_array(y_val_pred)
    rmse_list, _ = per_horizon_metrics_masked(
        y_val_inv.squeeze(), y_val_pred_inv.squeeze()
    )
    avg = np.nanmean(rmse_list)
    print(f"Val RMSE per horizon: {rmse_list} | avg={avg:.3f}")

    best_model = model
    best_params = params
    best_model.save(os.path.join(
        OUTPUT_DIR, f"{variant}_best_model_final.keras"
    ))

    # Test evaluation
    y_pred_test = best_model.predict(X_test)
    y_te_inv = inverse_scale_array(y_test).squeeze()
    y_pred_te_inv = inverse_scale_array(y_pred_test).squeeze()
    rmse_te, mae_te = per_horizon_metrics_masked(y_te_inv, y_pred_te_inv)

    rec = {"variant": variant, "best_params": best_params}
    for h in range(HORIZON):
        rec[f"MAE_test_t+{h+1}"] = float(mae_te[h])
        rec[f"RMSE_test_t+{h+1}"] = float(rmse_te[h])
    records.append(rec)

# Save CSV
pd.DataFrame(records).to_csv(
    os.path.join(OUTPUT_DIR, "metrics_summary_faithful_mse.csv"),
    index=False
)
print("\nSaved metrics_summary_faithful_mse.csv")

