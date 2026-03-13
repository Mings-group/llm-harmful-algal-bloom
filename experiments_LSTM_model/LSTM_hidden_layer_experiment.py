#!/usr/bin/env python3
"""
Layer Depth Sensitivity Experiment — Patched for Inverse-Scaled Validation MAE
- Fixed LR = 0.001
- Vary number of stacked LSTM layers only
- Saves models, per-epoch curves, plots, predictions, and a CSV summary
- Computes inverse-scaled validation MAE
- Bar chart: number of layers vs inverse-scaled validation MAE
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import pandas as pd
import json

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "..", "preprocessing_scripts", "preprocessed_bi_uni_proper_split", "w12", "bivariate")
SCALER_PATH = os.path.join(BASE_DIR, "..", "preprocessed_bi_uni_proper_split", "scalers.json")
OUT_DIR = "layers_sensitivity_experiment_proper_split"
os.makedirs(OUT_DIR, exist_ok=True)
MODELS_DIR = os.path.join(OUT_DIR, "models")
CURVES_DIR = os.path.join(OUT_DIR, "curves")
PRED_DIR = os.path.join(OUT_DIR, "predictions")
for d in [MODELS_DIR, CURVES_DIR, PRED_DIR]:
    os.makedirs(d, exist_ok=True)


TARGET_COL = "dsp_toxins"

# ---------------- LOAD DATA ----------------
X_train = np.load(os.path.join(DATA_ROOT, "X_train.npy"))
y_train = np.load(os.path.join(DATA_ROOT, "y_train.npy"))
X_val   = np.load(os.path.join(DATA_ROOT, "X_val.npy"))
y_val   = np.load(os.path.join(DATA_ROOT, "y_val.npy"))
X_test  = np.load(os.path.join(DATA_ROOT, "X_test.npy"))
y_test  = np.load(os.path.join(DATA_ROOT, "y_test.npy"))

if y_train.ndim > 1: y_train = y_train[:, 0]
if y_val.ndim > 1:   y_val = y_val[:, 0]
if y_test.ndim > 1:  y_test = y_test[:, 0]

# ---------------- LOAD SCALER ----------------
with open(SCALER_PATH, "r") as f:
    scalers = json.load(f)

dsp_scaler = scalers[TARGET_COL]


COL_MIN = dsp_scaler["min"]
COL_MAX = dsp_scaler["max"]

def inverse_scale(values):
    values = np.asarray(values, dtype=float)
    mask = values != -1
    out = np.full_like(values, np.nan, dtype=float)
    out[mask] = values[mask] * (COL_MAX - COL_MIN) + COL_MIN
    return out

# ---------------- HYPERPARAMETERS ----------------
NUM_UNITS = 64
L1 = 1e-4
L2 = 1e-4
DELTA = 2.0
BATCH_SIZE = 8
MAX_EPOCHS = 75
PATIENCE = 75
MIN_DELTA = 1e-4
FIXED_LR = 0.001

layer_counts = [1, 2, 3, 4]

# ---------------- LOSS ----------------
def masked_huber(delta):
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
        err = y_true - y_pred
        abs_err = tf.abs(err)
        quadratic = tf.minimum(abs_err, delta)
        linear = abs_err - quadratic
        huber = 0.5 * quadratic**2 + delta * linear
        denom = tf.reduce_sum(mask)
        return tf.reduce_sum(mask * huber) / (denom + 1e-8)
    return loss

# ---------------- MODEL BUILDER ----------------
def build_model(num_layers):
    model = Sequential()
    for i in range(num_layers):
        return_seq = (i < num_layers - 1)
        model.add(LSTM(
            NUM_UNITS,
            return_sequences=return_seq,
            input_shape=(X_train.shape[1], X_train.shape[2]) if i == 0 else None,
            kernel_regularizer=regularizers.l1_l2(l1=L1, l2=L2)
        ))
    model.add(Dense(1))
    model.compile(
        optimizer=RMSprop(learning_rate=FIXED_LR),
        loss=masked_huber(DELTA)
    )
    return model

# ---------------- STABILITY METRICS ----------------
def compute_stability(losses):
    if len(losses) < 2:
        return 0.0, 0.0, 1.0
    slopes = np.diff(losses)
    signs = np.sign(slopes)
    sign_changes = np.sum(signs[1:] != signs[:-1])
    osc_score = float(sign_changes) / max(1, len(slopes))
    slope_variance = float(np.var(slopes))
    tail = losses[-20:] if len(losses) >= 20 else losses
    smoothness = float(losses[-1] / (np.mean(tail) + 1e-12))
    return osc_score, slope_variance, smoothness

# ---------------- RUN EXPERIMENT ----------------
summary_rows = []
inverse_mae_vals = []

for n_layers in layer_counts:
    print(f"\nTraining model with {n_layers} LSTM layer(s)")
    model = build_model(n_layers)

    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=MIN_DELTA,
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0
    )

    train_losses = np.array(history.history.get("loss", []), dtype=float)
    val_losses   = np.array(history.history.get("val_loss", []), dtype=float)

    # Save per-run curves
    np.save(os.path.join(CURVES_DIR, f"train_losses_layers_{n_layers}.npy"), train_losses)
    np.save(os.path.join(CURVES_DIR, f"val_losses_layers_{n_layers}.npy"), val_losses)

    # Save model
    model_path = os.path.join(MODELS_DIR, f"model_layers_{n_layers}.keras")
    model.save(model_path)

    # Stability metrics
    osc, var_slope, smooth = compute_stability(val_losses)

    # Predictions
    y_pred_val = model.predict(X_val).flatten()
    y_pred_train = model.predict(X_train).flatten()
    y_pred_test = model.predict(X_test).flatten()

    df_preds = pd.DataFrame({
        "set": ["train"]*len(y_pred_train) + ["val"]*len(y_pred_val) + ["test"]*len(y_pred_test),
        "true": np.concatenate([y_train, y_val, y_test]).flatten(),
        "pred": np.concatenate([y_pred_train, y_pred_val, y_pred_test]).flatten()
    })
    pred_csv_path = os.path.join(PRED_DIR, f"predictions_layers_{n_layers}.csv")
    df_preds.to_csv(pred_csv_path, index=False)

    # ---------------- INVERSE-SCALED VALIDATION MAE ----------------
    y_true_inv = inverse_scale(y_val)
    y_pred_inv = inverse_scale(y_pred_val)
    valid_mask = ~np.isnan(y_true_inv) & ~np.isnan(y_pred_inv)
    val_mae_inv = np.mean(np.abs(y_true_inv[valid_mask] - y_pred_inv[valid_mask]))
    inverse_mae_vals.append(val_mae_inv)

    # ---------------- SAVE SUMMARY ROW ----------------
    summary_rows.append({
        "n_layers": n_layers,
        "val_mae_inverse": val_mae_inv,
        "epochs_ran": len(val_losses),
        "oscillation": osc,
        "slope_variance": var_slope,
        "smoothness": smooth,
        "model_path": model_path,
        "pred_csv": pred_csv_path
    })



# ---------------- SAVE SUMMARY CSV ----------------
csv_path = os.path.join(OUT_DIR, "layers_experiment_summary.csv")
pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
print(f"\nSaved summary CSV -> {csv_path}")



# ---------------- BAR PLOT ----------------
plt.figure(figsize=(6, 4))
plt.bar([str(l) for l in layer_counts], inverse_mae_vals, width=0.6)
plt.xlabel("Number of Layers", fontsize=14)
plt.ylabel("Validation MAE (µg OA equiv. kg$^{-1}$)", fontsize=14)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "layers_vs_inverse_scaled_MAE.png"), dpi=300)
plt.close()

print(f"\nBar plot saved -> {os.path.join(OUT_DIR, 'layers_vs_inverse_scaled_MAE.png')}")
