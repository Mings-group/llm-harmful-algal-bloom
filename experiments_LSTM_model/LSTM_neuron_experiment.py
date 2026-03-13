#!/usr/bin/env python3
"""
LSTM Neuron-Count Sensitivity Experiment
----------------------------------------
- Tests different numbers of LSTM units: 16, 32, 64, 128
- 1 hidden LSTM layer only
- Uses masked Huber loss + RMSprop
- Tracks train/val loss, oscillation, variance, smoothness
- Saves predictions AND model weights
- Analyzes effective (active) weights
- Computes inverse-scaled validation MAE
- Plots Neurons vs Inverse-Scaled MAE
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers

# ---------------- CONFIG ----------------
DATA_DIR = "../preprocessing_scripts/preprocessed_bi_uni_proper_split/w12"
SCALER_PATH = "../preprocessing_scripts/preprocessed_bi_uni_proper_split/scalers.json"
OUT_DIR = "neuron_experiment_proper_split"
SITE_ID = "RIAV1"
TARGET_COL = "dsp_toxins"

os.makedirs(OUT_DIR, exist_ok=True)

MODELS_DIR = os.path.join(OUT_DIR, "models")
PRED_DIR = os.path.join(OUT_DIR, "predictions")
WEIGHTS_DIR = os.path.join(OUT_DIR, "weights")

for d in [MODELS_DIR, PRED_DIR, WEIGHTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------- LOAD DATA ----------------
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

if y_train.ndim > 1:
    y_train = y_train[:, 0]
if y_val.ndim > 1:
    y_val = y_val[:, 0]

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

# ---------------- SETTINGS ----------------
DELTA = 2.0
BATCH_SIZE = 8
MAX_EPOCHS = 75
PATIENCE = 75
L1 = 1e-4
L2 = 1e-4
MIN_DELTA = 1e-4
LEARNING_RATE = 0.001
NEURON_COUNTS = [16, 32, 64, 128]
WEIGHT_EPS = 1e-3

# ---------------- LOSS ----------------
def masked_huber(delta):
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
        err = y_true - y_pred
        abs_err = tf.abs(err)
        quadratic = tf.minimum(abs_err, delta)
        linear = abs_err - quadratic
        huber = 0.5 * quadratic**2 + delta * linear
        return tf.reduce_sum(mask * huber) / tf.reduce_sum(mask)
    return loss

# ---------------- MODEL BUILDER ----------------
def build_model(num_units):
    model = Sequential()
    model.add(LSTM(
        num_units,
        return_sequences=False,
        input_shape=(X_train.shape[1], X_train.shape[2]),
        kernel_regularizer=regularizers.l1_l2(l1=L1, l2=L2)
    ))
    model.add(Dense(1))
    model.compile(
        optimizer=RMSprop(learning_rate=LEARNING_RATE),
        loss=masked_huber(DELTA),
    )
    return model

# ---------------- STABILITY METRICS ----------------
def compute_stability(losses):
    slopes = np.diff(losses)
    signs = np.sign(slopes)
    sign_changes = np.sum(signs[1:] != signs[:-1])
    osc_score = sign_changes / len(slopes) if len(slopes) > 0 else 0
    slope_variance = np.var(slopes) if len(slopes) > 0 else 0
    tail = losses[-20:] if len(losses) >= 20 else losses
    smoothness = losses[-1] / np.mean(tail)
    return osc_score, slope_variance, smoothness

# ---------------- RUN EXPERIMENT ----------------
results = []

for units in NEURON_COUNTS:
    print(f"\n=== Training with {units} LSTM units ===")

    model = build_model(units)

    early_stop = tf.keras.callbacks.EarlyStopping(
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

    train_losses = np.array(history.history["loss"])
    val_losses = np.array(history.history["val_loss"])
    osc, var_slope, smooth = compute_stability(val_losses)

    # ---------------- SAVE MODEL ----------------
    model_path = os.path.join(MODELS_DIR, f"model_{units}_units.keras")
    model.save(model_path)

    # ---------------- SAVE WEIGHTS ----------------
    weights = model.get_weights()
    flat_weights = np.concatenate([w.flatten() for w in weights])

    weights_path = os.path.join(WEIGHTS_DIR, f"weights_{units}_units.npy")
    np.save(weights_path, flat_weights)

    total_weights = flat_weights.size
    active_weights = int(np.sum(np.abs(flat_weights) > WEIGHT_EPS))
    fraction_active = active_weights / total_weights

    # ---------------- SAVE PREDICTIONS ----------------
    y_pred_val = model.predict(X_val).flatten()

    pred_df = pd.DataFrame({
        "true": y_val,
        "pred": y_pred_val
    })

    pred_csv = os.path.join(PRED_DIR, f"predictions_{units}_units.csv")
    pred_df.to_csv(pred_csv, index=False)

    # ---------------- INVERSE-SCALED MAE ----------------
    y_true_inv = inverse_scale(y_val)
    y_pred_inv = inverse_scale(y_pred_val)

    valid = ~np.isnan(y_true_inv) & ~np.isnan(y_pred_inv)
    final_val_mae = np.mean(np.abs(y_true_inv[valid] - y_pred_inv[valid]))

    # ---------------- RECORD RESULTS ----------------
    results.append({
        "units": units,
        "val_MAE_inverse": final_val_mae,
        "oscillation": osc,
        "slope_variance": var_slope,
        "smoothness": smooth,
        "total_weights": total_weights,
        "active_weights": active_weights,
        "fraction_active": fraction_active
    })

# ---------------- SAVE SUMMARY ----------------
df = pd.DataFrame(results)
summary_csv = os.path.join(OUT_DIR, "neuron_experiment_summary.csv")
df.to_csv(summary_csv, index=False)

# ---------------- BAR PLOT ----------------
plt.figure(figsize=(6, 4))
plt.bar(df["units"].astype(str), df["val_MAE_inverse"], width=0.6)
plt.xlabel("Number of Neurons", fontsize=14)
plt.ylabel("Validation MAE (µg AO equiv. kg$^{-1}$)", fontsize=14)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "neurons_vs_inverse_scaled_MAE.png"),
    dpi=300
)
plt.close()

print(f"\nExperiment complete. Results saved to: {OUT_DIR}")
