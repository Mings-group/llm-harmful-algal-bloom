#!/usr/bin/env python3
"""
Learning Rate Sensitivity Experiment (Exactly Same Model, Only LR Changes)
------------------------------------------------------------------------
- Uses the original RMSprop + masked_huber setup.
- Only the learning rate changes between runs.
- Includes stability metrics + 2×2 learning curve subplots.
- Computes inverse-scaled validation MAE and plots it vs training time.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

# ============================================================
# Load Data
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "..", "preprocessed_bi_uni_proper_split", "w12", "bivariate")

X_train = np.load(os.path.join(DATA_ROOT, "X_train.npy"))
y_train = np.load(os.path.join(DATA_ROOT, "y_train.npy"))
X_val   = np.load(os.path.join(DATA_ROOT, "X_val.npy"))
y_val   = np.load(os.path.join(DATA_ROOT, "y_val.npy"))
X_test  = np.load(os.path.join(DATA_ROOT, "X_test.npy"))
y_test  = np.load(os.path.join(DATA_ROOT, "y_test.npy"))

# Ensure correct shape
if y_train.ndim > 1:
    y_train = y_train[:, 0]
if y_val.ndim > 1:
    y_val = y_val[:, 0]
if y_test.ndim > 1:
    y_test = y_test[:, 0]
    

OUT_DIR = "lr_sensitivity_single_horizon_proper_split"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Settings
# ============================================================
NUM_LAYERS = 1
NUM_UNITS = 64
L1 = 1e-4
L2 = 1e-4
DROP = 0.2
DELTA = 2.0
BATCH_SIZE = 8
MAX_EPOCHS = 75
PATIENCE = 75 
MIN_DELTA = 1e-4


print("\n================ DATA DIAGNOSTICS ================")
print("Train samples:", len(y_train))
print("Val samples:", len(y_val))
print("Test samples:", len(y_test))

print("Valid targets in VAL:", np.sum(y_val != -1))
print("Invalid (-1) targets in VAL:", np.sum(y_val == -1))
print("All VAL targets are -1?", np.all(y_val == -1))
for i in range(0, len(y_val), BATCH_SIZE):
    batch = y_val[i:i+BATCH_SIZE]
    if len(batch) == BATCH_SIZE and np.all(batch == -1):
        print("⚠ Found ALL-MASKED validation batch starting at index:", i)

print("==================================================\n")


learning_rates = [1e-3, 5e-3, 1e-2, 1e-1]

# ============================================================
# Load scalers & inverse scaling
# ============================================================
scalers_path = os.path.join(BASE_DIR, "..", "preprocessed_bi_uni_proper_split", "scalers.json")
SCALERS = json.load(open(scalers_path)) if os.path.exists(scalers_path) else None

def inverse_scale_array(arr, feature_name="dsp_toxins"):
    if SCALERS is None: 
        return arr
    sc = SCALERS[feature_name]
    minv, maxv = sc["min"], sc["max"]
    a = np.array(arr, dtype=float)
    return np.where(a == -1, -1.0, a * (maxv - minv) + minv)

# ============================================================
# Custom masked_huber
# ============================================================
def masked_huber(delta):
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
        err = y_true - y_pred
        abs_err = tf.abs(err)
        quadratic = tf.minimum(abs_err, delta)
        linear = abs_err - quadratic
        huber = 0.5 * quadratic**2 + delta * linear
        return tf.math.divide_no_nan(
    tf.reduce_sum(mask * huber),
    tf.reduce_sum(mask)
    )

    return loss

# ============================================================
# Model Builder
# ============================================================
def build_model(lr):
    model = Sequential()
    model.add(LSTM(
        NUM_UNITS,
        return_sequences=False,
        input_shape=(X_train.shape[1], X_train.shape[2]),
        kernel_regularizer=regularizers.l1_l2(l1=L1, l2=L2)
    ))
    model.add(Dense(1))
    model.compile(
        optimizer=RMSprop(learning_rate=lr),
        loss=masked_huber(DELTA)
    )
    return model

# ============================================================
# Stability Metrics
# ============================================================
def compute_stability(losses):
    slopes = np.diff(losses)
    signs = np.sign(slopes)
    sign_changes = np.sum(signs[1:] != signs[:-1])
    osc_score = sign_changes / len(slopes) if len(slopes) > 0 else 0
    slope_variance = np.var(slopes) if len(slopes) > 0 else 0
    tail = losses[-20:] if len(losses) >= 20 else losses
    smoothness = losses[-1] / np.mean(tail)
    return osc_score, slope_variance, smoothness

# ============================================================
# Run LR experiment
# ============================================================
results = {}

for lr in learning_rates:
    print("\n=====================================")
    print(f"Training with learning rate = {lr}")
    print("=====================================")

    model = build_model(lr)

    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=MIN_DELTA,
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0
    )
    duration = time.time() - start
    train_losses = np.array(history.history["loss"])
    val_losses = np.array(history.history["val_loss"])

    save_path = os.path.join(OUT_DIR, f"model_lr_{lr}.keras")
    model.save(save_path)
    print(f"Saved model for LR={lr} → {save_path}")

    osc, var_slope, smooth = compute_stability(val_losses)

    # Predictions for inverse-scaled MAE
    y_val_pred = model.predict(X_val, batch_size=len(X_val)).flatten()
    y_test_pred = model.predict(X_test, batch_size=len(X_test)).flatten()

    y_val_inv = inverse_scale_array(y_val).flatten()
    y_val_pred_inv = inverse_scale_array(y_val_pred).flatten()
    mae_val_inv = mean_absolute_error(y_val_inv, y_val_pred_inv)

    y_test_inv = inverse_scale_array(y_test).flatten()
    y_test_pred_inv = inverse_scale_array(y_test_pred).flatten()
    mae_test_inv = mean_absolute_error(y_test_inv, y_test_pred_inv)

    # True Loss Evaluation
    true_train_loss = model.evaluate(X_train, y_train, batch_size=len(X_train), verbose=0)
    true_val_loss   = model.evaluate(X_val, y_val, batch_size=len(X_val), verbose=0)
    true_test_loss  = model.evaluate(X_test, y_test, batch_size=len(X_test), verbose=0)

    results[lr] = {
        "train_losses": train_losses,
        "losses": val_losses,
        "epochs": len(val_losses),
        "time": duration,
        "oscillation": osc,
        "slope_variance": var_slope,
        "smoothness": smooth,
        "true_train_loss": true_train_loss,
        "true_val_loss": true_val_loss,
        "true_test_loss": true_test_loss,
        "MAE_val_inv": mae_val_inv,
        "MAE_test_inv": mae_test_inv,
        "y_val_pred": y_val_pred,
        "y_test_pred": y_test_pred
    }

    # Save predictions CSV
    df_preds = pd.DataFrame({
        "set": ["train"]*len(y_train) + ["val"]*len(y_val) + ["test"]*len(y_test),
        "true": np.concatenate([y_train, y_val, y_test]),
        "pred": np.concatenate([
            model.predict(X_train, batch_size=len(X_train)).flatten(),
            y_val_pred,
            y_test_pred
        ])
    })
    csv_path = os.path.join(OUT_DIR, f"predictions_lr_{lr}.csv")
    df_preds.to_csv(csv_path, index=False)

# ============================================================
# Save inverse-scaled metrics to CSV
# ============================================================
metrics_records = []
for lr in learning_rates:
    metrics_records.append({
        "learning_rate": lr,
        "MAE_val_inv": results[lr]["MAE_val_inv"],
        "MAE_test_inv": results[lr]["MAE_test_inv"],
        "true_train_loss": results[lr]["true_train_loss"],
        "true_val_loss": results[lr]["true_val_loss"],
        "true_test_loss": results[lr]["true_test_loss"],
        "training_time_sec": results[lr]["time"]
    })

metrics_df = pd.DataFrame(metrics_records)
metrics_csv_path = os.path.join(OUT_DIR, "lr_metrics_inverse_scaled.csv")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Saved inverse-scaled LR metrics to {metrics_csv_path}")

# ============================================================
# 2x2 Learning Curve Subplots — PATCH: set custom ylim
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, lr in enumerate(learning_rates):
    ax = axes[i]
    train = results[lr]["train_losses"]
    val   = results[lr]["losses"]
    test  = results[lr]["true_test_loss"]

    ax.plot(train, label="Train Loss", alpha=0.8)
    ax.plot(val,   label="Val Loss", alpha=0.8)
    ax.axhline(test, color="red", linestyle="--", linewidth=1.5, label=f"Test Loss = {test:.4f}")

    ax.set_title(f"LR={lr}", fontsize=18)
    if i < 2:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", labelbottom=False)
    else:
        ax.set_xlabel("Epoch", fontsize=16)
        ax.tick_params(axis="x", labelsize=14)

    ax.set_ylabel("Loss", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(fontsize=14)

    # --- SET CUSTOM YLIM PER AXIS ---
    if i in [0, 1, 2]:       # Top row and bottom left
        ax.set_ylim(0, 0.011)
    elif i == 3:              # Bottom right
        ax.set_ylim(0, 0.3)

for j in range(len(learning_rates), 4):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(os.path.join(OUT_DIR, "best_learning_curves_train_val_2x2.png"), dpi=300)
plt.close()

# ============================================================
# Stability Metrics Plot
# ============================================================
osc_vals = [results[lr]["oscillation"] for lr in learning_rates]
var_vals = [results[lr]["slope_variance"] for lr in learning_rates]
smooth_vals = [results[lr]["smoothness"] for lr in learning_rates]

plt.figure(figsize=(12,6))
plt.plot(learning_rates, osc_vals, marker="o", label="Oscillation Score")
plt.plot(learning_rates, var_vals, marker="s", label="Slope Variance")
plt.plot(learning_rates, smooth_vals, marker="^", label="Smoothness Score")
plt.xscale("log")
plt.title("Stability Metrics vs Learning Rate")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Metric Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "stability_metrics.png"), dpi=300)
plt.close()

# ============================================================
# Original LR-Sensitivity Overlay
# ============================================================
plt.figure(figsize=(14,7))
linestyles = ["-", "--", "-.", ":", (0, (3,1,1,1))]

for idx, lr in enumerate(learning_rates):
    series = results[lr]["losses"]
    smooth = np.convolve(series, np.ones(7)/7, mode="valid")
    plt.plot(smooth, label=f"LR={lr}", linewidth=2.2, linestyle=linestyles[idx % len(linestyles)], alpha=0.9)

plt.title("Learning Rate Sensitivity (Validation Loss)")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lr_sensitivity_single_horizon.png"), dpi=300)
plt.close()

# ============================================================
# Elapsed-time vs Inverse-scaled Validation MAE (dual-axis)
# ============================================================
plt.figure(figsize=(8,5))
ax1 = plt.gca()
ax2 = ax1.twinx()

x = np.arange(len(learning_rates))
val_mae_inv = [results[lr]["MAE_val_inv"] for lr in learning_rates]
elapsed_time = [results[lr]["time"] for lr in learning_rates]

ax1.bar(x - 0.15, val_mae_inv, width=0.3, color="tab:red", label=" Validation MAE (µg AO equiv. kg⁻¹)")
ax2.bar(x + 0.15, elapsed_time, width=0.3, color="tab:blue", label="Training Time (s)")

ax1.set_xlabel("Learning Rate", fontsize=18)
ax1.set_ylabel("Validation MAE (µg AO equiv. kg⁻¹)", color="tab:red", fontsize=18)
ax2.set_ylabel("Training Time (s)", color="tab:blue",fontsize=18)
ax1.tick_params(axis='y', labelcolor="tab:red", labelsize=16)
ax2.tick_params(axis='y', labelcolor="tab:blue", labelsize=16)
ax1.set_xticks(x)
ax1.set_xticklabels([str(lr) for lr in learning_rates], fontsize=16)
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "val_mae_vs_training_time.png"), dpi=300)
plt.close()

print("\nSaved all results and figures to:", OUT_DIR)
