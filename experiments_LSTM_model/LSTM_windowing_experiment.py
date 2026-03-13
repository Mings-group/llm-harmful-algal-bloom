#!/usr/bin/env python3
"""
Faithful LSTM training — bivariate only, multi-window
Automatically loops over preprocessed window sizes (8, 12, 16)
and reduces horizon from 4->1 for t+1 forecasts
Plots learning curves for all windows in a single figure
and a bar chart of validation MAE vs. training time.
"""

import os
import json
import math
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATA_DIR = "../preprocessing_scripts/preprocessed_bi_uni_proper_split/w12"
OUTPUT_DIR = "windowing_experiment_results_proper_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VARIANT = "bivariate"
WINDOW_FOLDERS = ["w8", "w12", "w16"]
WINDOWS = [8,12,16]
HORIZON_FULL = 4
PAPER_EPOCHS = 75
PAPER_BATCH = 8

GRID = [
    {"neurons": [64],  "l1": 1e-4, "l2": 1e-4, "lr": 0.001, "delta": 2},
]

# ---------------- Load scalers ----------------
scalers_path = os.path.join(DATA_DIR, "scalers.json")
if os.path.exists(scalers_path):
    with open(scalers_path) as f:
        SCALERS = json.load(f)
else:
    SCALERS = None

def inverse_scale_array(arr, feature_name="dsp_toxins"):
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
        xp, yp = os.path.join(variant_dir, f"X_{s}.npy"), os.path.join(variant_dir, f"y_{s}.npy")
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
    raise ValueError(f"Unexpected X shape {X.shape}")

def fix_y_shape_single_step(y):
    y = np.array(y)
    if y.ndim == 3 and y.shape[2] == 1:
        y = y[:, 0, 0:1]
    elif y.ndim == 2:
        y = y[:, 0:1]
    elif y.ndim == 1:
        y = y.reshape(-1, 1)
    else:
        raise ValueError(f"Unexpected y shape {y.shape}")
    return y[..., np.newaxis]

# ---------------- Loss & metrics ----------------
def masked_huber(delta=1.0):
    def loss_fn(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
        err = y_true - y_pred
        abs_err = tf.abs(err)
        quad = tf.minimum(abs_err, delta)
        lin = abs_err - quad
        hub = 0.5 * tf.square(quad) + delta * lin
        masked = hub * mask
        denom = tf.reduce_sum(mask)
        return tf.reduce_sum(masked) / (denom + 1e-8)
    return loss_fn

def per_horizon_metrics_masked(y_true_inv, y_pred_inv):
    H = y_true_inv.shape[1]
    rmse_list, mae_list = [], []
    for h in range(H):
        mask = (y_true_inv[:, h] != -1)
        if np.sum(mask) == 0:
            rmse_list.append(np.nan); mae_list.append(np.nan); continue
        yt = y_true_inv[mask, h]; yp = y_pred_inv[mask, h]
        rmse_list.append(math.sqrt(mean_squared_error(yt, yp)))
        mae_list.append(mean_absolute_error(yt, yp))
    return rmse_list, mae_list

# ---------------- Model builder ----------------
def build_model(input_shape, output_steps, neurons, l1=0.0, l2=0.0):
    model = Sequential()
    for i, n in enumerate(neurons):
        return_seq = (i < len(neurons) - 1)
        model.add(LSTM(n, input_shape=input_shape if i==0 else None,
                       return_sequences=return_seq,
                       kernel_regularizer=l1_l2(l1=l1, l2=l2)))
    model.add(Dense(output_steps, activation="linear"))
    return model

# ---------------- Training loop ----------------
records = []
histories = {}  # For grouped subplot

for wf in WINDOW_FOLDERS:
    variant_dir = os.path.join(DATA_DIR, wf, VARIANT)
    print(f"\n=== WINDOW FOLDER: {wf} ===")
    X_splits, y_splits = load_split_files(variant_dir)
    if X_splits is None:
        raise FileNotFoundError(f"No split files found in {variant_dir}")

    X_train, X_val, X_test = [fix_X_shape(X_splits[k]) for k in ["train","val","test"]]
    y_train, y_val, y_test = [fix_y_shape_single_step(y_splits[k]) for k in ["train","val","test"]]

    input_shape = (X_train.shape[1], X_train.shape[2])
    best_avg, best_model, best_params, best_history, best_time = np.inf, None, None, None, None

    for gid, params in enumerate(GRID, 1):
        model = build_model(input_shape, 1, params["neurons"], params["l1"], params["l2"])
        model.compile(optimizer=RMSprop(learning_rate=params["lr"]),
                      loss=masked_huber(params["delta"]))

        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=PAPER_EPOCHS,
            batch_size=PAPER_BATCH,
            verbose=0
        )
        elapsed_time = time.time() - start_time

        y_val_pred = model.predict(X_val)
        y_val_inv = inverse_scale_array(y_val).reshape(-1,1)
        y_val_pred_inv = inverse_scale_array(y_val_pred).reshape(-1,1)
        _, mae_list = per_horizon_metrics_masked(y_val_inv, y_val_pred_inv)
        avg = np.nanmean(mae_list)

        if avg < best_avg:
            best_avg, best_model, best_params, best_history, best_time = avg, model, params, history, elapsed_time
            best_model.save(os.path.join(OUTPUT_DIR, f"{wf}_best_intermediate.keras"))

    histories[wf] = best_history.history
    print(f"Best params for {wf}: {best_params}")
    print(f"Training time for best model: {best_time:.2f}s")

    # Individual learning curve plot
    train_loss = best_history.history["loss"]
    val_loss = best_history.history["val_loss"]
    epochs = range(1, len(train_loss)+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, label="Train", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation", linestyle="--", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Learning Curve ({wf})")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{wf}_learning_curve.png"), dpi=300)
    plt.close()

    # Final predictions
    y_pred_train = best_model.predict(X_train)
    y_pred_val = best_model.predict(X_val)
    y_pred_test = best_model.predict(X_test)

    y_tr_inv = inverse_scale_array(y_train).reshape(-1,1)
    y_val_inv = inverse_scale_array(y_val).reshape(-1,1)
    y_te_inv = inverse_scale_array(y_test).reshape(-1,1)
    y_pred_tr_inv = inverse_scale_array(y_pred_train).reshape(-1,1)
    y_pred_val_inv = inverse_scale_array(y_pred_val).reshape(-1,1)
    y_pred_te_inv = inverse_scale_array(y_pred_test).reshape(-1,1)

    # Save predictions CSV
    df_preds = pd.DataFrame({
        "set": ["train"]*len(y_tr_inv) + ["val"]*len(y_val_inv) + ["test"]*len(y_te_inv),
        "true": np.concatenate([y_tr_inv, y_val_inv, y_te_inv]).flatten(),
        "pred": np.concatenate([y_pred_tr_inv, y_pred_val_inv, y_pred_te_inv]).flatten()
    })
    pred_csv_path = os.path.join(OUTPUT_DIR, f"{wf}_predictions.csv")
    df_preds.to_csv(pred_csv_path, index=False)
    print(f"Saved predictions CSV → {pred_csv_path}")

    # Compute metrics
    rmse_tr, mae_tr = per_horizon_metrics_masked(y_tr_inv, y_pred_tr_inv)
    rmse_val, mae_val = per_horizon_metrics_masked(y_val_inv, y_pred_val_inv)
    rmse_te, mae_te = per_horizon_metrics_masked(y_te_inv, y_pred_te_inv)

    rec = {
        "window_folder": wf,
        "best_params": best_params,
        "training_time_s": best_time,
        "MAE_test_t+1": float(mae_te[0]) if not np.isnan(mae_te[0]) else np.nan,
        "RMSE_test_t+1": float(rmse_te[0]) if not np.isnan(rmse_te[0]) else np.nan,
        "MAE_val_t+1": float(np.nanmean(mae_val))  # record validation MAE for bar plot
    }
    records.append(rec)

    best_model.save(os.path.join(OUTPUT_DIR, f"{wf}_best_model_final.keras"))
    print(f"Saved best model for {wf}.")

# Save metrics CSV
pd.DataFrame(records).to_csv(os.path.join(OUTPUT_DIR,"metrics_summary_bivariate_windows_t+1.csv"), index=False)
print("\nSaved metrics_summary_bivariate_windows_t+1.csv")

# ---------------- Grouped subplot for all windows ----------------
fig, axes = plt.subplots(1, len(WINDOW_FOLDERS), figsize=(18,5), sharey=True)
for i, wf in enumerate(WINDOW_FOLDERS):
    history = histories[wf]
    epochs = range(1, len(history["loss"])+1)
    axes[i].plot(epochs, history["loss"], label="Train", linewidth=2)
    axes[i].plot(epochs, history["val_loss"], label="Validation", linestyle="--", linewidth=2)
    axes[i].set_title(f"{wf} Window")
    axes[i].set_xlabel("Epoch")
    if i == 0:
        axes[i].set_ylabel("Loss")
    axes[i].grid(True)
    axes[i].legend()
plt.suptitle("Learning Curves for Different Window Sizes", fontsize=16)
plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(os.path.join(OUTPUT_DIR,"all_window_learning_curves.png"), dpi=300)
plt.close()

# ---------------- Bar plot: Validation MAE vs Training Time ----------------
val_maes = [rec["MAE_val_t+1"] for rec in records]
train_times = [rec["training_time_s"] for rec in records]

x = np.arange(len(WINDOW_FOLDERS))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8,5))

# Traint time bars
color = 'tab:blue'
ax1.bar(x - width/2, train_times, width, color=color, label='Training Time (s)')
ax1.set_xlabel('Window Size')
ax1.set_ylabel('Training Time (s)', color=color)
ax1.set_xticks(x)
ax1.set_xticklabels(WINDOWS)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

# validation MAE bars (secondary axis)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.bar(x + width/2, val_maes, width, color=color, label='Validation MAE')
ax2.set_ylabel('Validation MAE', color=color)
ax2.tick_params(axis='y', labelcolor=color)



plt.title("Validation MAE and Training Time by Window Size")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "val_mae_vs_training_time.png"), dpi=300)
plt.close()
print("Saved bar plot: val_mae_vs_training_time.png")
