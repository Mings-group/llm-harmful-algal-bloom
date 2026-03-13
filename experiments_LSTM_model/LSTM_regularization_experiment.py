#!/usr/bin/env python3
"""
Loss Function × Regularization Experiment
(CORRECT inverse-scaled metrics, average val MAE epochs 50-75, RMSE)
"""

import os, numpy as np, matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import regularizers
import tensorflow as tf
import pandas as pd
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib.lines import Line2D

# ---------------- CONFIG ----------------
DATA_DIR = "../preprocessing_scripts/preprocessed_bi_uni_proper_split/w12"
BASE_OUT = "loss_vs_regularization_experiment_proper_split"
os.makedirs(BASE_OUT, exist_ok=True)

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))

if y_train.ndim > 1:
    y_train = y_train[:, 0]
if y_val.ndim > 1:
    y_val = y_val[:, 0]

# ---------------- LOAD SCALING METADATA ----------------
scalers_path = os.path.join("../preprocessed_bi_uni_split_E", "scalers.json")
if os.path.exists(scalers_path):
    with open(scalers_path) as f:
        SCALERS = json.load(f)
else:
    SCALERS = None

def inverse_scale_array(arr, feature_name="dsp_toxins"):
    """Inverse scale to original units."""
    if SCALERS is None:
        return arr
    sc = SCALERS[feature_name]
    minv, maxv = sc["min"], sc["max"]
    a = np.array(arr, dtype=float)
    return np.where(a == -1, -1.0, a * (maxv - minv) + minv)

# ---------------- SETTINGS ----------------
NUM_UNITS = 64
DELTA = 2.0
BATCH_SIZE = 8
MAX_EPOCHS = 75
PATIENCE = 75
MIN_DELTA = 1e-4
LEARNING_RATE = 0.001
REG_GRID = [0.0, 1e-4, 1e-3, 1e-2]

# ---------------- MASKED LOSSES ----------------
def masked_mae():
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
        num = tf.reduce_sum(tf.abs(y_true - y_pred) * mask)
        denom = tf.reduce_sum(mask)
        return tf.math.divide_no_nan(num, denom)
    return loss

def masked_mse():
    """Masked MSE loss ignoring -1 placeholders."""
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
        err = tf.square(y_true - y_pred)                     # ✅ squared error
        num = tf.reduce_sum(err * mask)
        denom = tf.reduce_sum(mask)
        return tf.math.divide_no_nan(num, denom)
    return loss


def masked_huber(delta):
    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
        err = y_true - y_pred
        abs_err = tf.abs(err)
        quad = tf.minimum(abs_err, delta)
        lin = abs_err - quad
        hub = 0.5 * quad**2 + delta * lin

        num = tf.reduce_sum(mask * hub)
        denom = tf.reduce_sum(mask)
        return tf.math.divide_no_nan(num, denom)
    return loss


LOSS_MAP = {
    "MAE": masked_mae(),
    "MSE": masked_mse(),
    "Huber": masked_huber(DELTA),
}

# ---------------- MODEL BUILDER ----------------
def build_model(l1, l2, loss_fn):
    model = Sequential([
        LSTM(NUM_UNITS,
             input_shape=(X_train.shape[1], X_train.shape[2]),
             kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)),
        Dense(1)
    ])
    model.compile(
        optimizer=RMSprop(learning_rate=LEARNING_RATE),
        loss=loss_fn
    )
    return model

# ---------------- CUSTOM CALLBACK FOR VAL MAE ----------------
class ValMAECallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_val_pred = self.model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
        y_val_pred_inv = inverse_scale_array(y_val_pred).reshape(-1,1)
        y_val_inv = inverse_scale_array(y_val).reshape(-1,1)
        mask = (y_val_inv[:,0] != -1)
        mae = mean_absolute_error(y_val_inv[mask], y_val_pred_inv[mask])
        if not hasattr(self, "val_maes"):
            self.val_maes = []
        self.val_maes.append(mae)

# ---------------- EXPERIMENT RUNNER ----------------
def run_experiment(exp_name, reg_list, loss_name):
    out_dir = os.path.join(BASE_OUT, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    results = []

    for l1, l2 in reg_list:
        print(f"{exp_name} | {loss_name} | L1={l1}, L2={l2}")

        model = build_model(l1, l2, LOSS_MAP[loss_name])

        val_mae_cb = ValMAECallback()

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[val_mae_cb,
                       tf.keras.callbacks.EarlyStopping(
                           monitor="val_loss",
                           patience=PATIENCE,
                           restore_best_weights=True)],
            verbose=0
        )

        # ---------- AVERAGE VAL MAE OVER EPOCHS 50-75 ----------
        start_epoch = 49  # zero-indexed
        end_epoch = min(75, len(val_mae_cb.val_maes))
        avg_val_mae = np.mean(val_mae_cb.val_maes[start_epoch:end_epoch])

        # ---------- FINAL PREDICTIONS ----------
        y_val_pred = model.predict(X_val, batch_size=BATCH_SIZE)
        y_val_inv      = inverse_scale_array(y_val).reshape(-1,1)
        y_val_pred_inv = inverse_scale_array(y_val_pred).reshape(-1,1)
        mask = (y_val_inv[:,0] != -1)

        # RMSE
        rmse = np.sqrt(mean_squared_error(y_val_inv[mask], y_val_pred_inv[mask]))

        results.append({
            "l1": l1,
            "l2": l2,
            "val_mae_avg_original": avg_val_mae,
            "val_rmse_avg_original": rmse,
            "loss_name": loss_name
        })

    return results

# ---------------- RUN ALL EXPERIMENTS ----------------
all_results = []
all_results += run_experiment("MAE_L1_only", [(l1, 0.0) for l1 in REG_GRID], "MAE")
all_results += run_experiment("MSE_L2_only", [(0.0, l2) for l2 in REG_GRID], "MSE")
all_results += run_experiment("Huber_ElasticNet", [(v, v) for v in REG_GRID], "Huber")

df = pd.DataFrame(all_results)
df.to_csv(os.path.join(BASE_OUT, "all_regularization_metrics_rescaled.csv"), index=False)

# ---------------- PLOTS ----------------
# Only show non-zero L1 or L2 in x-axis labels
labels = [
    "None" if r['l1']==0 and r['l2']==0 else
    f"{'L1='+str(r['l1']) if r['l1'] != 0 else ''}{' ' if r['l1']!=0 and r['l2']!=0 else ''}{'L2='+str(r['l2']) if r['l2'] != 0 else ''}"
    for r in all_results
]


color_map = {"MAE": "blue", "MSE": "green", "Huber": "orange"}
legend_label_map = {
    "MAE": "L1",
    "MSE": "L2",
    "Huber": "Elastic Net"
}

colors = [color_map[r["loss_name"]] for r in all_results]

# ---- MAE ----
plt.figure(figsize=(12, 6))
plt.scatter(range(len(all_results)), df["val_mae_avg_original"], c=colors, s=80)
plt.xticks(range(len(labels)), labels, rotation=45, fontsize=14)
plt.yticks(fontsize=16)
plt.ylabel("Validation MAE (µg OA equiv. kg$^{-1}$)", fontsize=16)
plt.grid(True)

# Add legend
legend_elements = [
    Line2D(
        [0], [0],
        marker='o',
        color='w',
        label=legend_label_map[k],
        markerfacecolor=v,
        markersize=10
    )
    for k, v in color_map.items()
]
leg1 = plt.legend(handles=legend_elements, title="Regularization",fontsize=16)
leg1.get_title().set_fontsize(18)

plt.tight_layout()
plt.savefig(os.path.join(BASE_OUT, "mae_inverse_scaled.png"), dpi=300)
plt.close()

# ---- RMSE ----
plt.figure(figsize=(12, 6))
plt.scatter(range(len(all_results)), df["val_rmse_avg_original"], c=colors, s=80)
plt.xticks(range(len(labels)), labels, rotation=45, fontsize=14)
plt.yticks(fontsize=16)
plt.ylabel("Validation RMSE (µg OA equiv. kg$^{-1}$)", fontsize=16)
plt.grid(True)

# Add legend
leg2 = plt.legend(handles=legend_elements, title="Regularization", fontsize=16)

leg2.get_title().set_fontsize(18)

plt.tight_layout()
plt.savefig(os.path.join(BASE_OUT, "rmse_inverse_scaled.png"), dpi=300)
plt.close()

print("✔ Metrics are now in ORIGINAL UNITS, averaged over epochs 50-75, and plotted (MAE & RMSE).")
