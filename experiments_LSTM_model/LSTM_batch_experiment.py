#!/usr/bin/env python3
"""
Faithful LSTM — Batch Size Experiment (bivariate, t+1)
"""

import os, json, math, numpy as np, pandas as pd, tensorflow as tf, time
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_absolute_error, mean_squared_error



# ---------------- CONFIG ----------------
DATA_DIR = "../preprocessing_scripts/preprocessed_bi_uni_proper_split/w12"
OUTPUT_DIR = "results_lstm_faithful_batchsize_experiment_proper_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VARIANT = "bivariate"
WINDOW_SIZE = 12
HORIZON = 1
EPOCHS = 75
BATCH_SIZES = [4, 8, 16, 32]

# Model hyperparameters (bivariate)
MODEL_PARAMS = {"neurons": [64], "l1": 1e-4, "l2": 1e-4, "lr": 0.001, "delta": 2.0}

# ---------------- Load scalers ----------------
scalers_path = os.path.join(DATA_DIR, '..', "scalers.json")
SCALERS = json.load(open(scalers_path)) if os.path.exists(scalers_path) else None

def inverse_scale_array(arr, feature_name="dsp_toxins"):
    if SCALERS is None: return arr
    sc = SCALERS[feature_name]
    minv, maxv = sc["min"], sc["max"]
    a = np.array(arr, dtype=float)
    return np.where(a == -1, -1.0, a * (maxv - minv) + minv)

# ---------------- Shape helpers ----------------
def load_split_files(variant_dir):
    X_splits, y_splits = {}, {}
    for s in ["train","val","test"]:
        xp, yp = os.path.join(variant_dir,f"X_{s}.npy"), os.path.join(variant_dir,f"y_{s}.npy")
        X_splits[s] = np.load(xp)
        y_splits[s] = np.load(yp)
    return X_splits, y_splits

def fix_X_shape(X):
    if X.ndim == 3: return X
    if X.ndim == 4 and X.shape[-1] == 1: return np.squeeze(X, axis=-1)
    raise ValueError(f"Unexpected X shape {X.shape}")

def fix_y_shape_single_step(y):
    y = np.array(y)
    if y.ndim == 3 and y.shape[2] == 1: y = y[:,0,0:1]
    elif y.ndim == 2: y = y[:,0:1]
    elif y.ndim == 1: y = y.reshape(-1,1)
    else: raise ValueError(f"Unexpected y shape {y.shape}")
    return y[..., np.newaxis]

# ---------------- Loss ----------------
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

# ---------------- Model builder ----------------
def build_model(input_shape, output_steps, neurons, l1=0.0, l2=0.0):
    model = Sequential()
    for i,n in enumerate(neurons):
        return_seq = (i < len(neurons) - 1)
        model.add(LSTM(n, input_shape=input_shape if i==0 else None,
                       return_sequences=return_seq,
                       kernel_regularizer=l1_l2(l1=l1,l2=l2)))
    model.add(Dense(output_steps, activation="linear"))
    return model

# ---------------- Load data ----------------
variant_dir = os.path.join(DATA_DIR, VARIANT)
X_splits, y_splits = load_split_files(variant_dir)
X_train, X_val, X_test = [fix_X_shape(X_splits[k]) for k in ["train","val","test"]]
y_train, y_val, y_test = [fix_y_shape_single_step(y_splits[k]) for k in ["train","val","test"]]
input_shape = (WINDOW_SIZE, X_train.shape[2])

# ---------------- Training loop ----------------
records = []
learning_curves = {}  # store losses for plotting

for batch_size in BATCH_SIZES:
    print(f"\n=== Training with batch size {batch_size} ===")
    model = build_model(input_shape, HORIZON, MODEL_PARAMS["neurons"], MODEL_PARAMS["l1"], MODEL_PARAMS["l2"])
    model.compile(optimizer=RMSprop(MODEL_PARAMS["lr"]), loss=masked_huber(MODEL_PARAMS["delta"]))

    start_time = time.perf_counter()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=batch_size,
        verbose=0
    )
    end_time = time.perf_counter()
    training_time = end_time - start_time
    print(f"Training time for batch {batch_size}: {training_time:.2f} seconds")

    # Store learning curves
    learning_curves[batch_size] = {
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"]
    }

    # Save model
    model_name = f"{VARIANT}_batch{batch_size}.keras"
    model.save(os.path.join(OUTPUT_DIR, model_name))

    # ---------------- Predictions & CSV ----------------
    # Train predictions
    y_pred_train = model.predict(X_train)
    y_train_inv = inverse_scale_array(y_train).squeeze()
    y_pred_train_inv = inverse_scale_array(y_pred_train).squeeze()
    train_df = pd.DataFrame({
        "y_true": y_train_inv,
        "y_pred": y_pred_train_inv,
        "abs_error": np.abs(y_train_inv - y_pred_train_inv)
    })
    train_csv_path = os.path.join(OUTPUT_DIR, f"train_predictions_batch{batch_size}.csv")
    train_df.to_csv(train_csv_path, index=False)
    print(f"Saved train predictions to {train_csv_path}")

    # Validation predictions
    y_pred_val = model.predict(X_val)
    y_val_inv = inverse_scale_array(y_val).squeeze()
    y_pred_val_inv = inverse_scale_array(y_pred_val).squeeze()
    val_df = pd.DataFrame({
        "y_true": y_val_inv,
        "y_pred": y_pred_val_inv,
        "abs_error": np.abs(y_val_inv - y_pred_val_inv)
    })
    val_csv_path = os.path.join(OUTPUT_DIR, f"val_predictions_batch{batch_size}.csv")
    val_df.to_csv(val_csv_path, index=False)
    print(f"Saved validation predictions to {val_csv_path}")
    mae_val = mean_absolute_error(y_val_inv, y_pred_val_inv)

    # Test predictions
    y_pred_test = model.predict(X_test)
    y_test_inv = inverse_scale_array(y_test).squeeze()
    y_pred_inv = inverse_scale_array(y_pred_test).squeeze()
    test_df = pd.DataFrame({
        "y_true": y_test_inv,
        "y_pred": y_pred_inv,
        "abs_error": np.abs(y_test_inv - y_pred_inv)
    })
    test_csv_path = os.path.join(OUTPUT_DIR, f"test_predictions_batch{batch_size}.csv")
    test_df.to_csv(test_csv_path, index=False)
    print(f"Saved test predictions to {test_csv_path}")
    rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    print(f"Batch {batch_size} → Test MAE={mae:.3f}, RMSE={rmse:.3f}")

    records.append({
        "batch_size": batch_size,
        "MAE_val": mae_val,
        "MAE_test": mae,
        "RMSE_test": rmse,
        "training_time_sec": training_time
    })

# ---------------- Overlay Learning Curve Plot ----------------
# ---------------- Subplots: Learning Curves ----------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, batch_size in zip(axes, BATCH_SIZES):
    train_loss = learning_curves[batch_size]["train_loss"]
    val_loss = learning_curves[batch_size]["val_loss"]
    epochs_range = range(1, len(train_loss)+1)

    ax.plot(epochs_range, train_loss, label="Train Loss", linestyle='-', linewidth=2)
    ax.plot(epochs_range, val_loss, label="Val Loss", linestyle='--', linewidth=2)
    ax.set_title(f"Batch Size = {batch_size}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

plt.suptitle("Learning Curves Across Batch Sizes", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
subplot_path = os.path.join(OUTPUT_DIR, "learning_curves_subplots_batchsize.png")
plt.savefig(subplot_path, dpi=300)
plt.close()

print(f"Saved learning_curves_subplots_batchsize.png in {OUTPUT_DIR}")



#Plotting figure with dual axis
df = pd.DataFrame(records)

fig, ax1 = plt.subplots(figsize=(8,5))

# Left axis: Training time
color1 = "tab:blue"
ax1.set_xlabel("Batch Size")
ax1.set_ylabel("Training Time (seconds)", color=color1)
ax1.plot(df["batch_size"], df["training_time_sec"], marker="o", color=color1, linewidth=2, label="Training Time")
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True)

# Right axis: MAE
ax2 = ax1.twinx()
color2 = "tab:red"
ax2.set_ylabel("Val MAE", color=color2)
ax2.plot(df["batch_size"], df["MAE_val"], marker="s", color=color2, linewidth=2, label="Val MAE")
ax2.tick_params(axis='y', labelcolor=color2)

# Title & layout
plt.title("Training Time and Val MAE vs Batch Size")
fig.tight_layout()

# Save figure
plot_path = os.path.join(OUTPUT_DIR, "training_time_and_mae_vs_batchsize.png")
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Saved training_time_and_mae_vs_batchsize.png in {OUTPUT_DIR}")


# Save CSV
pd.DataFrame(records).to_csv(os.path.join(OUTPUT_DIR,"batchsize_metrics.csv"),index=False)
print(f"\nSaved batchsize_metrics.csv in {OUTPUT_DIR}")
