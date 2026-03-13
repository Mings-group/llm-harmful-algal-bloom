#!/usr/bin/env python3
"""
Single-file evaluation + paper-style time series plot
DSP exceedance classification (bivariate LSTM)
FULL train/val/test continuous timeline
"""

# ================= IMPORTS ================= #
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ================= CONFIG ================= #
VARIANT = "bivariate"
SITE_ID = "RIAV1"

DATA_DIR = "../../preprocessed_classification"
MODEL_PATH = "results/bivariate_classification/lstm_classifier.keras"
OUTPUT_DIR = f"eval_{SITE_ID}_{VARIANT}"

TIME_HORIZON = 4
THRESHOLD = 0.5
REG_LIMIT = 160.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD MODEL ================= #
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)

# =========================================================
# --- LOAD + PREDICT ALL SPLITS (train / val / test) ---
# =========================================================
variant_dir = os.path.join(DATA_DIR, VARIANT)

all_dates = []
all_y_obs = []
all_pred_event = []
all_actual_event = []
split_boundaries = []

print("Loading data and running predictions...")

for split in ["train", "val", "test"]:
    X = np.load(os.path.join(variant_dir, f"X_{split}.npy"))
    y = np.load(os.path.join(variant_dir, f"y_{split}.npy"))
    y_cont = np.load(os.path.join(variant_dir, f"y_cont_{split}.npy"))
    dates = np.load(os.path.join(variant_dir, f"dates_{split}.npy"), allow_pickle=True)
    sites = np.load(os.path.join(variant_dir, f"site_ids_{split}.npy"), allow_pickle=True)

    mask = sites == SITE_ID
    if not np.any(mask):
        continue

    X = X[mask]
    y = y[mask]
    y_cont = y_cont[mask]
    dates = pd.to_datetime(dates[mask])

    # Predict
    y_probs = model.predict(X, verbose=0)
    y_pred = (y_probs >= THRESHOLD).astype(int)

    # t+1 horizon only
    y_obs = y_cont[:, 0]
    pred_event = y_pred[:, 0] == 1
    actual_event = y_obs >= REG_LIMIT

    all_dates.append(dates)
    all_y_obs.append(y_obs)
    all_pred_event.append(pred_event)
    all_actual_event.append(actual_event)

    split_boundaries.append(dates[-1])


# Concatenate full timeline
dates_all = pd.to_datetime(np.concatenate(all_dates))
y_obs_all = np.concatenate(all_y_obs)
pred_event_all = np.concatenate(all_pred_event)
actual_event_all = np.concatenate(all_actual_event)

# ================= METRICS (TEST ONLY) ================= #
print("\nClassification metrics (TEST split):")
test_mask = np.arange(len(dates_all)) >= (
    len(dates_all) - len(all_dates[-1])
)

print(classification_report(
    actual_event_all[test_mask].astype(int),
    pred_event_all[test_mask].astype(int),
    digits=3,
    zero_division=0
))

# =========================================================
# --- PAPER-STYLE CLASSIFIED TIME SERIES ---
# =========================================================
correct = actual_event_all == pred_event_all
incorrect = ~correct

# Offset used ONLY for the decision indicator (not the dots)
delta = 15.0
y_pred_class = np.where(
    pred_event_all,
    REG_LIMIT + delta,
    REG_LIMIT - delta
)

fig, ax = plt.subplots(figsize=(13, 6))

# --- Continuous observations ---
ax.plot(
    dates_all,
    y_obs_all,
    color="blue",
    linewidth=1.2,
    label="Observations",
    zorder=1
)

# --- Correct predictions (blue dots at observed values) ---
ax.scatter(
    dates_all[correct],
    y_obs_all[correct],
    edgecolors="blue",
    s=60,
    label="Correct prediction",
    zorder=3,
    facecolors='none'
)


# --- Incorrect prediction markers (RED DOTS AT OBSERVED VALUES) ---
ax.scatter(
    dates_all[incorrect],
    y_obs_all[incorrect],   # ✅ FIX: observed value, not decision band
    edgecolors="red",
    facecolors="none",
    s=60,
    label="Incorrect prediction",
    zorder=4,
    
)

# --- Regulatory limit ---
ax.axhline(
    REG_LIMIT,
    color="gray",
    linestyle="--",
    linewidth=1.0,
    label="Regulatory limit"
)

# --- Train / Val / Test boundaries ---
for boundary in split_boundaries[:-1]:
    ax.axvline(
        boundary,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8
    )

ax.set_ylabel("DSP toxins (µg AO equiv. kg$^{-1}$)",fontsize=24)
ax.set_xlabel("Date",fontsize=24)
ax.set_ylim(top=1100)

ax.tick_params(axis='x', labelsize=20)  
ax.tick_params(axis='y', labelsize=20)  

#ax.set_title(f"{SITE_ID} — Bivariate LSTM — t+1 Classification")

ax.legend(
    loc="upper left",
    ncol=4,
    frameon=False,
    fontsize=14
)

ax.grid(False)

plt.figure(figsize=(13, 6))
fig.tight_layout()
fig.savefig(
    os.path.join(OUTPUT_DIR, "classified_timeseries_t1_full_splits.png"),
    dpi=400,
    bbox_inches="tight"
)
plt.close(fig)


print("Saved plot to:", OUTPUT_DIR)


# ================= CONFUSION MATRIX (TEST, t+1) ================= #

# Binary arrays (TEST only)
y_true = actual_event_all[test_mask].astype(int)   # 1 = Yes, 0 = No
y_pred = pred_event_all[test_mask].astype(int)

# Standard confusion matrix: [[TN, FP],
#                              [FN, TP]]
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

# Reorder to:
# rows = Observed [Yes, No]
# cols = Predicted [No, Yes]
cm_reordered = np.array([
    [cm[1, 0], cm[1, 1]],  # Observed Yes → FN, TP
    [cm[0, 0], cm[0, 1]]   # Observed No  → TN, FP
])

fig, ax = plt.subplots(figsize=(8, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_reordered,
    display_labels=["No", "Yes"]
)

disp.plot(
    ax=ax,
    cmap=plt.cm.Blues,
    colorbar=False,
    values_format="d"
)

for text in disp.text_.ravel():
    text.set_fontsize(20)

# ---- Make cells square (important for paper figures) ----
ax.set_aspect("equal")

# ---- MATCH COLORBAR HEIGHT TO MATRIX ----
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

cbar = fig.colorbar(disp.im_, cax=cax)
cbar.ax.tick_params(labelsize=20)

# Axis labels
ax.set_xlabel("Predicted contamination", fontsize=22)
ax.set_ylabel("Observed contamination", fontsize=22)

# Tick labels
ax.set_xticklabels(["No", "Yes"], fontsize=20)
ax.set_yticklabels(["Yes", "No"], fontsize=20)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix_test_t1.png"),
    dpi=350,
    bbox_inches="tight"
)
plt.close(fig)

