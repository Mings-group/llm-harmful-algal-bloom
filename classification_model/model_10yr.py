#!/usr/bin/env python3
"""
Faithful LSTM Classification Evaluation (RIAV1-only)
---------------------------------------------------
Mirrors regression evaluation structure exactly.

Includes:
 - Per-horizon metrics (accuracy, precision, recall, F1)
 - Confusion matrices
 - Timeline plot (t+1)
 - Proper inverse scaling for visualization only
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ================= CONFIG ================= #
DATA_DIR = "../preprocessing_scripts/preprocessed_classification_proper_split/w12"
MODEL_DIR = "results"
OUTPUT_DIR = "riav1_eval_lstm_classification_faithful_proper_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CM_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices")
os.makedirs(CM_DIR, exist_ok=True)

VARIANTS = ["bivariate"]
SITE_ID = "RIAV1"
REG_LIMIT = 160.0
HORIZON = 4
THRESHOLD = 0.5
# ========================================== #

# ---------------- Load scalers ----------------
with open(os.path.join(DATA_DIR,'..', "scalers.json")) as f:
    SCALERS = json.load(f)

tox_min = SCALERS["dsp_toxins"]["min"]
tox_max = SCALERS["dsp_toxins"]["max"]

def inverse_scale(values):
    vals = np.array(values, dtype=float)
    return vals * (tox_max - tox_min) + tox_min


all_variant_metrics = []

# ================= EVALUATION ================= #
for VARIANT in VARIANTS:

    print(f"\nEvaluating {VARIANT.upper()}")

    model_path = os.path.join(
        MODEL_DIR,
        f"{VARIANT}_classification_proper_split/lstm_classifier.keras"
    )

    if not os.path.exists(model_path):
        print("Model not found — skipping.")
        continue

    model = load_model(model_path, compile=False)
    variant_path = os.path.join(DATA_DIR, VARIANT)

    all_dates = []
    all_y_true = []
    all_y_pred = []
    split_boundaries = []

    for split in ["train", "val", "test"]:

        X = np.load(os.path.join(variant_path, f"X_{split}.npy"))
        y_bin = np.load(os.path.join(variant_path, f"y_{split}.npy"))
        y_cont = np.load(os.path.join(variant_path, f"y_cont_{split}.npy"))
        dates = np.load(os.path.join(variant_path, f"dates_{split}.npy"), allow_pickle=True)
        site_ids = np.load(os.path.join(variant_path, f"site_ids_{split}.npy"), allow_pickle=True)

        mask = site_ids == SITE_ID
        if mask.sum() == 0:
            continue

        X = X[mask]
        y_bin = y_bin[mask]
        y_cont = y_cont[mask]
        dates = dates[mask]

        y_probs = model.predict(X, verbose=0)
        y_pred = (y_probs >= THRESHOLD).astype(int)

        # ---------------- Metrics per horizon ----------------
        for h in range(HORIZON):

            yt = y_bin[:, h]
            yp = y_pred[:, h]

            acc = accuracy_score(yt, yp)
            prec = precision_score(yt, yp, zero_division=0)
            rec = recall_score(yt, yp, zero_division=0)
            f1 = f1_score(yt, yp, zero_division=0)

            all_variant_metrics.append({
                "variant": VARIANT,
                "site": SITE_ID,
                "split": split,
                "horizon": f"t+{h+1}",
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1
            })

            # ===== CUSTOM CLEAN CONFUSION MATRIX =====
            cm = confusion_matrix(yt, yp, labels=[0, 1])

            fig, ax = plt.subplots(figsize=(8, 8))

            # Set square axis
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, 1.5)

            # Color layout:
            # Top row: FP, TP
            # Bottom row: TN, FN
            colors = np.array([
                ["#66c2a5", "#e31a1c"],   # Top row: FP, TP
                ["#ff7f00", "#1f78b4"]    # Bottom row: TN, FN
            ])

            # Draw colored squares
            for i in range(2):
                for j in range(2):
                    rect = plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1, 1,
                        facecolor=colors[i, j],
                        alpha=0.55
                    )
                    ax.add_patch(rect)

            # Add counts (mapping from sklearn cm)
            # cm layout from sklearn:
            # [[TN, FP],
            #  [FN, TP]]

            ax.text(0, 1, f"{cm[0,1]}", ha="center", va="center",
                    fontsize=24, fontweight="bold")  # FP

            ax.text(1, 1, f"{cm[1,1]}", ha="center", va="center",
                    fontsize=24, fontweight="bold")  # TP

            ax.text(0, 0, f"{cm[0,0]}", ha="center", va="center",
                    fontsize=24, fontweight="bold")  # TN

            ax.text(1, 0, f"{cm[1,0]}", ha="center", va="center",
                    fontsize=24, fontweight="bold")  # FN

            # Axis ticks
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])

            ax.set_xticklabels(["Below", "Above"], fontsize=20)
            ax.set_yticklabels(["Below", "Above"], fontsize=20)

            ax.set_xlabel("Observed Contamination", fontsize=22)
            ax.set_ylabel("Predicted Contamination", fontsize=22)

            ax.set_aspect("equal")
            ax.grid(False)

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    CM_DIR,
                    f"{VARIANT}_{split}_t+{h+1}_confusion_matrix.png"
                ),
                dpi=350,
                bbox_inches="tight"
            )
            plt.close(fig)


        # Store for timeline (t+1 only)
        all_dates.extend(dates)
        all_y_true.append(y_cont[:, 0])
        all_y_pred.append(y_pred[:, 0])
        split_boundaries.append(dates[-1])

   # ================= Timeline Plot =================
if all_y_true:

    all_y_true = inverse_scale(np.concatenate(all_y_true))
    all_y_pred = np.concatenate(all_y_pred)
    all_dates = pd.to_datetime(np.array(all_dates))

    correct = all_y_pred == (
        (all_y_true > REG_LIMIT).astype(int)
    )

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(
        all_dates,
        all_y_true,
        color="blue",
        linewidth=1.2,
        label="Observed DSP toxins"
    )

    ax.scatter(
        all_dates[correct],
        all_y_true[correct],
        edgecolors="blue",
        s=60,
        facecolors="none",
        label="Correct"
    )

    ax.scatter(
        all_dates[~correct],
        all_y_true[~correct],
        edgecolors="red",
        s=60,
        facecolors="none",
        label="Incorrect"
    )

    ax.axhline(
        REG_LIMIT,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Regulatory Limit"
    )

    for boundary in split_boundaries[:-1]:
        ax.axvline(
            pd.to_datetime(boundary),
            color="grey",
            linestyle="--",
            linewidth=1.0
        )

    # ---- HARD LIMIT YEARS 2015–2025 ----
    ax.set_xlim(
        pd.Timestamp("2015-01-01"),
        pd.Timestamp("2024-12-31")
    )

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ---- MATCH REGRESSION FONT SIZES ----
    ax.set_ylabel(
        "DSP toxins (µg OA equiv. kg$^{-1}$)",
        fontsize=24
    )
    ax.set_xlabel("Date", fontsize=24)
    ax.set_ylim(0, 1300)

    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)

    if VARIANT == "bivariate":
        ax.legend(fontsize=14, loc="upper left")

    ax.grid(False)
    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{VARIANT}_RIAV1_t+1_classification.png"
        ),
        dpi=400,
        bbox_inches="tight"
    )

    plt.close(fig)


# ================= SAVE METRICS =================
pd.DataFrame(all_variant_metrics).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lstm_RIAV1_all_variants_classification_metrics.csv"
    ),
    index=False
)

print("\nClassification evaluation complete.")
