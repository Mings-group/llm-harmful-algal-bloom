#!/usr/bin/env python3
"""
Faithful LSTM Evaluation (RIAV1-only, t+1 to t+4) — MSE Models
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix
)

# ================= CONFIG ================= #
DATA_DIR = "../preprocessing_scripts/preprocessed_bi_uni_proper_split/w12"
MODEL_DIR = "results_lstm_faithful_mse_proper_split"
OUTPUT_DIR = "riav1_eval_lstm_faithful_mse_proper_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CM_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices")
os.makedirs(CM_DIR, exist_ok=True)

VARIANTS = ["univariate", "bivariate"]
SITE_ID = "RIAV1"
REG_LIMIT = 160.0
HORIZON = 4
# ========================================= #


def load_scalers(path):
    with open(path, "r") as f:
        return json.load(f)


def get_scaler_for_site(scalers, site_id, col_name="dsp_toxins"):
    if site_id in scalers and col_name in scalers[site_id]:
        return scalers[site_id][col_name]
    return scalers[col_name]


def safe_inverse_scale(values, scaler_dict):
    vals = np.array(values, dtype=float)
    mask = vals != -1
    out = np.full_like(vals, np.nan, dtype=float)
    col_min, col_max = scaler_dict["min"], scaler_dict["max"]
    out[mask] = vals[mask] * (col_max - col_min) + col_min
    return out


def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan
    rmse = math.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    r2 = r2_score(y_true[mask], y_pred[mask])
    return rmse, mae, r2


# ---------------- Load scalers ---------------- #
scalers = load_scalers(os.path.join(DATA_DIR, "..","scalers.json"))

all_variant_metrics = []

# ================= EVALUATION ================= #
for VARIANT in VARIANTS:

    print(f"\nEvaluating {VARIANT.upper()} (MSE model)")

    model_path = os.path.join(
        MODEL_DIR, f"{VARIANT}_best_model_final.keras"
    )
    if not os.path.exists(model_path):
        print(" Model not found — skipping.")
        continue

    model = load_model(model_path, compile=False)
    variant_path = os.path.join(DATA_DIR, VARIANT)

    dsp_scaler = get_scaler_for_site(scalers, SITE_ID)

    all_dates = []
    all_y_true = []
    all_y_pred = []
    split_boundaries = []

    for split in ["train", "val", "test"]:

        X = np.load(os.path.join(variant_path, f"X_{split}.npy"))
        y = np.load(os.path.join(variant_path, f"y_{split}.npy"))
        dates = np.load(
            os.path.join(variant_path, f"dates_{split}.npy"),
            allow_pickle=True
        )
        site_ids = np.load(
            os.path.join(variant_path, f"site_ids_{split}.npy"),
            allow_pickle=True
        )

        mask = site_ids == SITE_ID
        if mask.sum() == 0:
            continue

        X, y, dates = X[mask], y[mask], dates[mask]
        y_pred = model.predict(X, verbose=0)

        y_true_inv = safe_inverse_scale(y.squeeze(), dsp_scaler)
        y_pred_inv = safe_inverse_scale(y_pred.squeeze(), dsp_scaler)

        all_dates.extend(dates)
        all_y_true.append(y_true_inv)
        all_y_pred.append(y_pred_inv)
        split_boundaries.append(dates[-1])

        for h in range(HORIZON):

            rmse, mae, r2 = compute_metrics(
                y_true_inv[:, h], y_pred_inv[:, h]
            )

            all_variant_metrics.append({
                "variant": VARIANT,
                "site": SITE_ID,
                "split": split,
                "horizon": f"t+{h+1}",
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2
            })

            # ===== CLEAN CONFUSION MATRIX (BOTTOM-LEFT TN / TOP-RIGHT TP) =====
            exceed_true = (y_true_inv[:, h] > REG_LIMIT).astype(int)
            exceed_pred = (y_pred_inv[:, h] > REG_LIMIT).astype(int)

            cm = confusion_matrix(exceed_true, exceed_pred, labels=[0, 1])

            fig, ax = plt.subplots(figsize=(8, 8))

            # Set axis limits for square cells
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, 1.5)

            # Darker colors for better visibility
            # Order: bottom-left TN, bottom-right FN, top-left FP, top-right TP
            colors = np.array([
                  
                ["#66c2a5", "#e31a1c"],   # Top row: FP, TP
                ["#ff7f00", "#1f78b4"]  # Bottom row: TN, FN
            ])

            # Draw colored squares
            for i in range(2):
                for j in range(2):
                    rect = plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1, 1,
                        facecolor=colors[i, j],
                        alpha=0.55  # darker than before
                    )
                    ax.add_patch(rect)

            # Add counts in each cell
            # Flip cm indexing to match colors
            ax.text(0, 1, f"{cm[0,1]}", ha="center", va="center", fontsize=24, fontweight="bold")  # FP
            ax.text(1, 1, f"{cm[1,1]}", ha="center", va="center", fontsize=24, fontweight="bold")  # TP
            ax.text(0, 0, f"{cm[0,0]}", ha="center", va="center", fontsize=24, fontweight="bold")  # TN
            ax.text(1, 0, f"{cm[1,0]}", ha="center", va="center", fontsize=24, fontweight="bold")  # FN

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
                    f"{VARIANT}_{split}_t+{h+1}_confusion_matrix_mse.png"
                ),
                dpi=350,
                bbox_inches="tight"
            )
            plt.close(fig)


    # ================= TIME SERIES (t+1) ================= #
    if all_y_true:

        all_y_true = np.concatenate(all_y_true, axis=0)
        all_y_pred = np.concatenate(all_y_pred, axis=0)
        all_dates = np.array(all_dates)

        horizon_idx = 0
        shift_days = 7

        all_dates_dt = pd.to_datetime(all_dates)
        all_dates_shifted = all_dates_dt + pd.Timedelta(days=shift_days)

        fig, ax = plt.subplots(figsize=(13, 6))

        ax.plot(
            all_dates_shifted,
            all_y_true[:, horizon_idx],
            linewidth=1.2,
            label="Observed DSP toxins",
            color='blue'
        )

        ax.plot(
            all_dates_dt,
            all_y_pred[:, horizon_idx],
            linewidth=1.2,
            label="Predicted",
            color='red'
        )

        ax.axhline(
            REG_LIMIT,
            linestyle="--",
            linewidth=1.0,
            color='black',
            label="Regulatory Limit"
        )

        for boundary in split_boundaries[:-1]:
            ax.axvline(
                pd.to_datetime(boundary),
                linestyle="--",
                linewidth=1.0,
                color='black'
            )

        ax.set_xlim(
            pd.Timestamp("2015-01-01"),
            pd.Timestamp("2024-12-31")
        )

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax.set_ylabel("DSP toxins (µg OA equiv. kg$^{-1}$)", fontsize=22)
        ax.set_xlabel("Date", fontsize=22)
        ax.set_ylim(0, 1300)

        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)

        ax.legend(fontsize=14, loc="upper left")

        ax.grid(False)
        fig.tight_layout()

        fig.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{VARIANT}_RIAV1_t+1_forecast_mse.png"
            ),
            dpi=400,
            bbox_inches="tight"
        )

        plt.close(fig)

# ================= SAVE OUTPUTS ================= #
metrics_df = pd.DataFrame(all_variant_metrics)
metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lstm_RIAV1_all_variants_metrics_mse.csv"
    ),
    index=False
)

print("\nMSE evaluation complete.")
print(f"Results saved to: {OUTPUT_DIR}")


# ================= DUAL AXIS TEST METRICS PLOT ================= #

# Filter for test split and t+1 horizon
test_metrics = metrics_df[
    (metrics_df["split"] == "test") &
    (metrics_df["horizon"] == "t+1")
].copy()

# Map variants to number of features
feature_map = {
    "univariate": 1,
    "bivariate": 2,
    "multivariate":3
}

test_metrics["num_features"] = test_metrics["variant"].map(feature_map)

# Sort by number of features
test_metrics = test_metrics.sort_values("num_features")

x = test_metrics["num_features"].values
mae = test_metrics["MAE"].values
rmse = test_metrics["RMSE"].values

fig, ax1 = plt.subplots(figsize=(7, 4))

# ----- Left axis (MAE) -----
ax1.plot(
    x,
    mae,
    marker='o',
    linewidth=2,
    color='tab:red'
)

ax1.set_xlabel("Number of Features", fontsize=16)
ax1.set_ylabel(
    "MAE (µg AO equiv. kg$^{-1}$)",
    fontsize=14,
    color='tab:red'
)

ax1.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
ax1.tick_params(axis='x', labelsize=14)

ax1.set_xticks([1, 2, 3])
ax1.set_xticklabels(["1", "2", "3"], fontsize=14)

# ----- Right axis (RMSE) -----
ax2 = ax1.twinx()
ax2.plot(
    x,
    rmse,
    marker='o',
    linewidth=2,
    color='tab:blue'
)

ax2.set_ylabel("RMSE (µg AO equiv. kg$^{-1}$)", fontsize=14, color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
fig.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "RIAV1_test_dual_axis_MAE_RMSE.png"
    ),
    dpi=400,
    bbox_inches="tight"
)

plt.close(fig)