#!/usr/bin/env python3
"""
Faithful CNN Evaluation (all sites, all horizons)
-------------------------------------------------
- Matches LSTM evaluation structure
- Site-specific folders for plots + metrics
- Plots all forecast horizons (t+1 … t+4)
- Site-specific or global scaling
- Safe inverse-scaling (masks -1)
- Split-level metrics (train, val, test)
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ================ CONFIG ================= #
DATA_DIR = "../preprocessed"
MODEL_DIR = "results_cnn_windowed"
OUTPUT_DIR = "cnn_eval_faithful"
os.makedirs(OUTPUT_DIR, exist_ok=True)

VARIANTS = ["univariate", "bivariate", "multivariate"]
SITES = ["ETJ1", "FAR1", "L5b", "L7c1", "LAG", "LAL", "POR2", "RIAV1"]
REG_LIMIT = 160.0
HORIZON = 4
# ========================================= #


def load_scalers(path):
    with open(path, "r") as f:
        return json.load(f)


def get_scaler_for_site(scalers, site_id, col_name="dsp_toxins"):
    """Return correct scaler dict (per-site if available, else global)."""
    if site_id in scalers and col_name in scalers[site_id]:
        return scalers[site_id][col_name], "site-specific"
    else:
        return scalers[col_name], "global"


def safe_inverse_scale(values, scaler_dict):
    """Inverse-scale safely (masking -1 placeholders)."""
    vals = np.array(values, dtype=float)
    mask = vals != -1
    out = np.full_like(vals, np.nan, dtype=float)
    col_min = scaler_dict["min"]
    col_max = scaler_dict["max"]
    out[mask] = vals[mask] * (col_max - col_min) + col_min
    return out


def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask) == 0:
        return np.nan, np.nan, np.nan
    rmse = math.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    r2 = r2_score(y_true[mask], y_pred[mask])
    return rmse, mae, r2


# ---------------- Load scalers ---------------- #
scalers = load_scalers(os.path.join(DATA_DIR, "scalers.json"))

# ---------------- Evaluate all sites & variants ---------------- #
all_variant_metrics = []

for SITE_ID in SITES:
    site_output_dir = os.path.join(OUTPUT_DIR, SITE_ID)
    os.makedirs(site_output_dir, exist_ok=True)
    print(f"\n==============================")
    print(f" Evaluating SITE: {SITE_ID}")
    print(f"==============================")

    site_metrics = []

    for VARIANT in VARIANTS:
        print(f"\n---- VARIANT: {VARIANT.upper()} ----")
        model_path = os.path.join(MODEL_DIR, f"{VARIANT}_cnn_final.keras")
        if not os.path.exists(model_path):
            print(f" Model not found: {model_path}")
            continue

        model = load_model(model_path, compile=False)
        variant_path = os.path.join(DATA_DIR, VARIANT)

        all_dates, all_y_true, all_y_pred, split_boundaries = [], [], [], []

        dsp_scaler, scaler_type = get_scaler_for_site(scalers, SITE_ID, "dsp_toxins")
        print(f"Scaler ({scaler_type}) min={dsp_scaler['min']}, max={dsp_scaler['max']}")

        # Evaluate each split
        for split in ["train", "val", "test"]:
            print(f"\n--- Evaluating {split.upper()} split ---")
            X_path = os.path.join(variant_path, f"X_{split}.npy")
            y_path = os.path.join(variant_path, f"y_{split}.npy")
            dates_path = os.path.join(variant_path, f"dates_{split}.npy")
            site_ids_path = os.path.join(variant_path, f"site_ids_{split}.npy")

            if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(dates_path) and os.path.exists(site_ids_path)):
                print(f" Missing files for {variant_path} / {split}, skipping.")
                continue

            X = np.load(X_path)
            y = np.load(y_path)
            dates = np.load(dates_path, allow_pickle=True)
            site_ids = np.load(site_ids_path, allow_pickle=True)

            # Filter current site
            mask = site_ids == SITE_ID
            X_site, y_site, dates_site = X[mask], y[mask], dates[mask]

            if len(X_site) == 0:
                print(f" No {SITE_ID} samples in {split} split.")
                continue

            y_pred = model.predict(X_site, verbose=0)
            if y_pred.ndim == 3:
                y_pred = y_pred.squeeze(-1)
            if y_site.ndim == 3:
                y_site = y_site.squeeze(-1)

            y_true_inv = safe_inverse_scale(y_site, dsp_scaler)
            y_pred_inv = safe_inverse_scale(y_pred, dsp_scaler)

            # Metrics
            for h in range(HORIZON):
                rmse, mae, r2 = compute_metrics(y_true_inv[:, h], y_pred_inv[:, h])
                rec = {
                    "variant": VARIANT,
                    "site": SITE_ID,
                    "split": split,
                    "horizon": f"t+{h+1}",
                    "RMSE": rmse,
                    "MAE": mae,
                    "R2": r2,
                }
                site_metrics.append(rec)
                all_variant_metrics.append(rec)
                print(f" {VARIANT} | {split.upper()} | t+{h+1} → MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")

            all_dates.extend(dates_site)
            all_y_true.append(y_true_inv)
            all_y_pred.append(y_pred_inv)
            split_boundaries.append(dates_site[-1])

        if not all_y_true:
            continue

        all_y_true = np.concatenate(all_y_true, axis=0)
        all_y_pred = np.concatenate(all_y_pred, axis=0)
        all_dates = np.array(all_dates)

        # Plot all horizons
        for h in range(HORIZON):
            horizon_label = f"t+{h+1}"
            shift_days = 7 * (h + 1)
            all_dates_shifted = pd.to_datetime(all_dates) + pd.Timedelta(days=shift_days)

            plt.figure(figsize=(13, 6))
            plt.plot(all_dates_shifted, all_y_true[:, h], color="blue", label=f"Actual DSP toxins (shifted {horizon_label})")
            plt.plot(all_dates, all_y_pred[:, h], color="red", label=f"Predicted ({horizon_label})")
            plt.axhline(REG_LIMIT, color="black", linewidth=1, label="Regulatory Limit")

            for boundary in split_boundaries[:-1]:
                plt.axvline(boundary, color="grey", linestyle="--", linewidth=1)

            plt.title(f"{SITE_ID} – CNN {VARIANT.capitalize()} Forecast ({horizon_label}) [{scaler_type}]")
            plt.xlabel("Time")
            plt.ylabel("DSP toxins (µg AO kg$^{-1}$)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(site_output_dir, f"{VARIANT}_{horizon_label}_forecast.png"), dpi=300)
            plt.close()

    # Save per-site metrics
    site_metrics_df = pd.DataFrame(site_metrics)
    site_metrics_path = os.path.join(site_output_dir, f"{SITE_ID}_cnn_metrics.csv")
    site_metrics_df.to_csv(site_metrics_path, index=False)
    print(f"📊 Saved per-site metrics: {site_metrics_path}")

# ---------------- Save global metrics ---------------- #
metrics_df = pd.DataFrame(all_variant_metrics)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "cnn_all_sites_all_variants_metrics.csv"), index=False)

print("\n✅ Evaluation complete.")
print(f"Results saved under: {OUTPUT_DIR}")
