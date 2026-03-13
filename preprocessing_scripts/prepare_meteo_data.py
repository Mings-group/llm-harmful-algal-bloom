import pandas as pd
import numpy as np
import os

# =========================
# CONFIG
# =========================
INPUT_DIR = "../data/meteorological_data"
OUTPUT_FILE = "METEO_ALL_SITES_2021-2024_weekly.csv"
# =========================

all_sites_weekly = []

# -------------------------------------------------
# Loop through meteorological CSV files
# -------------------------------------------------
for fname in os.listdir(INPUT_DIR):

    if not fname.endswith(".csv"):
        continue

    print(f"\nProcessing: {fname}")

    # -----------------------------------------
    # 🔹 Extract site names from filename
    # -----------------------------------------
    # Example: ETJ1_L5b_LAL_METEO_2021-2024.csv
    base_name = fname.replace(".csv", "")
    site_part = base_name.split("_METEO")[0]
    sites = site_part.split("_")

    print(f"Sites detected: {sites}")

    file_path = os.path.join(INPUT_DIR, fname)
    df = pd.read_csv(file_path)

    df = df.replace(-990, np.nan)

    df["date"] = pd.to_datetime(
        df["ANO"].astype(str) + "-" +
        df["MS"].astype(str) + "-" +
        df["DI"].astype(str),
        errors="coerce"
    )

    df = df[[
        "date",
        "T_MED",
        "T_MAX",
        "T_MIN",
        "FF_MED",
        "DD_MED",
        "DD_PRE",
        "PR_QTD"
    ]]

    df = df.rename(columns={
        "T_MED": "mean_temp",
        "T_MAX": "max_temp",
        "T_MIN": "min_temp",
        "FF_MED": "mean_wind_intensity",
        "DD_MED": "mean_wind_dir",
        "DD_PRE": "wind_dir",
        "PR_QTD": "rainfall"
    })

    df = df.set_index("date").sort_index()

    # -------------------------------------------------
    # Aggregate daily → weekly
    # -------------------------------------------------
    def mean_with_nan(series):
        clean = series.dropna()
        if clean.empty:
            return np.nan
        return clean.mean()

    weekly = df.resample("W-Wed").agg({
        "mean_temp": "mean",
        "max_temp": "max",
        "min_temp": "min",
        "mean_wind_intensity": "mean",
        "mean_wind_dir": "mean",
        "wind_dir": mean_with_nan,
        "rainfall": "sum"
    }).reset_index()

    # -------------------------------------------------
    # 🔹 Expand weekly data for each site
    # -------------------------------------------------
    for site in sites:
        site_df = weekly.copy()
        site_df["Site"] = site
        all_sites_weekly.append(site_df)

# -------------------------------------------------
# Combine everything
# -------------------------------------------------
final_meteo = pd.concat(all_sites_weekly, ignore_index=True)

# Sort properly
final_meteo = final_meteo.sort_values(["Site", "date"])

# Save unified file
final_meteo.to_csv(OUTPUT_FILE, index=False)

print("\n✔ Unified meteorological dataset saved:")
print(OUTPUT_FILE)