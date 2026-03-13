import pandas as pd
import numpy as np
import glob
import os
import json


# ====================== CONFIG ====================== #
INPUT_PATH = "../data/all_sites/*.xlsx"
EXTENSION_FILE = "../data/MASTER_DATA_FILE.csv"
OUTPUT_DIR = "preprocessed_bi_uni_proper_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SITES_USED = ["RIAV1", "L5b", "ETJ1", "LAL", "L7c1", "LAG", "POR2", "FAR1"]

EXPECTED_COLS = [
    "dsp_toxins",
    "dsp_phyto",
    "mean_sst",
    "mean_chl",
    "mean_temp",
    "max_temp",
    "min_temp",
    "mean_wind_intensity",
    "wind_dir",
    "rainfall"
]

FORECAST_HORIZON = 4
WINDOW_SIZES = [8,12,16]
# ===================================================== #


# =====================================================
# LOAD + MERGE ORIGINAL FILES WITH EXTENSION CSV
# =====================================================
def load_and_merge_sites(input_path, extension_csv):

    all_dfs = []

    print("\nLoading master extension dataset...")
    ext = pd.read_csv(extension_csv, parse_dates=["Date"])

    # normalize site names for safety
    ext["site"] = ext["site"].astype(str).str.strip().str.upper()

    for file in glob.glob(input_path):

        site_code = os.path.basename(file).split("_")[0]

        if site_code not in SITES_USED:
            continue

        site_code_norm = site_code.strip().upper()

        print(f"Processing site: {site_code}")

        # -------- ORIGINAL XLSX --------
        df_orig = pd.read_excel(file, parse_dates=["Date"])

        # normalize site col in original if exists
        if "Site" in df_orig.columns:
            df_orig["Site"] = df_orig["Site"].astype(str).str.strip().str.upper()

        # -------- EXTENSION FILTER --------
        df_ext = ext[ext["site"] == site_code_norm].copy()

        if len(df_ext) > 0:
            df_ext = df_ext.drop(columns=["site"])

        # -------- COMBINE --------
        df = pd.concat([df_orig, df_ext], ignore_index=True)

        df = df.set_index("Date").sort_index()

        # weekly resample (paper rule)
        df = df.resample("W-WED").mean()

        # ensure expected cols exist
        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = np.nan

        # clip negatives
        df["dsp_toxins"] = df["dsp_toxins"].clip(lower=0)

        # missing handling (UNCHANGED)
        missing_frac = df[EXPECTED_COLS].isna().mean()

        for col in EXPECTED_COLS:
            if missing_frac[col] <= 0.25:
                df[col] = df[col].interpolate(method="time", limit_direction="both")
            else:
                df[col] = df[col].fillna(-1)

        df["Site"] = site_code
        all_dfs.append(df)

    df_all = pd.concat(all_dfs).sort_index()
    df_all[EXPECTED_COLS] = df_all[EXPECTED_COLS].fillna(-1)

    return df_all


# =====================================================
# NORMALIZATION
# =====================================================
def global_normalize(df, cols):

    scalers = {}

    for col in cols:

        valid_vals = df.loc[df[col] != -1, col]

        if valid_vals.empty:
            col_min, col_max = 0.0, 1.0
        else:
            col_min, col_max = valid_vals.min(), valid_vals.max()

        scalers[col] = {"min": float(col_min), "max": float(col_max)}

        if col_max > col_min:
            df[col] = np.where(
                df[col] == -1,
                -1,
                (df[col] - col_min) / (col_max - col_min)
            )
        else:
            df[col] = 0.0

    return df, scalers


# =====================================================
# SPLIT
# =====================================================
def chronological_split(df):

    df = df.sort_index()

    unique_dates = np.sort(df.index.unique())
    n_total = len(unique_dates)

    # -------------------------
    # Paper ratio = 5 : 3 : 2
    # -------------------------
    ratio_train = 5
    ratio_val   = 3
    ratio_test  = 2

    total_ratio = ratio_train + ratio_val + ratio_test

    train_end_idx = int(n_total * (ratio_train / total_ratio))
    val_end_idx   = int(n_total * ((ratio_train + ratio_val) / total_ratio))

    train_end_date = unique_dates[train_end_idx]
    val_end_date   = unique_dates[val_end_idx]

    def get_split(date):

        if date <= train_end_date:
            return "train"
        elif date <= val_end_date:
            return "val"
        else:
            return "test"

    df["split"] = df.index.to_series().apply(get_split)

    print("\nChronological split summary:")
    print("Total weeks:", n_total)
    print("Ratio used: 5:3:2")
    print("Train until:", train_end_date)
    print("Val until:", val_end_date)

    return df


def chronological_split_manual(df):
    df = df.sort_index()
    
    # Explicit date boundaries
    train_end_date = pd.Timestamp("2019-12-31")
    val_end_date   = pd.Timestamp("2022-12-31")
    
    def get_split(date):
        if date <= train_end_date:
            return "train"
        elif date <= val_end_date:
            return "val"
        else:
            return "test"
    
    df["split"] = df.index.to_series().apply(get_split)
    return df




# =====================================================
# SUPERVISED WINDOWS
# =====================================================
def make_supervised(df, features, target, time_steps, horizon):

    X_all, y_all, dates_all, site_ids_all, splits_all = [], [], [], [], []

    for site, group in df.groupby("Site"):

        group = group.sort_index()
        data = group[features + [target]].values
        target_idx = group.columns.get_loc(target)
        dates = group.index.values
        split_col = group["split"].values

        for i in range(len(group) - time_steps - horizon+1):

            X_all.append(data[i:i + time_steps, :len(features)])
            y_all.append(data[i + time_steps:i + time_steps + horizon, target_idx])

            dates_all.append(dates[i + time_steps])
            site_ids_all.append(site)
            splits_all.append(split_col[i + time_steps])

    return (
        np.array(X_all),
        np.array(y_all),
        np.array(dates_all),
        np.array(site_ids_all),
        np.array(splits_all)
    )


# =====================================================
# SAVE
# =====================================================
def save_datasets(X, y, dates, site_ids, splits, output_path):

    os.makedirs(output_path, exist_ok=True)

    for s in ["train","val","test"]:

        idx = splits == s

        np.save(os.path.join(output_path,f"X_{s}.npy"),X[idx])
        np.save(os.path.join(output_path,f"y_{s}.npy"),y[idx])
        np.save(os.path.join(output_path,f"dates_{s}.npy"),dates[idx])
        np.save(os.path.join(output_path,f"site_ids_{s}.npy"),site_ids[idx])


def apply_scalers(df, scalers, cols):

    for col in cols:

        col_min = scalers[col]["min"]
        col_max = scalers[col]["max"]

        if col_max > col_min:
            df[col] = np.where(
                df[col] == -1,
                -1,
                (df[col] - col_min) / (col_max - col_min)
            )
        else:
            df[col] = 0.0

    return df



# =====================================================
# PIPELINE
# =====================================================
print("\nLoading and merging all sites...")
df_all = load_and_merge_sites(INPUT_PATH, EXTENSION_FILE)

# Date filter
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"

df_all = df_all.loc[
    (df_all.index >= START_DATE) &
    (df_all.index <= END_DATE)
]

print("\nDate range after filtering:")
print("Start:", df_all.index.min())
print("End:", df_all.index.max())
print("Total unique weeks:", df_all.index.nunique())

# Split first
print("\nApplying split...")
df_all = chronological_split_manual(df_all)

# Compute scalers on TRAIN only
print("\nComputing scalers from TRAIN only...")
train_df = df_all[df_all["split"] == "train"].copy()
_, scalers = global_normalize(train_df, EXPECTED_COLS)

# Apply scalers everywhere
print("\nApplying scalers to full dataset...")
df_all = apply_scalers(df_all, scalers, EXPECTED_COLS)

# Save cleaned data
df_all.to_csv(os.path.join(OUTPUT_DIR,"merged_cleaned.csv"))

with open(os.path.join(OUTPUT_DIR,"scalers.json"),"w") as f:
    json.dump(scalers,f,indent=4)



VARIANTS = {
    "univariate": ["dsp_toxins"],
    "bivariate": ["dsp_toxins","dsp_phyto"],
    "multivariate": EXPECTED_COLS
}
for w in WINDOW_SIZES:

    print(f"\n=== Window {w} ===")

    for variant, feats in VARIANTS.items():

        print(f"Processing {variant}")

        X,y,dates,site_ids,splits = make_supervised(
            df_all,
            feats,
            target="dsp_toxins",
            time_steps=w,
            horizon=FORECAST_HORIZON
        )

        out_dir = os.path.join(OUTPUT_DIR,f"w{w}",variant)

        save_datasets(X,y,dates,site_ids,splits,out_dir)




print("\n✅ UNI + BI PREPROCESSING COMPLETE")



