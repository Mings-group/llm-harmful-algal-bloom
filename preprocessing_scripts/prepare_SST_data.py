#!/usr/bin/env python3

import xarray as xr
import pandas as pd
import numpy as np

# ================= CONFIG ================= #

REP_FILE = "../data/SST_data/METOFFICE-GLO-SST-L4-REP-OBS-SST_1772548026992.nc"
NRT_FILE = "../data/SST_data/METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2_1772548009194.nc"

VARIABLE_NAME = "analysed_sst"

sites = {
    "LAL":   (38.51254, -9.17211),
    "RIAV1": (40.69556, -8.71167),
    "ETJ1":  (38.77817, -9.03867),
    "L5b":   (38.614473, -9.22468),
    "LAG":   (37.134, -8.62205),
    "L7c1":  (37.03423, -8.84875),
    "POR2":  (37.13217, -8.5975),
    "FAR1":  (37.01867, -7.94833),
}

OUTPUT_CSV = "weekly_SST_all_sites.csv"

SEARCH_RADIUS = 0.1

# ========================================== #

print("Opening REP dataset...")
ds_rep = xr.open_dataset(REP_FILE)

print("Opening NRT dataset...")
ds_nrt = xr.open_dataset(NRT_FILE)

print("Concatenating REP + NRT...")
ds = xr.concat([ds_rep, ds_nrt], dim="time")
ds = ds.sortby("time")

# Remove duplicate timestamps if overlap exists
time_index = ds.get_index("time")
ds = ds.sel(time=~time_index.duplicated())

print("Dataset ready.")

# Store all results here (LONG format)
all_sites_long = []

# ---------------- LOOP THROUGH SITES ----------------
for site_name, (lat, lon) in sites.items():

    print(f"\nProcessing site: {site_name}")

    box = ds.sel(
        latitude=slice(lat - SEARCH_RADIUS, lat + SEARCH_RADIUS),
        longitude=slice(lon - SEARCH_RADIUS, lon + SEARCH_RADIUS)
    )

    sst_data = box[VARIABLE_NAME]

    valid_mask = ~np.isnan(sst_data).all(dim="time")
    valid = np.where(valid_mask.values)

    if len(valid[0]) == 0:
        print("⚠ No valid SST pixels nearby")
        continue

    dist = (
        (box.latitude.values[valid[0]] - lat)**2 +
        (box.longitude.values[valid[1]] - lon)**2
    )

    idx = np.argmin(dist)

    selected_lat = box.latitude.values[valid[0][idx]]
    selected_lon = box.longitude.values[valid[1][idx]]

    print(f"Using SST pixel at: {selected_lat}, {selected_lon}")

    point = ds.sel(latitude=selected_lat, longitude=selected_lon)

    series = point[VARIABLE_NAME].to_pandas()

    if series.isna().all():
        print("⚠ Selected pixel contains only NaNs")
        continue

    weekly = series.resample("W-WED").mean()

    # Convert to long format
    df_site = weekly.reset_index()
    df_site.columns = ["time", "mean_sst"]
    df_site["site"] = site_name

    all_sites_long.append(df_site)

# ----------------------------------------------------

print("\nCombining all sites...")

final_df = pd.concat(all_sites_long, ignore_index=True)

# Reorder columns
final_df = final_df[["time", "site", "mean_sst"]]

print("Saving dataset...")
final_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved to: {OUTPUT_CSV}")
print("Done.")