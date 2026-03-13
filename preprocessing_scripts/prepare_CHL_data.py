#!/usr/bin/env python3

import xarray as xr
import pandas as pd
import numpy as np

# ================= CONFIG ================= #

NC_FILE = "../data/chl_data/cmems_obs-oc_atl_bgc-plankton_my_l4-gapfree-multi-1km_P1D_1772465064549.nc"
VARIABLE_NAME = "CHL"

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

OUTPUT_CSV = "weekly_CHL_all_sites_long.csv"

SEARCH_RADIUS = 0.1

# ========================================== #

print("Opening NetCDF file...")
ds = xr.open_dataset(NC_FILE)
print("Dataset loaded.")

# Store all site dataframes here
all_sites_list = []

# ---------------- LOOP THROUGH SITES ----------------
for site_name, (lat, lon) in sites.items():

    print(f"\nProcessing site: {site_name}")

    box = ds.sel(
        latitude=slice(lat - SEARCH_RADIUS, lat + SEARCH_RADIUS),
        longitude=slice(lon - SEARCH_RADIUS, lon + SEARCH_RADIUS)
    )

    chl_data = box[VARIABLE_NAME]

    # Valid pixel = at least one non-NaN over time
    valid_mask = ~np.isnan(chl_data).all(dim="time")
    valid = np.where(valid_mask.values)

    if len(valid[0]) == 0:
        print("⚠ No valid CHL pixels nearby")
        continue

    # Distance calculation
    dist = (
        (box.latitude.values[valid[0]] - lat)**2 +
        (box.longitude.values[valid[1]] - lon)**2
    )

    idx = np.argmin(dist)

    selected_lat = box.latitude.values[valid[0][idx]]
    selected_lon = box.longitude.values[valid[1][idx]]

    print(f"Using CHL pixel at: {selected_lat}, {selected_lon}")

    point = ds.sel(latitude=selected_lat, longitude=selected_lon)

    series = point[VARIABLE_NAME].to_pandas()

    # Weekly aggregation
    weekly = series.resample("W-WED").mean()

    # Convert to dataframe
    site_df = weekly.reset_index()
    site_df.columns = ["Date", "mean_chl"]

    # Add site column
    site_df["Site"] = site_name

    # Append to list
    all_sites_list.append(site_df)

# ----------------------------------------------------

print("\nCombining all sites...")

final_df = pd.concat(all_sites_list, ignore_index=True)

# Sort properly
final_df = final_df.sort_values(["Site", "Date"])

print("Saving long-format CHL dataset...")
final_df.to_csv(OUTPUT_CSV, index=False)

print(f"Saved to: {OUTPUT_CSV}")
print("Done.")