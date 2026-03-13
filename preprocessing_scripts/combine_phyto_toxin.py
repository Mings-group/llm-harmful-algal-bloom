import pandas as pd

VALID_STATIONS = ["ETJ1", "FAR1", "L5b", "L7c1", "LAG", "LAL", "POR2", "RIAV1"]

print("Loading data...")

phyto = pd.read_csv("../data/phytoplankton_data/all_phyto.csv", parse_dates=["date"])
dsp = pd.read_csv("DSP_toxins_combined.csv", parse_dates=["date"])

phyto["date"] = pd.to_datetime(phyto["date"]).astype("datetime64[ns]")
dsp["date"] = pd.to_datetime(dsp["date"]).astype("datetime64[ns]")

# Ensure correct sorting (REQUIRED for merge_asof)
phyto = phyto.sort_values(["Station", "date"]).reset_index(drop=True)
dsp = dsp.sort_values(["Station", "date"]).reset_index(drop=True)

merged_sites = []

for site in VALID_STATIONS:

    print(f"\n🔎 Merging site: {site}")

    dsp_site = dsp[dsp["Station"] == site].copy()
    phyto_site = phyto[phyto["Station"] == site].copy()

    if dsp_site.empty:
        print(f"⚠️  No DSP data for {site} — skipping")
        continue

    if phyto_site.empty:
        print(f"⚠️  No phyto data for {site} — phyto will be NaN")

    dsp_site = dsp_site.sort_values("date").reset_index(drop=True)
    phyto_site = phyto_site.sort_values("date").reset_index(drop=True)

    # Only keep columns needed from phyto
    phyto_subset = phyto_site[["date", "phyto"]].copy()

    merged = pd.merge_asof(
        dsp_site,
        phyto_subset,
        on="date",
        direction="nearest",
        tolerance=pd.Timedelta(days=3)
    )

    print(f"   DSP rows: {len(dsp_site)}")
    print(f"   matched phyto rows: {merged['phyto'].notna().sum()}")

    merged_sites.append(merged)

# Combine all stations
final_df = pd.concat(merged_sites, ignore_index=True)
final_df = final_df.sort_values(["Station", "date"]).reset_index(drop=True)

# Save output
output_file = "DSP_phyto_merged_all_stations.csv"
final_df.to_csv(output_file, index=False)

print(f"\n🎉 Final merged dataset created: {output_file}")
