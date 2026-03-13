#!/usr/bin/env python3
import pandas as pd

# ===================== CONFIG ===================== #
OUTPUT_FILE = "MASTER_DATA_FILE.csv"

CHL_FILE   = "weekly_CHL_all_sites_long.csv"
SST_FILE   = "weekly_SST_all_sites.csv"
METEO_FILE = "METEO_ALL_SITES_2021-2024_weekly.csv"
DSP_FILE   = "DSP_phyto_merged_all_stations.csv"
# =================================================== #

# ---------- Load datasets ----------
dsp = pd.read_csv(DSP_FILE, parse_dates=['date'])
dsp = dsp.rename(columns={'Station':'site','date':'Date'})

chl = pd.read_csv(CHL_FILE, parse_dates=['Date'])
chl = chl.rename(columns={'Site':'site'})

sst = pd.read_csv(SST_FILE, parse_dates=['time'])
sst = sst.rename(columns={'site':'site','analysed_sst':'mean_sst'})
sst = sst[['time','site','mean_sst']].rename(columns={'time':'Date'})

meteo = pd.read_csv(METEO_FILE, parse_dates=['date'])
meteo = meteo.rename(columns={'Site':'site','date':'Date'})

# =========================================================
# 🔹 Align DSP dates to Wednesday weekly anchor
# =========================================================
dsp["Date"] = (
    dsp["Date"]
    .dt.to_period("W-WED")
    .dt.end_time
    .dt.normalize()
)

# =========================================================
# 🔹 Merge on aligned weekly Date
# =========================================================
df_master = (
    dsp
    .merge(chl, on=['site','Date'], how='left')
    .merge(sst, on=['site','Date'], how='left')
    .merge(meteo, on=['site','Date'], how='left')
)

# ---------- Sort and save ----------
df_master = df_master.sort_values(['site','Date']).reset_index(drop=True)
df_master.to_csv(OUTPUT_FILE, index=False)

print(f"Master data file saved: {OUTPUT_FILE}")