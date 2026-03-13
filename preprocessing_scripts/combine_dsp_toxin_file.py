import pandas as pd

# =========================
XLSX_FILE = "../data/all_sites/Dados_2021-2025.xlsx"
OUTPUT_CSV = "DSP_toxins_combined.csv"
# =========================

# Portuguese month mapping
month_map = {
    "jan": "01", "fev": "02", "mar": "03", "abr": "04", "mai": "05", "jun": "06",
    "jul": "07", "ago": "08", "set": "09", "out": "10", "nov": "11", "dez": "12"
}

xls = pd.ExcelFile(XLSX_FILE)
sheet_names = xls.sheet_names
all_data = []

for sheet in sheet_names:
    print(f"Processing sheet: {sheet}")
    df = pd.read_excel(xls, sheet_name=sheet)

    # Determine year from sheet
    try:
        year = int(sheet)
    except ValueError:
        year = None

    # Normalize the DSP column
    if "DSP_Toxicity" in df.columns:
        df = df.rename(columns={"DSP_Toxicity": "dsp_toxins"})
    elif "DSP_toxixity" in df.columns:
        df = df.rename(columns={"DSP_toxixity": "dsp_toxins"})
    else:
        raise KeyError(f"No DSP toxicity column found in sheet '{sheet}'")

    # Process Date column
    if "Date" in df.columns:
        # Case 1: Already datetime (Excel datetime objects)
        if pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["date"] = df["Date"]
        else:
            # Convert to string and lowercase
            date_str = df["Date"].astype(str).str.lower().str.strip()
            # Replace month abbreviations with numbers
            for pt_month, num in month_map.items():
                date_str = date_str.str.replace(pt_month, num, regex=False)
            # Append year if known
            if year is not None:
                date_str = date_str + f"-{year}"
            # Parse to datetime
            df["date"] = pd.to_datetime(date_str, format="%d-%m-%Y", errors="coerce")

    # Keep only relevant columns
    df = df[["Station", "date", "dsp_toxins"]]

    # Drop rows where date parsing failed
    df = df.dropna(subset=["date"])

    all_data.append(df)

# Combine all sheets
combined_df = pd.concat(all_data, ignore_index=True)

# Sort by station and date
combined_df = combined_df.sort_values(["Station", "date"])

# Reset time to midnight (drop any time info)
combined_df["date"] = combined_df["date"].dt.normalize()

# Save CSV
combined_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Combined CSV saved as: {OUTPUT_CSV}")
