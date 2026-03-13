import pandas as pd
import os

# -------- CONFIG --------
data_dir = "."
sheet_name = 0   # first sheet
# ------------------------

for fname in os.listdir(data_dir):
    if not fname.lower().endswith(".xlsx"):
        continue

    xlsx_path = os.path.join(data_dir, fname)
    csv_name = os.path.splitext(fname)[0] + ".csv"
    csv_path = os.path.join(data_dir, csv_name)

    print(f"Converting: {fname}")

    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df.to_csv(csv_path, index=False)

    print(f"  Saved: {csv_name}")

print("✔ All XLSX files converted to CSV (filename preserved).")
