import os
import re
import json
import pandas as pd

consolidated_paths = {
    "618D": "/dataset_618D/consolidated_batch_predictions.json",
    "619D": "/dataset_619D/consolidated_batch_predictions.json"
}
summary_path = "/downloads/FLC Failure Summary.xlsx"

# 1. Load predictions map
predictions_map = set()
for batch, path in consolidated_paths.items():
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for row in data:
                sn_val = row.get("SN")
                if sn_val is not None:
                    try:
                        predictions_map.add((batch, int(sn_val)))
                    except ValueError:
                        pass

# 2. Load summary SNs
df_sum = pd.read_excel(summary_path, sheet_name="FLC Failure Summary")
sns = df_sum["S/N"].dropna().astype(str).tolist()
sns = [s.strip() for s in sns if not ("FAILURES" in s or "failures" in s.upper() or s.startswith("SUMMARY"))]

# 3. Identify missing
missing_units = []
unit_pattern = re.compile(r"^(6\d{2}D)(\d+)$")

for s in sns:
    match = unit_pattern.match(s)
    if match:
        batch_prefix = match.group(1)
        sn_num = int(match.group(2))
        if (batch_prefix, sn_num) not in predictions_map:
            missing_units.append((s, batch_prefix, sn_num))
    else:
        # Check if it has a different format
        missing_units.append((s, None, None))

print(f"Total summary units: {len(sns)}")
print(f"Missing units: {len(missing_units)}")
print("List of missing units:")
for m in missing_units:
    print(f"  {m[0]} (Batch: {m[1]}, SN: {m[2]})")
