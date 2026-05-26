from pathlib import Path
import pandas as pd

csv1 = Path(r"C:\tmp\reid_samurai_trackeval_sweep_full\reid_samurai_sweep_full_trackeval_sweep_20260523_040646.csv")
csv2 = Path(r"C:\tmp\rs_sweep2\rs_trackeval_sweep_20260523_135538.csv")

out_dir = Path(r"C:\tmp\reid_samurai_trackeval_sweep_merged")
out_dir.mkdir(parents=True, exist_ok=True)

out_all = out_dir / "reid_samurai_sweep_merged_all.csv"
out_valid = out_dir / "reid_samurai_sweep_merged_valid.csv"
out_sorted_hota = out_dir / "reid_samurai_sweep_sorted_by_hota.csv"
out_sorted_idf1 = out_dir / "reid_samurai_sweep_sorted_by_idf1.csv"
out_sorted_idsw = out_dir / "reid_samurai_sweep_sorted_by_idsw.csv"

df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)

df = pd.concat([df1, df2], ignore_index=True)

# Keep a full copy, including failed rows, just for record.
df.to_csv(out_all, index=False)

# Keep only successful runs.
df_valid = df[df["returncode"].astype(str) == "0"].copy()

# Remove duplicated config_idx if any accidentally exist.
# If the resumed run contains config 28 and the old CSV contains failed config 28,
# this keeps the successful one.
df_valid = df_valid.sort_values(["config_idx", "returncode"]).drop_duplicates(
    subset=["config_idx"],
    keep="last",
)

# Convert metric columns to numeric.
metric_cols = [
    "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDP", "IDR",
    "IDSW", "CLR_FP", "CLR_FN", "CLR_TP",
]
for col in metric_cols:
    if col in df_valid.columns:
        df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")

# Save valid merged results.
df_valid.to_csv(out_valid, index=False)

# Sorted versions.
df_valid.sort_values(["HOTA", "IDF1", "MOTA"], ascending=[False, False, False]).to_csv(
    out_sorted_hota, index=False
)

df_valid.sort_values(["IDF1", "HOTA", "MOTA"], ascending=[False, False, False]).to_csv(
    out_sorted_idf1, index=False
)

df_valid.sort_values(["IDSW", "HOTA", "IDF1"], ascending=[True, False, False]).to_csv(
    out_sorted_idsw, index=False
)

print(f"Total rows including failed: {len(df)}")
print(f"Valid rows: {len(df_valid)}")
print()
print(f"Saved: {out_all}")
print(f"Saved: {out_valid}")
print(f"Saved: {out_sorted_hota}")
print(f"Saved: {out_sorted_idf1}")
print(f"Saved: {out_sorted_idsw}")

print("\nTop 10 by HOTA:")
cols = [
    "config_idx", "run_name",
    "min_obj_score_logits",
    "memory_bank_reid_threshold",
    "reid_thr",
    "reid_gallery_add_sim_threshold",
    "HOTA", "MOTA", "IDF1", "IDSW",
]
print(df_valid.sort_values(["HOTA", "IDF1", "MOTA"], ascending=[False, False, False])[cols].head(10).to_string(index=False))

print("\nTop 10 by IDF1:")
print(df_valid.sort_values(["IDF1", "HOTA", "MOTA"], ascending=[False, False, False])[cols].head(10).to_string(index=False))

print("\nTop 10 by lowest IDSW:")
print(df_valid.sort_values(["IDSW", "HOTA", "IDF1"], ascending=[True, False, False])[cols].head(10).to_string(index=False))