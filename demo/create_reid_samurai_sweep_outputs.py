from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
CSV_PATH = Path(r"C:\tmp\reid_samurai_trackeval_sweep_merged\reid_samurai_sweep_merged_valid.csv")
OUT_DIR = Path(r"C:\tmp\reid_samurai_trackeval_sweep_merged\thesis_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

metric_cols = [
    "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDP", "IDR",
    "IDSW", "CLR_FP", "CLR_FN", "CLR_TP",
]
param_cols = [
    "min_obj_score_logits",
    "memory_bank_reid_threshold",
    "reid_thr",
    "reid_gallery_add_sim_threshold",
]

for col in metric_cols + param_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["returncode"].astype(str) == "0"].copy()


# ---------------------------------------------------------------------
# Helper formatting
# ---------------------------------------------------------------------
def f3(x):
    return f"{float(x):.3f}"


def f2(x):
    return f"{float(x):.2f}"


def f1(x):
    return f"{float(x):.1f}"


def config_short_name(row):
    return (
        f"$s_{{\\mathrm{{obj}}}}={row['min_obj_score_logits']:.1f}$, "
        f"$\\tau_{{\\mathrm{{mem}}}}={row['memory_bank_reid_threshold']:.2f}$, "
        f"$\\tau_{{\\mathrm{{ReID}}}}={row['reid_thr']:.2f}$, "
        f"$\\tau_{{\\mathrm{{gal}}}}={row['reid_gallery_add_sim_threshold']:.2f}$"
    )


# ---------------------------------------------------------------------
# Selected configs
# ---------------------------------------------------------------------
selected_ids = [73, 27, 78]
selected_roles = {
    73: "Best HOTA/IDF1",
    27: "Lowest IDSW",
    78: "Selected operating point",
}

selected_df = df[df["config_idx"].isin(selected_ids)].copy()
selected_df["Role"] = selected_df["config_idx"].map(selected_roles)

# Order manually for the story
selected_df["order"] = selected_df["config_idx"].map({73: 1, 27: 2, 78: 3})
selected_df = selected_df.sort_values("order")


# ---------------------------------------------------------------------
# Plot: HOTA vs IDSW
# ---------------------------------------------------------------------
plt.figure(figsize=(7.2, 4.8))

plt.scatter(
    df["IDSW"],
    df["HOTA"],
    alpha=0.55,
    s=42,
    label="Sweep configurations",
)

highlight_styles = {
    73: {"marker": "*", "s": 220, "label": "Config 73: best HOTA/IDF1"},
    27: {"marker": "X", "s": 130, "label": "Config 27: lowest IDSW"},
    78: {"marker": "D", "s": 120, "label": "Config 78: selected"},
}

for cfg_id, style in highlight_styles.items():
    row = df[df["config_idx"] == cfg_id].iloc[0]
    plt.scatter(
        [row["IDSW"]],
        [row["HOTA"]],
        marker=style["marker"],
        s=style["s"],
        edgecolors="black",
        linewidths=0.8,
        label=style["label"],
        zorder=5,
    )
    plt.annotate(
        f"{cfg_id}",
        (row["IDSW"], row["HOTA"]),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=9,
    )

plt.xlabel("Identity switches (IDSW) ↓")
plt.ylabel("HOTA ↑")
plt.title("Hyperparameter sweep: tracking accuracy vs. identity stability")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend(fontsize=8)
plt.tight_layout()

plot_png = OUT_DIR / "reid_samurai_sweep_hota_vs_idsw.png"
plot_pdf = OUT_DIR / "reid_samurai_sweep_hota_vs_idsw.pdf"

plt.savefig(plot_png, dpi=300)
plt.savefig(plot_pdf)
plt.close()

print(f"Saved plot PNG: {plot_png}")
print(f"Saved plot PDF: {plot_pdf}")


# ---------------------------------------------------------------------
# Table 1: selected configurations
# ---------------------------------------------------------------------
table1_path = OUT_DIR / "table_sweep_selected_configs.tex"

with table1_path.open("w", encoding="utf-8") as f:
    f.write(r"""\begin{table}[H]
\centering
\caption{Representative configurations from the ReID-SAMURAI hyperparameter sweep. The selected operating point trades a small decrease in HOTA and IDF1 for fewer identity switches.}
\label{tab:reid_samurai_sweep_selected_configs}
\renewcommand{\arraystretch}{1.12}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccccc}
\toprule
Role & Config. & $s_{\mathrm{obj}}$ & $\tau_{\mathrm{mem}}$ & $\tau_{\mathrm{ReID}}$ & $\tau_{\mathrm{gal}}$ & HOTA $\uparrow$ & MOTA $\uparrow$ & IDF1 $\uparrow$ & IDSW $\downarrow$ \\
\midrule
""")

    for _, row in selected_df.iterrows():
        role = row["Role"]
        cfg = int(row["config_idx"])
        f.write(
            f"{role} & {cfg} & "
            f"{f1(row['min_obj_score_logits'])} & "
            f"{f2(row['memory_bank_reid_threshold'])} & "
            f"{f2(row['reid_thr'])} & "
            f"{f2(row['reid_gallery_add_sim_threshold'])} & "
            f"{f3(row['HOTA'])} & "
            f"{f3(row['MOTA'])} & "
            f"{f3(row['IDF1'])} & "
            f"{int(row['IDSW'])} \\\\\n"
        )

    f.write(r"""\bottomrule
\end{tabular}%
}
\end{table}
""")

print(f"Saved LaTeX table 1: {table1_path}")


# ---------------------------------------------------------------------
# Table 2: average parameter trends
# ---------------------------------------------------------------------
param_labels = {
    "min_obj_score_logits": r"$s_{\mathrm{obj}}$",
    "memory_bank_reid_threshold": r"$\tau_{\mathrm{mem}}$",
    "reid_thr": r"$\tau_{\mathrm{ReID}}$",
    "reid_gallery_add_sim_threshold": r"$\tau_{\mathrm{gal}}$",
}

trend_rows = []

for param in param_cols:
    g = (
        df.groupby(param)[["HOTA", "MOTA", "IDF1", "IDSW"]]
        .mean()
        .reset_index()
        .sort_values(param)
    )

    for _, row in g.iterrows():
        trend_rows.append(
            {
                "Parameter": param_labels[param],
                "Value": row[param],
                "HOTA": row["HOTA"],
                "MOTA": row["MOTA"],
                "IDF1": row["IDF1"],
                "IDSW": row["IDSW"],
            }
        )

trend_df = pd.DataFrame(trend_rows)

table2_path = OUT_DIR / "table_sweep_parameter_trends.tex"

with table2_path.open("w", encoding="utf-8") as f:
    f.write(r"""\begin{table}[H]
\centering
\caption{Average effect of each swept hyperparameter value across the ReID-SAMURAI sweep. Values are averaged over all configurations containing the corresponding parameter value.}
\label{tab:reid_samurai_sweep_parameter_trends}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{llcccc}
\toprule
Parameter & Value & HOTA $\uparrow$ & MOTA $\uparrow$ & IDF1 $\uparrow$ & IDSW $\downarrow$ \\
\midrule
""")

    last_param = None
    for _, row in trend_df.iterrows():
        param = row["Parameter"]
        param_text = param if param != last_param else ""
        f.write(
            f"{param_text} & "
            f"{f2(row['Value'])} & "
            f"{f3(row['HOTA'])} & "
            f"{f3(row['MOTA'])} & "
            f"{f3(row['IDF1'])} & "
            f"{f3(row['IDSW'])} \\\\\n"
        )
        last_param = param

    f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")

print(f"Saved LaTeX table 2: {table2_path}")


# ---------------------------------------------------------------------
# Also save compact CSVs for checking
# ---------------------------------------------------------------------
selected_df.to_csv(OUT_DIR / "selected_configs.csv", index=False)
trend_df.to_csv(OUT_DIR / "parameter_trends.csv", index=False)

print(f"Saved selected configs CSV: {OUT_DIR / 'selected_configs.csv'}")
print(f"Saved parameter trends CSV: {OUT_DIR / 'parameter_trends.csv'}")

print("\nSelected configurations:")
print(
    selected_df[
        [
            "Role",
            "config_idx",
            "min_obj_score_logits",
            "memory_bank_reid_threshold",
            "reid_thr",
            "reid_gallery_add_sim_threshold",
            "HOTA",
            "MOTA",
            "IDF1",
            "IDSW",
        ]
    ].to_string(index=False)
)

print("\nDone.")