import argparse
from pathlib import Path


METRICS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]
HIGHER_IS_BETTER = {
    "HOTA": True,
    "DetA": True,
    "AssA": True,
    "MOTA": True,
    "IDF1": True,
    "IDSW": False,
}


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def load_summary(summary_path: Path) -> dict:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing TrackEval summary file: {summary_path}")

    lines = [line.strip() for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Could not parse summary file: {summary_path}")

    header = lines[0].split()
    values = lines[1].split()

    if len(header) != len(values):
        raise RuntimeError(
            f"Header/value length mismatch in {summary_path}\n"
            f"Header has {len(header)} columns, values has {len(values)} columns."
        )

    data = {}
    for key, value in zip(header, values):
        try:
            data[key] = float(value)
        except ValueError:
            data[key] = value

    return data


def fmt_metric(metric: str, value: float, is_best: bool) -> str:
    if metric == "IDSW":
        text = str(int(round(value)))
    else:
        text = f"{value:.2f}"

    if is_best:
        return rf"\best{{{text}}}"
    return text


def parse_system_arg(system_arg: str) -> dict:
    # Format:
    # "System name|Checkpoint name|tracker_name|masks"
    parts = system_arg.split("|")
    if len(parts) != 4:
        raise ValueError(
            "Each --system must have format: "
            '"System name|Checkpoint name|trackeval_tracker_name|masks"'
        )

    system_name, checkpoint_name, tracker_name, masks = parts
    masks_bool = masks.strip() in {"1", "true", "True", "yes", "Yes", "cmark"}

    return {
        "system": system_name.strip(),
        "checkpoint": checkpoint_name.strip(),
        "tracker": tracker_name.strip(),
        "masks": masks_bool,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trackeval_root", required=True)
    parser.add_argument("--benchmark", default="KTP-5Hz")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        help='Format: "System name|Checkpoint name|trackeval_tracker_name|masks"',
    )
    parser.add_argument("--out_tex", required=True)
    args = parser.parse_args()

    trackeval_root = Path(args.trackeval_root)
    tracker_base = (
        trackeval_root
        / "data"
        / "trackers"
        / "mot_challenge"
        / f"{args.benchmark}-{args.split}"
    )

    rows = []
    for system_arg in args.system:
        row = parse_system_arg(system_arg)
        summary_path = tracker_base / row["tracker"] / "pedestrian_summary.txt"

        print(f"[load] {row['tracker']}")
        print(f"       {summary_path}")

        metrics = load_summary(summary_path)
        row["metrics"] = {m: float(metrics[m]) for m in METRICS}
        rows.append(row)

    # Find best values.
    best_values = {}
    for metric in METRICS:
        vals = [row["metrics"][metric] for row in rows]
        if HIGHER_IS_BETTER[metric]:
            best_values[metric] = max(vals)
        else:
            best_values[metric] = min(vals)

    latex = []
    latex.append(r"\begin{table}[H]")
    latex.append(r"\centering")
    latex.append(
        r"\caption{Effect of the SAMURAI checkpoint size on the proposed ReID-SAMURAI system. "
        r"Higher values are better for HOTA, DetA, AssA, MOTA, and IDF1, while lower values are better for IDSW. "
        r"The best result for each metric is highlighted in blue.}"
    )
    latex.append(r"\label{tab:reid_samurai_checkpoint_comparison}")
    latex.append(r"\renewcommand{\arraystretch}{1.10}")
    latex.append(r"\resizebox{\textwidth}{!}{%")
    latex.append(r"\begin{tabular}{llccccccc}")
    latex.append(r"\toprule")
    latex.append(
        r"System & Checkpoint & Masks & HOTA $\uparrow$ & DetA $\uparrow$ & AssA $\uparrow$ & "
        r"MOTA $\uparrow$ & IDF1 $\uparrow$ & IDSW $\downarrow$ \\"
    )
    latex.append(r"\midrule")

    previous_system = None
    for i, row in enumerate(rows):
        system = latex_escape(row["system"])
        checkpoint = latex_escape(row["checkpoint"])
        masks = r"\cmark" if row["masks"] else r"\xmark"

        if previous_system is not None and row["system"] != previous_system:
            latex.append(r"\midrule")

        metric_cells = []
        for metric in METRICS:
            value = row["metrics"][metric]
            is_best = abs(value - best_values[metric]) < 1e-9
            metric_cells.append(fmt_metric(metric, value, is_best))

        latex.append(
            f"{system} & {checkpoint} & {masks} & "
            + " & ".join(metric_cells)
            + r" \\"
        )

        previous_system = row["system"]

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}%")
    latex.append(r"}")
    latex.append(r"\end{table}")

    out_path = Path(args.out_tex)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(latex), encoding="utf-8")

    print(f"\n[done] wrote LaTeX table to:")
    print(out_path)


if __name__ == "__main__":
    main()