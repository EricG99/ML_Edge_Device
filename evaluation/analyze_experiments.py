
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze experiment outputs for a fixed (or given) algorithm/profile.
Defaults: algorithm=random_forest, profile=edge.

It scans typical Output folders for:
- ErrorMetrics_all_runs.csv (to discover runs and/or JSON paths)
- ErrorMetrics_*.json       (per-run metadata: run_id, lags, horizon, model info)
- StepPredictions_*.csv     (per-run step metrics)

Then it computes per-(lags,horizon) means of inference_time_s, total_time_s,
cpu_percent, ram_percent - skipping the first step per run - and writes:
- Experiment_Aggregated_Summary.csv
- charts/heatmap_inference_time.png
- charts/heatmap_cpu.png
- charts/heatmap_ram.png

Usage examples:
  python analyze_experiments.py
  python analyze_experiments.py --algorithm random_forest --profile edge --roots "/path/to/Output,/another/root" --out "/path/to/save"
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ utilities ------------------------
def find_files(patterns: List[str], roots: List[Path]) -> List[Path]:
    hits = []
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            hits.extend(root.rglob(pat))
    # de-duplicate while preserving order
    uniq = []
    seen = set()
    for p in hits:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq

def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def read_json(fp: Path) -> Optional[dict]:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def step_csv_skip_first(df: pd.DataFrame) -> pd.DataFrame:
    """
    Skip the first step per 'session' if we can detect a step/session-like column.
    Otherwise, drop the first row of the file.
    """
    # Potential id columns
    import re as _re
    cand_cols = [c for c in df.columns if _re.search(r"(step|session|run).*id", c, flags=_re.I)]
    if cand_cols:
        cid = cand_cols[0]
        # drop first row per group (by index order)
        return df.groupby(cid, as_index=False, group_keys=False).apply(lambda g: g.iloc[1:])
    else:
        return df.iloc[1:, :]

def summarize_step_predictions(step_csv: Path) -> Dict[str, float]:
    df = pd.read_csv(step_csv)
    df = step_csv_skip_first(df)
    out = {}
    for col in ["inference_time_s", "total_time_s", "cpu_percent", "ram_percent"]:
        if col in df.columns:
            out[f"{col}_mean"] = float(pd.to_numeric(df[col], errors="coerce").dropna().mean())
        else:
            out[f"{col}_mean"] = np.nan
    out["n_steps_used"] = int(len(df))
    return out

def pivot_and_plot(df: pd.DataFrame, value_col: str, title: str, outfile: Path):
    """
    Create a simple heatmap: rows = lags, cols = horizon, values = metric
    """
    if not {"lags","horizon", value_col}.issubset(df.columns):
        return None
    p = df.pivot_table(index="lags", columns="horizon", values=value_col, aggfunc="mean")
    if p.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(p.values, aspect="auto")
    ax.set_xticks(np.arange(len(p.columns)))
    ax.set_xticklabels(p.columns)
    ax.set_yticks(np.arange(len(p.index)))
    ax.set_yticklabels(p.index)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Lags")
    ax.set_title(title)
    # value annotations
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            val = p.values[i, j]
            if not (isinstance(val, float) and np.isnan(val)):
                ax.text(j, i, f"{val:.3g}", ha="center", va="center")
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outfile

# ------------------------ core ------------------------
def evaluate_experiments(
    search_roots: List[Path],
    algorithm: str = "random_forest",
    profile: str = "edge",
    out_dir: Path = Path(".")
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """
    Collect runs for (algorithm, profile), compute summary metrics, and produce heatmaps.
    Robust to partial inputs: it will use any of the following, if present:
      - ErrorMetrics_all_runs.csv (to discover run JSONs)
      - ErrorMetrics_*.json (direct)
      - StepPredictions_*.csv (fallback if JSON points to non-local paths)
    """
    algo_key = algorithm.lower()
    profile_key = profile.lower()

    # discover aggregated CSVs and JSONs
    agg_csv_candidates = find_files(["ErrorMetrics_all_runs.csv"], search_roots)
    json_candidates = find_files([
        "ErrorMetrics_*_train_*_*.json",
        "ErrorMetrics_*_infer_*_*.json",
        "ErrorMetrics_*.json"
    ], search_roots)
    step_csv_candidates = find_files(["StepPredictions_*.csv"], search_roots)

    # Build initial run index from aggregated CSV if we have one
    run_rows = []
    if agg_csv_candidates:
        try:
            df_all = pd.read_csv(agg_csv_candidates[0])
            # Heuristics to filter for algorithm/profile; fall back to include all
            f = pd.Series([True] * len(df_all))
            for c, val in (("algorithm", algo_key), ("profile", profile_key), ("model_name", f"{algo_key}_{profile_key}")):
                if c in df_all.columns:
                    f = f & df_all[c].astype(str).str.lower().str.contains(val, na=False)
            run_rows = df_all[f.fillna(False)].to_dict(orient="records")
        except Exception:
            pass

    # If no rows from aggregate, try to parse JSONs directly
    if not run_rows:
        for j in json_candidates:
            js = read_json(j)
            if not js:
                continue
            run_rows.append({"json_path": str(j)})

    # Normalize: ensure we have json_path for each row
    for r in run_rows:
        if "json_path" not in r:
            if "json" in r:
                r["json_path"] = r["json"]

    # Extract metrics per run
    records = []
    for row in run_rows:
        js = None
        if "json_path" in row and isinstance(row["json_path"], str):
            p = Path(row["json_path"])
            if not p.exists():
                # Try to locate a file with same name in known roots
                candidates = find_files([p.name], search_roots)
                p = candidates[0] if candidates else p
            js = read_json(p) if p.exists() else None

        if not js:
            continue

        # Identify algorithm/profile
        tr = js.get("training_config", {}) or {}
        inf = js.get("inference_config", {}) or {}
        run_id = js.get("run", {}).get("run_id") or tr.get("run_id") or inf.get("run_id") or ""

        # Try to match algorithm via model_name or output path
        model_name = (tr.get("model_name") or inf.get("model_name") or "").lower()
        is_algo_match = (algo_key in model_name) or (algorithm.replace("_"," ").title().replace(" ", "_") in (tr.get("paths", {}) or {}).get("Base_Output_Path",""))
        if not is_algo_match and algo_key not in (tr.get("model_name","") + inf.get("model_name","")).lower():
            continue

        # We keep profile filtering light; many JSONs may not encode it explicitly
        is_profile_ok = True
        if profile_key == "edge":
            is_profile_ok = True  # avoid over-filtering

        if not is_profile_ok:
            continue

        lags = tr.get("lags", inf.get("lags", np.nan))
        horizon = tr.get("horizon", inf.get("horizon", np.nan))
        model_size = tr.get("model_size_MB", inf.get("model_size_MB", np.nan))
        training_time = tr.get("training_time_s", np.nan)
        model_tag = (js.get("extra_info", {}) or {}).get("model_tag", "")

        # Find the StepPredictions CSV
        pred_path = (js.get("extra_info", {}) or {}).get("predictions_file_path", "")
        pred = Path(pred_path) if pred_path else None
        if not pred or not pred.exists():
            if run_id:
                pat = f"StepPredictions_{run_id}_*.csv"
                hits = find_files([pat], search_roots)
                pred = hits[0] if hits else None
        if not pred or not pred.exists():
            pred = step_csv_candidates[0] if step_csv_candidates else None

        metrics = {
            "inference_time_s_mean": np.nan,
            "total_time_s_mean": np.nan,
            "cpu_percent_mean": np.nan,
            "ram_percent_mean": np.nan,
            "n_steps_used": 0,
        }
        if pred and pred.exists():
            try:
                m = summarize_step_predictions(pred)
                metrics.update(m)
            except Exception:
                pass

        rec = {
            "algorithm": algorithm,
            "profile": profile,
            "lags": int(lags) if not pd.isna(lags) else np.nan,
            "horizon": int(horizon) if not pd.isna(horizon) else np.nan,
            "model_variant": model_tag or "unknown",
            "model_size_mb": safe_float(model_size),
            "training_time_s": safe_float(training_time),
            "run_id": run_id,
            "step_csv": str(pred) if pred else "",
        }
        rec.update(metrics)
        records.append(rec)

    # Assemble dataframe
    if not records:
        df = pd.DataFrame(columns=[
            "algorithm","profile","lags","horizon","model_variant",
            "inference_time_s_mean","total_time_s_mean","cpu_percent_mean","ram_percent_mean",
            "model_size_mb","training_time_s","run_id","step_csv","n_steps_used"
        ])
    else:
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=["run_id","lags","horizon","model_variant"])
        df = df.sort_values(["lags","horizon"])

    # Save summary
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "Experiment_Aggregated_Summary.csv"
    df.to_csv(summary_csv, index=False)

    # Heatmaps
    outputs = {"summary_csv": summary_csv}
    if len(df) > 0:
        charts_dir = out_dir / "charts"
        charts_dir.mkdir(exist_ok=True, parents=True)
        f1 = pivot_and_plot(df, "inference_time_s_mean", "Ø Inference Time (s)", charts_dir / "heatmap_inference_time.png")
        f2 = pivot_and_plot(df, "cpu_percent_mean", "Ø CPU (%)", charts_dir / "heatmap_cpu.png")
        f3 = pivot_and_plot(df, "ram_percent_mean", "Ø RAM (%)", charts_dir / "heatmap_ram.png")
        if f1: outputs["heatmap_inference_time"] = f1
        if f2: outputs["heatmap_cpu"] = f2
        if f3: outputs["heatmap_ram"] = f3

    print(f"[OK] Wrote summary to: {summary_csv}")
    for k,v in outputs.items():
        if k != "summary_csv":
            print(f"[OK] Wrote: {v}")
    return df, outputs

# ------------------------ cli ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze experiment outputs for random_forest edge runs (or as specified).")
    parser.add_argument("--algorithm", default="random_forest", help="Algorithm name (default: random_forest)")
    parser.add_argument("--profile", default="edge", help="Profile name (default: edge)")
    parser.add_argument("--roots", default=".", help="Comma-separated list of root directories to search (default: current directory)")
    parser.add_argument("--out", default=".", help="Output directory to write summary and charts (default: current directory)")
    return parser.parse_args()

def main():
    args = parse_args()
    roots = [Path(p.strip()) for p in args.roots.split(",") if p.strip()]
    out_dir = Path(args.out)
    evaluate_experiments(roots, algorithm=args.algorithm, profile=args.profile, out_dir=out_dir)

if __name__ == "__main__":
    main()
