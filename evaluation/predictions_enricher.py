
"""
predictions_enricher.py

Enrich an experiment summary with:
- Series (true + pred_h1..hN) per run
- MAE / R^2 per horizon (1..N)
- Aggregations across horizons (avg & pooled)
- Optional: attach Error_Metrics JSON fields (training_time_s, metrics.mse/r2, model_sizes_mb for the matching quantization)

Designed to be imported and used inside a notebook (or as a script).
"""

from __future__ import annotations
import os, glob, re, math, json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# -------------------------- Pfad-Resolver --------------------------------
FOLDER_BY_ALGO = {
    "cnn1d": "CNN1D",
    "lstm": "LSTM",
    "random_forest": "Random_Forest",
    "light_xgboost": "Light_XGBoost",
    "xgboost": "XGBoost",
}

def variant_tokens(model_variant: str) -> List[str]:
    mv = (model_variant or "").lower()
    toks = set()
    if not mv:
        return []
    toks |= {mv, mv.replace(".", "_"), os.path.splitext(mv)[0], os.path.basename(mv)}
    if mv.endswith(".keras") or mv == "model.keras" or "keras" in mv:
        toks |= {"keras", "model.keras", "model_keras"}
    if "int8" in mv:
        toks |= {
            "int8",
            "quant_int8",
            "model_quant_int8",
            "model_quant_int8.tflite",
            "tflite",
            "tflite_int8",
            "tflite_model_quant_int8",
        }
    if "float16" in mv or "fp16" in mv or "quant16" in mv:
        toks |= {
            "float16",
            "fp16",
            "quant_float16",
            "model_quant_float16",
            "model_quant_float16.tflite",
            "tflite",
            "tflite_float16",
            "tflite_model_quant_float16",
        }
    if mv.endswith(".joblib") or "joblib" in mv or "sklearn" in mv:
        toks |= {"joblib", "sklearn"}
    return sorted(toks)

def _algo_folder_name(algo: str) -> str:
    algo_l = (algo or "").lower()
    return FOLDER_BY_ALGO.get(algo_l, algo_l.upper())

def resolve_pred_csv_path(row: pd.Series, base: Path) -> Optional[str]:
    """
    Try to resolve the predictions CSV path for a given run row.
    """
    algo = str(row.get("algorithm", "")).lower()
    folder = _algo_folder_name(algo)
    run_id = str(row.get("run_id") or "")
    if not run_id:
        return None

    pred_dir = base / folder / run_id / "Prediction_Data"
    if not pred_dir.exists():
        # wider search as fallback
        pattern = str(base / "**" / run_id / "Prediction_Data")
        cands = [Path(p) for p in glob.glob(pattern, recursive=True) if os.path.isdir(p)]
        if cands:
            pred_dir = cands[0]
        else:
            return None

    ds = str(row.get("dataset", "mqtt_data_filtered.csv"))
    ds_name = os.path.splitext(os.path.basename(ds))[0].lower()
    mv_tokens = variant_tokens(str(row.get("model_variant", "")))
    algo_label = _algo_folder_name(algo)

    # Best guess file name attempts
    suf_candidates = mv_tokens or [
        "keras",
        "tflite_model_quant_int8",
        "tflite_model_quant_float16",
        "joblib",
    ]
    for suf in suf_candidates:
        name = f"StepPredictions_{run_id}_{algo_label}_{ds_name}__{suf}.csv"
        p = pred_dir / name
        if p.is_file():
            return str(p)

    # Fallback: score candidates
    cand = glob.glob(str(pred_dir / f"StepPredictions_*{run_id}*{ds_name}*.csv")) or glob.glob(
        str(pred_dir / "StepPredictions_*.csv")
    )
    if not cand:
        return None

    def score_for(path: str) -> Tuple[int, int]:
        name = os.path.basename(path).lower()
        s = 0
        if run_id.lower() in name:
            s += 100
        if algo_label.lower() in name:
            s += 30
        if ds_name in name:
            s += 20
        for t in mv_tokens:
            if t in name:
                s += 5
        return (s, -len(name))

    cand.sort(key=lambda p: score_for(p), reverse=True)
    return cand[0]

# ---------------------------- Hilfsfunktionen -----------------------------
def _safe_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[m], y_pred[m]

def mae(y, yhat):
    y, yhat = _safe_arrays(y, yhat)
    return float(np.mean(np.abs(yhat - y))) if y.size else float("nan")

def r2(y, yhat):
    y, yhat = _safe_arrays(y, yhat)
    if y.size == 0:
        return float("nan")
    ss_res = np.sum((yhat - y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def _fmt(x):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return ""
        return f"{float(x):.6g}"
    except Exception:
        return str(x)

def _series_to_tuple_string(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce")
    vals = [_fmt(v) for v in s if pd.notna(v)]
    return "(" + ",".join(vals) + ")" if vals else ""

def _to_numeric_series(v) -> pd.Series:
    if v is None:
        return pd.Series([], dtype=float)
    # scalar
    if isinstance(v, (int, float, str)):
        try:
            return pd.Series([pd.to_numeric(v, errors="coerce")], dtype=float)
        except Exception:
            return pd.Series([], dtype=float)
    # list/tuple/ndarray
    if isinstance(v, (list, tuple, set)):
        return pd.to_numeric(pd.Series(list(v)), errors="coerce")
    # numpy
    try:
        import numpy as _np
        if isinstance(v, _np.ndarray):
            return pd.to_numeric(pd.Series(v.tolist()), errors="coerce")
    except Exception:
        pass
    # dict: take values in key order if keys look sortable; else just values()
    if isinstance(v, dict):
        try:
            # keep natural order if keys like 'h1','h2', or ints
            def _key_k(k):
                if isinstance(k, int):
                    return k
                try:
                    return int(str(k).lstrip("h").lstrip("_h"))
                except Exception:
                    return str(k)
            items = sorted(v.items(), key=lambda kv: _key_k(kv[0]))
            vals = [kv[1] for kv in items]
        except Exception:
            vals = list(v.values())
        return pd.to_numeric(pd.Series(vals), errors="coerce")
    # fallback
    return pd.Series([], dtype=float)

def _metric_scalar_and_series(metrics: dict, key: str) -> tuple[float, str]:
    v = None
    if isinstance(metrics, dict):
        v = metrics.get(key)
        # sometimes nested under 'overall' or 'avg'
        if v is None and "overall" in metrics and isinstance(metrics["overall"], dict):
            v = metrics["overall"].get(key)
        if v is None and "avg" in metrics and isinstance(metrics["avg"], dict):
            v = metrics["avg"].get(key)
        if v is None and "mean" in metrics and isinstance(metrics["mean"], dict):
            v = metrics["mean"].get(key)

    s = _to_numeric_series(v)
    s = s[pd.notna(s) & np.isfinite(s)]
    if s.empty:
        return float("nan"), ""
    return float(s.mean()), _series_to_tuple_string(s)

    s = pd.to_numeric(s, errors="coerce")
    vals = [_fmt(v) for v in s if pd.notna(v)]
    return "(" + ",".join(vals) + ")" if vals else ""

# ------------------- Ausrichtung (Horizon) --------------------------------
# "t_plus_h":           pred_hh(t)  vs true_value(t+h)   => shift(-h)
# "t_plus_h_minus_1":   pred_hh(t)  vs true_value(t+h-1) => shift(-(h-1))
def _true_for_h(series_true: pd.Series, h: int, shift_mode: str) -> pd.Series:
    return series_true.shift(-(h - 1)) if shift_mode == "t_plus_h_minus_1" else series_true.shift(-h)

# ---------------- Error_Metrics JSON Handling -----------------------------
def _model_tag_from_tokens(tokens: List[str]) -> Optional[str]:
    if any(t in tokens for t in ["int8", "quant_int8", "tflite_int8", "tflite_model_quant_int8"]):
        return "tflite_model_quant_int8"
    if any(t in tokens for t in ["float16", "fp16", "tflite_float16", "tflite_model_quant_float16"]):
        return "tflite_model_quant_float16"
    if any(t in tokens for t in ["keras", "model.keras", "model_keras"]):
        return "keras"
    return None

def _select_size_mb(model_sizes: Dict[str, Any], model_tag: str) -> Optional[float]:
    if model_sizes is None:
        return None
    key_map = {
        "keras": ["model.keras", "keras"],
        "tflite_model_quant_float16": ["model_quant_float16.tflite", "tflite_model_quant_float16"],
        "tflite_model_quant_int8": ["model_quant_int8.tflite", "tflite_model_quant_int8"],
    }
    for k in key_map.get(model_tag, []):
        if k in model_sizes and model_sizes[k] is not None:
            try:
                return float(model_sizes[k])
            except Exception:
                pass
    return None

def _load_error_metrics_jsons(base: Path, run_id: str) -> List[Dict[str, Any]]:
    """
    Scan for ErrorMetrics JSONs for a given run_id under base/**/Error_Metrics/
    """
    pattern = str(base / "**" / "Error_Metrics" / f"ErrorMetrics*{run_id}*.json")
    files = glob.glob(pattern, recursive=True)
    res = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_json_path"] = fp
            res.append(data)
        except Exception:
            continue
    return res

def _match_json_for_variant(json_list: List[Dict[str, Any]], model_tag: str) -> Optional[Dict[str, Any]]:
    if not json_list:
        return None
    # prefer exact match by extra_info.model_tag
    for d in json_list:
        tag = (d.get("extra_info", {}) or {}).get("model_tag")
        if tag == model_tag:
            return d
    # alternatively, infer by filename suffix
    for d in json_list:
        path = d.get("_json_path", "").lower()
        if path.endswith(f"__{model_tag}.json"):
            return d
    # fallback: single JSON or first
    if len(json_list) == 1:
        return json_list[0]
    return json_list[0]

def attach_error_metrics_fields(out: Dict[str, Any], base: Path, row: pd.Series) -> None:
    """
    Mutates `out` in place by adding:
      - err_model_tag
      - err_model_size_mb
      - err_training_time_s
      - err_mse
      - err_r2
    if a matching ErrorMetrics JSON can be found.
    """
    run_id = str(row.get("run_id") or "")
    if not run_id:
        return
    mv_tokens = variant_tokens(str(row.get("model_variant", "")))
    model_tag = _model_tag_from_tokens(mv_tokens)
    json_list = _load_error_metrics_jsons(base, run_id)
    if not json_list:
        return
    chosen = _match_json_for_variant(json_list, model_tag or "")
    if not isinstance(chosen, dict):
        return

    # read size for the relevant file
    model_sizes = (chosen.get("run", {}) or {}).get("model_sizes_mb")
    size_mb = _select_size_mb(model_sizes or {}, model_tag or "")
    training_time_s = (chosen.get("run", {}) or {}).get("training_time_s")

    metrics = chosen.get("metrics") or chosen.get("run", {}).get("metrics") or {}
    mse_scalar, mse_series = _metric_scalar_and_series(metrics, "mse")
    r2_scalar, r2_series = _metric_scalar_and_series(metrics, "r2")

    out["err_model_tag"] = model_tag
    out["err_model_size_mb"] = float(size_mb) if size_mb is not None else float("nan")
    # training_time_s could be scalar or list (take mean if list)
    try:
        tt_series = _to_numeric_series(training_time_s)
        out["err_training_time_s"] = float(tt_series.mean()) if not tt_series.empty else float("nan")
    except Exception:
        out["err_training_time_s"] = float("nan")
    out["err_mse"] = mse_scalar
    out["err_mse_series"] = mse_series
    out["err_r2"] = r2_scalar
    out["err_r2_series"] = r2_series
    out["err_json_path"] = chosen.get("_json_path")

# ------------------- Kern: eine Zeile (ein Run) verarbeiten ----------------
def process_run_for_N(row: pd.Series, N: int, base: Path, shift_mode: str, include_error_json: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        # Ziel-H (N):
        "mae_from_preds": float("nan"),
        "r2_from_preds": float("nan"),
        # Aggregationen über h=1..N:
        "mae_all_avg_from_preds": float("nan"),
        "mae_all_pooled_from_preds": float("nan"),
        "r2_all_avg_from_preds": float("nan"),
        "r2_all_pooled_from_preds": float("nan"),
        # Serien:
        "true_value_series": "",
    }
    p = resolve_pred_csv_path(row, base)
    if not p:
        if include_error_json:
            attach_error_metrics_fields(out, base, row)
        return out

    dfp = pd.read_csv(p)
    if "date" in dfp.columns:
        try:
            dfp = dfp.sort_values("date")
        except Exception:
            pass

    # Serien: true
    if "true_value" in dfp.columns:
        out["true_value_series"] = _series_to_tuple_string(dfp["true_value"])

    # Welche pred_hk gibt es?
    avail = sorted(int(m.group(1)) for c in dfp.columns if (m := re.match(r"pred_h(\d+)$", c)))
    if not avail:
        if include_error_json:
            attach_error_metrics_fields(out, base, row)
        return out

    # Wir nutzen GENAU die Horizons 1..N, sofern vorhanden
    used_horizons = [h for h in range(1, int(N) + 1) if f"pred_h{h}" in dfp.columns]
    if not used_horizons:
        if include_error_json:
            attach_error_metrics_fields(out, base, row)
        return out

    y_true_all = pd.to_numeric(dfp.get("true_value"), errors="coerce")

    # Container für "pooled" (über alle Horizons gestapelt)
    pooled_true: List[np.ndarray] = []
    pooled_pred: List[np.ndarray] = []

    # Serien & MAE/R² je Horizon (nur h in 1..N)
    mae_vals: List[float] = []
    r2_vals: List[float] = []

    for h in used_horizons:
        col = f"pred_h{h}"
        # Serien speichern
        out[f"pred_h{h}_series"] = _series_to_tuple_string(dfp[col])

        # MAE/R² für h
        y_pred_h = pd.to_numeric(dfp[col], errors="coerce")
        y_true_h = pd.to_numeric(_true_for_h(y_true_all, h, shift_mode), errors="coerce")
        mask = y_pred_h.notna() & y_true_h.notna()
        y = y_true_h[mask].to_numpy()
        yhat = y_pred_h[mask].to_numpy()

        mae_h = mae(y, yhat)
        r2_h = r2(y, yhat)

        out[f"mae_h{h}_from_preds"] = mae_h
        out[f"r2_h{h}_from_preds"] = r2_h

        if np.isfinite(mae_h):
            mae_vals.append(mae_h)
        if np.isfinite(r2_h):
            r2_vals.append(r2_h)

        # für pooled
        if y.size:
            pooled_true.append(y)
            pooled_pred.append(yhat)

    # Durchschnitt (Alle Horizons gleich gewichtet)
    if mae_vals:
        out["mae_all_avg_from_preds"] = float(np.mean(mae_vals))
    if r2_vals:
        out["r2_all_avg_from_preds"] = float(np.mean(r2_vals))

    # Gepoolt (alle (y,ŷ) über h=1..N stapeln)
    if pooled_true and pooled_pred:
        y_pool = np.concatenate(pooled_true, axis=0)
        yhat_pool = np.concatenate(pooled_pred, axis=0)
        out["mae_all_pooled_from_preds"] = mae(y_pool, yhat_pool)
        out["r2_all_pooled_from_preds"] = r2(y_pool, yhat_pool)

    # MAE & R² für Ziel-Horizon N (nur, wenn pred_hN existiert)
    target_col = f"pred_h{int(N)}"
    if target_col in dfp.columns:
        y_pred_N = pd.to_numeric(dfp[target_col], errors="coerce")
        y_true_N = pd.to_numeric(_true_for_h(y_true_all, int(N), shift_mode), errors="coerce")
        maskN = y_pred_N.notna() & y_true_N.notna()
        yN = y_true_N[maskN].to_numpy()
        yhatN = y_pred_N[maskN].to_numpy()
        out["mae_from_preds"] = mae(yN, yhatN)
        out["r2_from_preds"] = r2(yN, yhatN)

    if include_error_json:
        attach_error_metrics_fields(out, base, row)

    return out

# --------------------- Enrichment Pipeline --------------------------------
def enrich_summary(
    base: str | Path,
    summary_file: str,
    out_file: Optional[str] = None,
    shift_mode: str = "t_plus_h",
    drop_old_metrics: bool = True,
    include_error_json: bool = True,
) -> pd.DataFrame:
    """
    Reads summary_file from `base`, enriches it, and writes CSV to:
      base/Analysis/<out_file or 'Experiment_Enriched_SingleMetricKachel.csv'>

    Returns the enriched DataFrame.
    """
    base = Path(base)
    summary_path = base / summary_file
    df = pd.read_csv(summary_path)

    # Horizon robust laden
    H_series = pd.to_numeric(df.get("horizon_num", df.get("horizon")), errors="coerce")

    rows: List[Dict[str, Any]] = []
    missing_preds = 0
    for i, row in df.iterrows():
        H = H_series.iat[i] if i < len(H_series) else None
        N = int(H) if (pd.notna(H) and float(H).is_integer()) else 1
        res = process_run_for_N(row, N, base, shift_mode, include_error_json)
        if np.isnan(res.get("mae_from_preds", np.nan)) and res.get("true_value_series", "") == "":
            missing_preds += 1
        rows.append(res)

    metrics_df = pd.DataFrame(rows)

    # Zusammenführen
    df_out = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)

    # Optional: alte metrics_value.*-Spalten entfernen
    if drop_old_metrics:
        drop_patterns = [r"^metrics_value\.", r"^metrics_mean\.", r"^metrics_h\d+\.", r"^metrics\."]
        to_drop = [c for c in df_out.columns if any(re.match(pat, c) for pat in drop_patterns)]
        if to_drop:
            df_out = df_out.drop(columns=sorted(set(to_drop)))

    # Speichern
    if out_file is None:
        out_file = "Experiment_Enriched_SingleMetricKachel.csv"
    out_path = base / "Analysis" / out_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8")

    # Info print (for script usage)
    print(
        f"OK: Serien (true + pred_h1..hN), MAE/R² je h<=N, Aggregationen (avg/pooled) + MAE/R²@N + ErrorMetrics(JSON) → {out_path}"
    )
    if missing_preds:
        print(f"Warnung: {missing_preds} Zeilen ohne auffindbare Predictions-CSV.")

    return df_out

# --------------------- CLI entrypoint (optional) --------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enrich experiment summary with predictions and Error_Metrics JSONs.")
    parser.add_argument("--base", required=True, help="Base output directory")
    parser.add_argument("--summary", required=True, help="Summary CSV filename relative to base")
    parser.add_argument("--out", default=None, help="Output CSV filename (written under base/Analysis/)")
    parser.add_argument("--shift", default="t_plus_h", choices=["t_plus_h", "t_plus_h_minus_1"], help="Horizon alignment")
    parser.add_argument("--no-json", action="store_true", help="Do not include Error_Metrics JSON fields")
    parser.add_argument("--keep-old", action="store_true", help="Keep old metrics_* columns from summary")

    args = parser.parse_args()
    enrich_summary(
        base=args.base,
        summary_file=args.summary,
        out_file=args.out,
        shift_mode=args.shift,
        drop_old_metrics=not args.keep_old,
        include_error_json=not args.no_json,
    )
