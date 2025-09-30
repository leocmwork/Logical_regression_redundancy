import argparse
from pathlib import Path
import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


def parse_l1_ratio_from_method(method: str) -> float | None:
    m = method.lower()
    if m == "ridge":
        return 0.0
    if m == "lasso":
        return 1.0
    mobj = re.match(r"en_lr(\d+)_(\d+)$", m)
    if mobj:
        whole, frac = mobj.groups()
        return float(f"{int(whole)}.{int(frac)}")
    return None


def score_model_on_test(model_path: Path, X: pd.DataFrame, y: np.ndarray) -> dict:
    pipe: Pipeline = joblib.load(model_path)
    try:
        proba = pipe.predict_proba(X)
    except Exception:
        if hasattr(pipe, "decision_function"):
            from scipy.special import expit
            scores = pipe.decision_function(X)
            proba = np.vstack([1 - expit(scores), expit(scores)]).T
        else:
            raise RuntimeError(f"Model {model_path.name} lacks predict_proba/decision_function.")
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise RuntimeError(f"Unexpected predict_proba shape for {model_path.name}: {proba.shape}")
    y_score = proba[:, 1]
    auc = roc_auc_score(y, y_score)

    penalty = C = l1_ratio = None
    try:
        clf = pipe.named_steps.get("clf", None)
        if clf is not None:
            penalty = getattr(clf, "penalty", None)
            C = getattr(clf, "C", None)
            l1_ratio = getattr(clf, "l1_ratio", None)
    except Exception:
        pass

    return {
        "auc": float(auc),
        "penalty": penalty if penalty is not None else "",
        "C": float(C) if C is not None else np.nan,
        "l1_ratio_model": float(l1_ratio) if l1_ratio is not None else np.nan,
    }


def evaluate_one_dataset(test_csv: Path, target: str) -> pd.DataFrame | None:
    dataset_name = test_csv.stem.replace("_test", "")
    models_dir = Path("models") / dataset_name
    if not models_dir.exists():
        print(f"No models found for {dataset_name} in {models_dir}")
        return None

    df = pd.read_csv(test_csv)
    if target not in df.columns:
        print(f"Target '{target}' not in {test_csv.name}")
        return None

    X_test = df.drop(columns=[target])
    y_test = df[target].values
    pos_rate = float(np.mean(y_test))
    n_predictors = int(X_test.shape[1])

    rows = []
    for model_path in sorted(models_dir.glob("*.joblib")):
        method = model_path.stem
        try:
            res = score_model_on_test(model_path, X_test, y_test)
            l1_from_name = parse_l1_ratio_from_method(method)
            l1_used = l1_from_name if l1_from_name is not None else res["l1_ratio_model"]

            rows.append({
                "dataset": dataset_name,
                "method": method,
                "l1_ratio": l1_used,
                "penalty": res["penalty"],
                "C": res["C"],
                "auc_test": res["auc"],
                "n_predictors": n_predictors,
                "n_test": int(len(y_test)),
                "pos_rate_test": pos_rate,
            })
        except Exception as e:
            print(f"Skipped {dataset_name} | {model_path.name}: {e}")

    if not rows:
        print(f"No evaluated models for {dataset_name}")
        return None

    res_df = pd.DataFrame(rows).sort_values("auc_test", ascending=False)
    out_dir = Path("results") / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{dataset_name}_test_scores.csv"
    res_df.to_csv(out_csv, index=False)
    best = res_df.iloc[0]
    print(f"{dataset_name}: best={best['method']} | AUC_test={best['auc_test']:.4f} "
          f"(n={int(best['n_test'])}, pos_rate={best['pos_rate_test']:.3f}) | saved -> {out_csv}")
    return res_df


def _read_meta(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _build_titles(meta: dict | None, fallback: str = "") -> tuple[str, str]:
    if meta:
        noise = meta.get("noise_sd", "NA")
        main = f"Comparative AUC vs # added redundant variables (noise_sd={noise})"
        beta = meta.get("beta", "")
        rlist = meta.get("r_list", "")
        mlist = meta.get("mix_list", "")
        seed = meta.get("random_state", "")
        sub = f'beta={beta}  |  r-list="{rlist}"  |  mix-list="{mlist}"  |  random-state={seed}'
        return main, sub
    main = "Comparative AUC vs # added redundant variables"
    return main, fallback


def _plot_auc_vs_added(big: pd.DataFrame, meta: dict | None, fallback_title: str, out_path: Path):
    if big.empty:
        print("Nothing to plot.")
        return

    base_p = int(big["n_predictors"].min())
    plot_df = big.dropna(subset=["l1_ratio"]).copy()
    plot_df["added"] = plot_df["n_predictors"] - base_p
    plot_df["l1_ratio"] = plot_df["l1_ratio"].astype(float)

    main_title, subtitle = _build_titles(meta, fallback_title)

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        plot_df["added"].values,
        plot_df["auc_test"].values,
        c=plot_df["l1_ratio"].values,
        cmap="autumn_r",
        vmin=0.0, vmax=1.0,
        alpha=0.9,
        edgecolors="none",
        s=38,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("l1_ratio (0=Ridge, 1=Lasso)")

    ax.set_xlabel("# added redundant predictors")
    ax.set_ylabel("ROC AUC (test)")

    max_added = int(plot_df["added"].max())
    ax.set_xlim(-0.2, max_added + 0.5)
    ax.set_xticks(list(range(0, max_added + 1)))

    fig.suptitle(main_title, fontsize=12, fontweight="bold", y=0.98)
    ax.set_title(subtitle, fontsize=9, pad=8)

    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate all models on test sets (ROC AUC) and plot AUC vs #added.")
    ap.add_argument("--glob", default="data/test/redundancy_series_rank*_test.csv",
                    help="Glob for test CSVs.")
    ap.add_argument("--target", default="target", help="Target column.")
    ap.add_argument("--meta-json", default="results/summary/redundancy_series_meta.json",
                    help="Path to generator metadata JSON (for plot title).")
    ap.add_argument("--plot-title", default="", help="Fallback subtitle if meta not found.")
    args = ap.parse_args()

    test_files = sorted(Path().glob(args.glob))
    if not test_files:
        print(f"No test files matched: {args.glob}")
        return

    all_rows = []
    for test_csv in test_files:
        res_df = evaluate_one_dataset(test_csv, args.target)
        if res_df is not None:
            all_rows.append(res_df)

    if not all_rows:
        print("Nothing evaluated.")
        return

    big = pd.concat(all_rows, ignore_index=True)
    sum_dir = Path("results/summary"); sum_dir.mkdir(parents=True, exist_ok=True)
    out_all = sum_dir / "redundancy_series_test_summary.csv"
    big.to_csv(out_all, index=False)
    print(f"\nGlobal summary saved -> {out_all}")

    try:
        piv_pred = (big
                    .dropna(subset=["l1_ratio"])
                    .pivot_table(index="n_predictors", columns="l1_ratio",
                                 values="auc_test", aggfunc="mean"))
        piv_pred_out = sum_dir / "redundancy_series_test_auc_by_predictors_and_l1ratio.csv"
        piv_pred.to_csv(piv_pred_out)
        base_p = int(big["n_predictors"].min())
        big_with_added = big.copy()
        big_with_added["added"] = big_with_added["n_predictors"] - base_p
        piv_add = (big_with_added
                   .dropna(subset=["l1_ratio"])
                   .pivot_table(index="added", columns="l1_ratio",
                                values="auc_test", aggfunc="mean"))
        piv_add_out = sum_dir / "redundancy_series_test_auc_by_added_and_l1ratio.csv"
        piv_add.to_csv(piv_add_out)
        print(f"Saved pivots -> {piv_pred_out}  &  {piv_add_out}")
    except Exception as e:
        print(f"Pivot not saved: {e}")

    meta = _read_meta(Path(args.meta_json))
    out_png = sum_dir / "redundancy_series_test_auc_vs_added.png"
    _plot_auc_vs_added(big, meta, args.plot_title, out_png)


if __name__ == "__main__":
    main()
