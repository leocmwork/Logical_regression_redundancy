import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def build_logreg_for_l1(l1_ratio: float, C: float) -> LogisticRegression:
    common = dict(C=C, solver="saga", max_iter=20000, tol=1e-4, random_state=42)
    if l1_ratio <= 1e-12:
        return LogisticRegression(penalty="l2", **common)
    elif l1_ratio >= 1 - 1e-12:
        return LogisticRegression(penalty="l1", **common)
    else:
        return LogisticRegression(penalty="elasticnet", l1_ratio=float(l1_ratio), **common)


def method_name_for_l1(l1_ratio: float) -> str:
    if l1_ratio <= 1e-12:
        return "ridge"
    if l1_ratio >= 1 - 1e-12:
        return "lasso"
    return f"en_lr{l1_ratio:.1f}".replace(".", "_")


def crossval_and_save(X, y, pre, dataset_name: str, l1_ratio: float, C_grid, out_dir: Path, model_dir: Path):
    method = method_name_for_l1(l1_ratio)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    records = []
    best_auc = -np.inf
    best_C = None

    for C in C_grid:
        clf = build_logreg_for_l1(l1_ratio, C)
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        mean_auc, std_auc = float(np.mean(scores)), float(np.std(scores))
        records.append({
            "dataset": dataset_name,
            "method": method,
            "l1_ratio": float(l1_ratio),
            "C": C,
            "cv_mean_auc": mean_auc,
            "cv_std_auc": std_auc
        })
        if mean_auc > best_auc:
            best_auc, best_C = mean_auc, C

    best_clf = build_logreg_for_l1(l1_ratio, best_C)
    best_pipe = Pipeline([("pre", pre), ("clf", best_clf)])
    best_pipe.fit(X, y)

    meth_df = pd.DataFrame(records)
    meth_csv = out_dir / f"{method}_cv.csv"
    meth_df.to_csv(meth_csv, index=False)

    model_path = model_dir / f"{method}.joblib"
    joblib.dump(best_pipe, model_path)

    print(f"{dataset_name} | {method}: best C={best_C}, CV AUC={best_auc:.4f}")
    return meth_df


def train_one_dataset(train_csv: Path, target: str, c_grid, l1_grid):
    df = pd.read_csv(train_csv)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in {train_csv.name}")
    dataset_name = train_csv.stem.replace("_train", "")

    X = df.drop(columns=[target])
    y = df[target].values

    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = build_preprocessor(num_cols, cat_cols)

    out_dir = Path("results") / dataset_name
    model_dir = Path("models") / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    all_cv = []
    for lr in l1_grid:
        cv_df = crossval_and_save(X, y, pre, dataset_name, lr, c_grid, out_dir, model_dir)
        all_cv.append(cv_df)

    sweep_df = pd.concat(all_cv, ignore_index=True)
    agg_csv = out_dir / f"{dataset_name}_en_sweep_cv.csv"
    sweep_df.to_csv(agg_csv, index=False)
    print(f"Saved aggregated CV sweep -> {agg_csv}")
    return len(l1_grid)


def main():
    ap = argparse.ArgumentParser(description="EN sweep (11 l1_ratio) with 5-fold CV for redundancy series.")
    ap.add_argument("--glob", default="data/train/redundancy_series_rank*_train.csv", help="Glob for train CSVs.")
    ap.add_argument("--target", default="target", help="Target column.")
    ap.add_argument("--c-grid", nargs="+", type=float, default=[0.01, 0.1, 1, 10, 100], help="C values.")
    ap.add_argument("--l1-grid", nargs="+", type=float,
                    default=[round(x, 1) for x in np.linspace(0.0, 1.0, 11)],
                    help="l1_ratio values (0.0..1.0).")
    args = ap.parse_args()

    train_files = sorted(Path().glob(args.glob))
    if not train_files:
        print(f"No train files matched: {args.glob}")
        return

    total_models = 0
    for f in train_files:
        try:
            n = train_one_dataset(f, args.target, args.c_grid, args.l1_grid)
            total_models += n
        except Exception as e:
            print(f"Skipped {f.name}: {e}")

    print(f"\nDone. Datasets trained: {len(train_files)} | Models saved: {total_models}")
    print("Models in models/<dataset>/ | CV in results/<dataset>/")


if __name__ == "__main__":
    main()
