import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split_one(csv_path: Path, target: str, test_size: float, random_state: int) -> tuple[Path, Path]:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in {csv_path.name}")

    y = df[target]
    classes = y.dropna().unique()
    if len(classes) != 2:
        raise ValueError(f"Target must be binary in {csv_path.name}; found {len(classes)} classes.")

    X = df.drop(columns=[target])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    train_df = X_tr.copy(); train_df[target] = y_tr
    test_df  = X_te.copy();  test_df[target]  = y_te

    stem = csv_path.stem
    out_train = Path("data/train") / f"{stem}_train.csv"
    out_test  = Path("data/test") / f"{stem}_test.csv"
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test, index=False)

    print(f"Split {csv_path.name} -> {out_train.name} / {out_test.name} "
          f"(train={train_df.shape}, test={test_df.shape})")
    return out_train, out_test


def main():
    ap = argparse.ArgumentParser(description="Stratified 80/20 splits for redundancy series datasets.")
    ap.add_argument("--glob", default="data/raw/redundancy_series_rank*.csv", help="Glob for input CSVs.")
    ap.add_argument("--target", default="target", help="Target column name.")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test fraction.")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    files = sorted(Path().glob(args.glob))
    if not files:
        print(f"No files matched: {args.glob}")
        return

    n_ok = 0
    for f in files:
        try:
            split_one(f, args.target, args.test_size, args.random_state)
            n_ok += 1
        except Exception as e:
            print(f"Skipped {f.name}: {e}")

    print(f"\nDone. Datasets processed: {n_ok}/{len(files)}")


if __name__ == "__main__":
    main()
