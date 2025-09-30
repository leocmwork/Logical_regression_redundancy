import argparse
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Tuple


def compute_vif_ols(X: np.ndarray, cap: float = 1e5):
    X = np.asarray(X, float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-12)
    n, p = X.shape
    vifs = []
    for j in range(p):
        y = X[:, j]
        Xo = np.delete(X, j, axis=1)
        Xo_i = np.column_stack([np.ones(n), Xo])
        beta, *_ = np.linalg.lstsq(Xo_i, y, rcond=None)
        yhat = Xo_i @ beta
        resid = y - yhat
        RSS = float((resid ** 2).sum())
        TSS = float(((y - y.mean()) ** 2).sum())
        R2 = 1.0 - RSS / TSS if TSS > 0 else 0.0
        tol = max(1e-12, 1.0 - R2)
        vif = 1.0 / tol
        if not np.isfinite(vif) or vif > cap:
            vif = cap
        vifs.append(vif)
    vifs = np.array(vifs, float)
    return float(np.mean(vifs)), float(np.median(vifs)), float(np.max(vifs))


def parse_beta(s: str) -> np.ndarray:
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 5:
        raise ValueError("--beta must have exactly 5 numbers.")
    return np.asarray(vals, dtype=float)

def parse_r_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_mix_list(s: str) -> List[Tuple[float, float]]:
    pairs = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            a, b = chunk.split(":")
        elif "," in chunk:
            a, b = chunk.split(",")
        else:
            raise ValueError(f"Bad pair format: '{chunk}' (use 'a,b' or 'a:b').")
        a, b = float(a.strip()), float(b.strip())
        if a**2 + b**2 > 1 + 1e-9:
            raise ValueError(f"(alpha,beta)=({a},{b}) violates alpha^2+beta^2 <= 1.")
        pairs.append((a, b))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="Generate redundancy series datasets (with metadata).")
    ap.add_argument("--n-samples", type=int, default=4000)
    ap.add_argument("--beta", type=str, default="1.2,-1.0,0.8,0.6,0.4")
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--noise-sd", type=float, default=0.0)
    ap.add_argument("--r-list", type=str, default="0.9,0.8,0.7,0.6,0.5")
    ap.add_argument("--mix-list", type=str, default="0.9,0.1;0.8,0.2;0.7,0.3;0.6,0.4;0.5,0.5")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.random_state)

    Z = rng.standard_normal((args.n_samples, 5))
    Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0, ddof=0) + 1e-12)

    beta = parse_beta(args.beta)
    r_values = parse_r_list(args.r_list)
    mix_pairs = parse_mix_list(args.mix_list)

    eta = Z @ beta
    if args.noise_sd > 0:
        eta = eta + rng.normal(0.0, args.noise_sd, size=eta.shape[0])
    p = 1.0 / (1.0 + np.exp(-(eta + args.bias)))
    y = (rng.uniform(0.0, 1.0, size=p.shape[0]) < p).astype(int)
    pos_rate = float(np.mean(y))

    out_raw = Path("data/raw"); out_raw.mkdir(parents=True, exist_ok=True)
    sum_dir = Path("results/summary"); sum_dir.mkdir(parents=True, exist_ok=True)

    added_labels = []
    added_meta = []

    step = 1
    base_cols = {f"z{i+1}": Z[:, i] for i in range(5)}
    df = pd.DataFrame(base_cols)
    df["target"] = y
    out = out_raw / f"redundancy_series_rank{step:02d}.csv"
    df.to_csv(out, index=False)
    m, med, mx = compute_vif_ols(df.drop(columns=["target"]).values, cap=1e5)
    print(f"[{step:02d}] Saved {out} | step=BASE | p={df.shape[1]-1} | mean VIF={m:.2f} | pos_rate={pos_rate:.3f}")

    summary_rows = [{
        "dataset": f"redundancy_series_rank{step:02d}",
        "step": step,
        "block": "BASE",
        "added": "BASE",
        "params": "",
        "n_predictors": int(df.shape[1]-1),
        "mean_vif": m, "median_vif": med, "max_vif": mx,
        "pos_rate": pos_rate
    }]

    for r in r_values:
        step += 1
        E = rng.standard_normal(args.n_samples)
        x = r * Z[:, 0] + np.sqrt(max(0.0, 1.0 - r**2)) * E

        df = pd.DataFrame(base_cols)
        for label, meta in zip(added_labels, added_meta):
            df[label] = meta["x"]
        label = f"A_z1_r{r:.1f}".replace(".", "_")
        df[label] = x
        df["target"] = y

        out = out_raw / f"redundancy_series_rank{step:02d}.csv"
        df.to_csv(out, index=False)

        m, med, mx = compute_vif_ols(df.drop(columns=["target"]).values, cap=1e5)
        print(f"[{step:02d}] Saved {out} | added={label} | p={df.shape[1]-1} | mean VIF={m:.2f}")

        added_labels.append(label)
        added_meta.append({"block": "A", "r": r, "x": x})
        summary_rows.append({
            "dataset": f"redundancy_series_rank{step:02d}",
            "step": step,
            "block": "A",
            "added": label,
            "params": f"r={r}",
            "n_predictors": int(df.shape[1]-1),
            "mean_vif": m, "median_vif": med, "max_vif": mx,
            "pos_rate": pos_rate
        })

    for (a, b) in mix_pairs:
        step += 1
        gamma = np.sqrt(max(0.0, 1.0 - a*a - b*b))
        E = rng.standard_normal(args.n_samples)
        x = a * Z[:, 1] + b * Z[:, 2] + gamma * E

        df = pd.DataFrame(base_cols)
        for label, meta in zip(added_labels, added_meta):
            df[label] = meta["x"]
        label = f"B_z2z3_a{a:.1f}_b{b:.1f}".replace(".", "_")
        df[label] = x
        df["target"] = y

        out = out_raw / f"redundancy_series_rank{step:02d}.csv"
        df.to_csv(out, index=False)

        m, med, mx = compute_vif_ols(df.drop(columns=["target"]).values, cap=1e5)
        print(f"[{step:02d}] Saved {out} | added={label} | p={df.shape[1]-1} | mean VIF={m:.2f}")

        added_labels.append(label)
        added_meta.append({"block": "B", "alpha": a, "beta": b, "x": x})
        summary_rows.append({
            "dataset": f"redundancy_series_rank{step:02d}",
            "step": step,
            "block": "B",
            "added": label,
            "params": f"alpha={a},beta={b}",
            "n_predictors": int(df.shape[1]-1),
            "mean_vif": m, "median_vif": med, "max_vif": mx,
            "pos_rate": pos_rate
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = sum_dir / "redundancy_series_summary.csv"
    summary.to_csv(summary_path, index=False)

    meta = {
        "n_samples": args.n_samples,
        "beta": args.beta,
        "bias": args.bias,
        "noise_sd": args.noise_sd,
        "r_list": args.r_list,
        "mix_list": args.mix_list,
        "random_state": args.random_state,
        "cmdline": " ".join(["python"] + sys.argv),
    }
    meta_path = sum_dir / "redundancy_series_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved summary -> {summary_path}")
    print(f"Saved metadata -> {meta_path}")


if __name__ == "__main__":
    main()
