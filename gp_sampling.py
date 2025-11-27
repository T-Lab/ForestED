# gp_uncertainty_sampler.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.utils import check_random_state


def sample_by_gp_uncertainty(
    csv_path: str,
    output_csv: str | None = None,
    sample_frac: float = 0.05,
    max_train: int = 2000,
    pca_components: int = 20,
    random_state: int = 42,
):
    """
    Sample ~5% rows from a CSV using a Gaussian-Process-uncertainty-based algorithm.

    Adds an extra column `sample_row_id` to mark the original row index.
    """
    rng = check_random_state(random_state)

    # -------- 1) Load ----------
    df = pd.read_csv(csv_path)
    n = len(df)
    if n == 0:
        raise ValueError("Empty CSV.")

    # Identify column types
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]

    # Basic imputations to avoid failures
    df_num = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)
    df_cat = df[cat_cols].copy() if cat_cols else pd.DataFrame(index=df.index)

    if num_cols:
        for c in num_cols:
            if df_num[c].isna().any():
                df_num[c] = df_num[c].fillna(df_num[c].median())

    if cat_cols:
        for c in cat_cols:
            df_cat[c] = df_cat[c].astype("string").fillna("NA")

    df = pd.concat([df_num, df_cat], axis=1)

    # -------- 2) Preprocess pipeline ----------
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(with_mean=True, with_std=True), num_cols))
    if cat_cols:
        ohe = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=0.01  # merge very rare categories
        )
        transformers.append(("cat", ohe, cat_cols))

    ct = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)

    pipe = Pipeline([
        ("ct", ct),
        ("pca", PCA(n_components=min(pca_components, max(1, min(n, pca_components)))))
    ])

    # Fit transform
    X = pipe.fit_transform(df)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # -------- 3) Fit GP on subset ----------
    n_train = min(max_train, n)
    train_idx = rng.choice(n, size=n_train, replace=False)

    X_train = X[train_idx]
    y_train = np.zeros(n_train, dtype=float)

    kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=False,
        random_state=random_state
    )
    gpr.fit(X_train, y_train)

    # -------- 4) Predict std for all rows ----------
    _, std_all = gpr.predict(X, return_std=True)
    std_all = np.asarray(std_all)

    k = max(1, int(math.ceil(sample_frac * n)))
    k = min(100, k)
    top_idx = np.argpartition(-std_all, kth=k-1)[:k]
    top_idx = top_idx[np.argsort(std_all[top_idx])[::-1]]

    sampled_df = df.iloc[top_idx].copy()

    # 添加原始 row id
    sampled_df["sample_row_id"] = top_idx

    # -------- 5) Save ----------
    if output_csv:
        sampled_df.to_csv(output_csv, index=False)

    return sampled_df, top_idx


# Example CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sample ~5% rows via GP uncertainty.")
    parser.add_argument("csv_path", type=str, help="Path to input CSV file.")
    parser.add_argument("--out", type=str, default=None, help="Optional output CSV for the sample.")
    parser.add_argument("--frac", type=float, default=0.05, help="Sampling fraction (default 0.05).")
    parser.add_argument("--max-train", type=int, default=2000, help="Max rows to train the GP on.")
    parser.add_argument("--pca", type=int, default=20, help="PCA components (default 20).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    sampled, idx = sample_by_gp_uncertainty(
        args.csv_path, output_csv=args.out, sample_frac=args.frac,
        max_train=args.max_train, pca_components=args.pca, random_state=args.seed
    )
    print(f"Sampled {len(sampled)} rows. Added column 'sample_row_id' to track original indices.")
