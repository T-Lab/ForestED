# data_catalog.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any
import math
import random
import numpy as np
import pandas as pd

# -----------------------------
# Utilities
# -----------------------------

def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def _is_bool(series: pd.Series) -> bool:
    return pd.api.types.is_bool_dtype(series)

def _is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)

def _safe_to_str(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    return s if len(s) <= 256 else s[:256]

def _nan_ratio(series: pd.Series) -> float:
    return float(series.isna().mean())

def _distinct_ratio(series: pd.Series) -> float:
    n = len(series)
    if n == 0:
        return 0.0
    return float(series.dropna().nunique() / n)

def _infer_is_categorical(series: pd.Series) -> bool:
    """
    New rule:
    - Only use distinct ratio to decide categorical.
    - If unique ratio < 0.1 (10%), treat as categorical.
    - Otherwise, not categorical.
    """
    ratio = _distinct_ratio(series)
    return ratio < 0.1

# -----------------------------
# Column "embeddings" (length 300)
# -----------------------------
def _hash_embedding(series: pd.Series, dim: int = 300, seed: int = 13) -> np.ndarray:
    rng = random.Random(seed)
    emb = np.zeros(dim, dtype=np.float64)

    s = series.dropna()
    n = len(s)

    if n == 0:
        return emb

    if _is_numeric(series):
        vals = pd.to_numeric(s, errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            return emb

        # 1) histogram (50 bins)
        counts, _ = np.histogram(vals, bins=50)
        counts = counts.astype(np.float64)
        if counts.sum() > 0:
            counts /= counts.sum()
        emb[:50] = counts

        # 2) summary stats (5 dims)
        stats = np.array([
            np.mean(vals),
            np.std(vals) if np.std(vals) > 0 else 0.0,
            np.min(vals),
            np.median(vals),
            np.max(vals)
        ], dtype=np.float64)
        stats = (stats - stats.mean()) / (stats.std() + 1e-12)
        emb[50:55] = stats

        # 3) hashed buckets
        buckets = np.zeros(dim - 55, dtype=np.float64)
        for v in vals:
            token = f"{round(float(v), 3)}"
            h = (hash(token) % len(buckets))
            buckets[h] += 1.0
        if buckets.sum() > 0:
            buckets /= buckets.sum()
        emb[55:] = buckets
        return emb

    # Non-numeric: hashed bag
    buckets = np.zeros(dim, dtype=np.float64)
    sample_vals = s.sample(min(20000, len(s)), random_state=seed) if len(s) > 20000 else s
    for v in sample_vals:
        token = _safe_to_str(v)
        toks = token.split()
        toks.append(token)
        for t in toks:
            if not t:
                continue
            h = hash(t) % dim
            buckets[h] += 1.0
    if buckets.sum() > 0:
        buckets /= buckets.sum()
    return buckets

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# -----------------------------
# Correlations
# -----------------------------
def _pearson(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = ~(x.isna() | y.isna())
    if mask.sum() < 2:
        return 0.0
    return float(np.corrcoef(x[mask], y[mask])[0, 1])

def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    x = x.astype("category")
    y = y.astype("category")
    tbl = pd.crosstab(x, y)
    if tbl.size == 0:
        return 0.0
    chi2 = 0.0
    n = tbl.values.sum()
    row_sums = tbl.values.sum(axis=1, keepdims=True)
    col_sums = tbl.values.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = expected > 0
        chi2 = np.where(mask, (tbl.values - expected) ** 2 / expected, 0.0).sum()
    k = min(tbl.shape)
    if n == 0 or k <= 1:
        return 0.0
    return float(math.sqrt(chi2 / (n * (k - 1))))

def _correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    y = pd.to_numeric(measurements, errors="coerce")
    mask = ~(categories.isna() | y.isna())
    categories = categories[mask].astype("category")
    y = y[mask].astype(float)
    if len(y) < 2:
        return 0.0
    grand_mean = y.mean()
    ss_between = 0.0
    for g in categories.cat.categories:
        yg = y[categories == g]
        if len(yg) == 0:
            continue
        ss_between += len(yg) * (yg.mean() - grand_mean) ** 2
    ss_total = ((y - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    return float(math.sqrt(ss_between / ss_total))

def _pairwise_correlation(s1: pd.Series, s2: pd.Series, s1_cat: bool, s2_cat: bool) -> float:
    if not s1_cat and not s2_cat:
        val = _pearson(s1, s2)
        return 0.0 if np.isnan(val) else float(val)
    if s1_cat and s2_cat:
        return _cramers_v(s1, s2)
    if s1_cat and not s2_cat:
        return _correlation_ratio(s1, s2)
    return _correlation_ratio(s2, s1)

# -----------------------------
# Inclusion dependencies
# -----------------------------
def _inclusion_dependencies(df: pd.DataFrame, col: str) -> List[str]:
    A = set(_safe_to_str(x) for x in df[col].dropna().unique() if _safe_to_str(x) != "")
    if len(A) == 0:
        return []
    res = []
    for other in df.columns:
        if other == col:
            continue
        B = set(_safe_to_str(x) for x in df[other].dropna().unique() if _safe_to_str(x) != "")
        if not B:
            continue
        misses = len(A - B)
        if misses / max(1, len(A)) <= 0.02:
            res.append(other)
    return res

# -----------------------------
# Data Catalog entry
# -----------------------------
@dataclass
class ColumnCatalog:
    dataType: str
    isCategorical: bool
    distinctionPercentage: float
    missingPercentage: float
    inDeps: List[str]
    similarities: List[Tuple[str, float]]
    correlations: List[Tuple[str, float]]
    samples: List[Any]
    statistics: Dict[str, Any]

def getColumns(df: pd.DataFrame) -> List[str]:
    return list(df.columns)

def getColumnType(df: pd.DataFrame, col: str) -> Tuple[str, bool]:
    s = df[col]
    is_cat = _infer_is_categorical(s)
    if _is_bool(s):
        dtype = "bool"
    elif _is_numeric(s):
        dtype = "numeric"
    elif _is_datetime(s):
        dtype = "datetime"
    else:
        dtype = "string"
    return dtype, is_cat

def getDistinctionPercentage(df: pd.DataFrame, col: str) -> float:
    return _distinct_ratio(df[col])

def getMissingPercentage(df: pd.DataFrame, col: str) -> float:
    return _nan_ratio(df[col])

def getInclusionDependencies(df: pd.DataFrame, col: str) -> List[str]:
    return _inclusion_dependencies(df, col)

def getSimilarities(df: pd.DataFrame, col: str, embeddings: Dict[str, np.ndarray], topk: int = 5) -> List[Tuple[str, float]]:
    sims = []
    for other, vec in embeddings.items():
        if other == col:
            continue
        sims.append((other, _cosine(embeddings[col], vec)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:topk]

def getCorrelations(df: pd.DataFrame, col: str, is_cat: bool) -> List[Tuple[str, float]]:
    res = []
    s1 = df[col]
    for other in df.columns:
        if other == col:
            continue
        s2 = df[other]
        other_is_cat = _infer_is_categorical(s2)
        val = _pairwise_correlation(s1, s2, is_cat, other_is_cat)
        if np.isnan(val):
            val = 0.0
        res.append((other, float(val)))
    res.sort(key=lambda x: abs(x[1]), reverse=True)
    return res[:10]

def getSamples(df: pd.DataFrame, col: str, tau1: int) -> List[Any]:
    s = df[col].dropna()
    if len(s) == 0:
        return []

    if _infer_is_categorical(df[col]):
        uniq_vals = pd.unique(s)
        return list(uniq_vals)

    # 非 categorical → 只 sample 最多 100 条
    return list(s.sample(min(100, len(s)), random_state=42))

def getNumericalStatisticValues(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    s = pd.to_numeric(df[col], errors="coerce")
    s = s.dropna()
    if len(s) == 0:
        return {}
    return {
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "median": float(np.median(s))
    }

# -----------------------------
# Main entry: build catalog
# -----------------------------
def build_data_catalog(csv_path: str, tau1: int = 20) -> Dict[str, dict]:
    df = pd.read_csv(csv_path)
    for c in df.columns:
        if any(k in c.lower() for k in ["date", "time", "timestamp"]):
            try:
                df[c] = pd.to_datetime(df[c], errors="ignore")
            except Exception:
                pass

    cols = getColumns(df)
    P: Dict[str, Dict[str, Any]] = {}

    embeddings = {c: _hash_embedding(df[c]) for c in cols}

    for c in cols:
        dtype, is_cat = getColumnType(df, c)
        distinct_pct = getDistinctionPercentage(df, c)
        missing_pct = getMissingPercentage(df, c)
        indeps = getInclusionDependencies(df, c)
        sims = getSimilarities(df, c, embeddings)
        cors = getCorrelations(df, c, is_cat)
        samples = getSamples(df, c, tau1)
        stats = getNumericalStatisticValues(df, c) if not is_cat else {}

        P[c] = asdict(ColumnCatalog(
            dataType=dtype,
            isCategorical=bool(is_cat),
            distinctionPercentage=float(distinct_pct),
            missingPercentage=float(missing_pct),
            inDeps=indeps,
            similarities=sims,
            correlations=cors,
            samples=samples,
            statistics=stats
        ))

    return P

def to_python_obj(obj):
    if isinstance(obj, dict):
        return {k: to_python_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_obj(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

# -----------------------------
# Example CLI usage
# -----------------------------
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Build data catalog for a CSV table.")
    parser.add_argument("csv", help="Path to input CSV")
    parser.add_argument("--tau1", type=int, default=20, help="Number of samples to store for non-categorical columns (ignored, capped at 100)")
    parser.add_argument("--out", type=str, default="", help="Optional path to write JSON catalog")
    args = parser.parse_args()

    catalog = build_data_catalog(args.csv, tau1=args.tau1)
    catalog = to_python_obj(catalog)
    
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)
    else:
        import pprint
        pprint.pprint(catalog)
