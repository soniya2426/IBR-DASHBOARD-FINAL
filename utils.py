import pandas as pd
import numpy as np

CANONICAL_COLS = {
    "personalized": "personalisation",
    "authentic_brand": "brand_authenticity",
    "Halal_certification_trust": "halal_certified",
    "Gen_Z_consumers": "gen_z_perception",
    "brand_loyalty": "brand_loyalty",
    "purchase_intention": "willingness_to_buy",
    "pricing": "pricing",
    "transparency": "transparency",
    "consistency": "consistency",
    "sustainable": "sustainable",
    "would_recommend": "would_recommend"
}

KEY_ORDER = [
    "personalisation","brand_authenticity","halal_certified","pricing",
    "gen_z_perception","brand_loyalty","willingness_to_buy",
    "transparency","consistency","sustainable","would_recommend"
]

NUMERIC_GUESS = list(CANONICAL_COLS.values())

def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase + strip
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    canon = {}
    for col in df.columns:
        if col in CANONICAL_COLS:
            canon[col] = CANONICAL_COLS[col]
        else:
            # Already canonical or leave as is
            canon[col] = CANONICAL_COLS.get(col, col)
    df = df.rename(columns=canon)
    return df

def add_brand(df: pd.DataFrame, brand_name: str) -> pd.DataFrame:
    df = df.copy()
    df["brand"] = brand_name
    return df

def numeric_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number])

def nps_from_5pt(series: pd.Series):
    # promoters=5, passives=4, detractors=1-3
    counts = series.value_counts(dropna=True)
    tot = counts.sum() if counts.sum() else 1
    promoters = counts.get(5, 0) / tot * 100.0
    detractors = (counts.get(1,0) + counts.get(2,0) + counts.get(3,0)) / tot * 100.0
    nps = promoters - detractors
    return nps, promoters, detractors

def zscore(df, cols):
    out = df.copy()
    for c in cols:
        mu = out[c].mean()
        sd = out[c].std(ddof=0) or 1.0
        out[c+"_z"] = (out[c] - mu)/sd
    return out

def safe_merge(estee, shiffa):
    e = harmonize_columns(estee)
    s = harmonize_columns(shiffa)
    return e, s
