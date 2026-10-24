"""Shared helpers for schema-aware skincare review analysis."""
from pathlib import Path
import json, re
import pandas as pd
import numpy as np

COLUMN_ALIASES = {
    "rating": ["rating", "score", "stars", "review_rating", "review_score"],
    "review_text": ["review_text", "review", "content", "text", "review_body", "reviewtext"],
    "recommended": ["recommended", "recommendation", "is_recommended", "recommend", "recommended_ind"],
    "product_id": ["product_id", "productid", "product", "sku", "product_code", "productid"],
    "product_name": ["product_name", "product", "name", "product_title", "title"],
    "category": ["category", "primary_category", "product_category", "sub_category", "subcategory"],
    "brand": ["brand", "brand_name"],
    "price": ["price", "value_price", "sale_price", "current_price", "price_usd"],
    "review_date": ["review_date", "date", "submission_time", "created_at", "review_time"],
    "review_id": ["review_id", "reviewid", "id"],
    "skin_type": ["skin_type", "skintype", "skin type"],
    "verified_purchase": ["verified_purchase", "verified", "is_verified_purchase"],
}

def normalize_col(s):
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

def read_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}. Download a permitted dataset and place it in data/raw/.")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)

def normalize_schema(df):
    df = df.copy()
    df.columns = [normalize_col(c) for c in df.columns]
    # Map known aliases to canonical names; do not overwrite an existing canonical column.
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical not in df.columns:
            for alias in aliases:
                alias_n = normalize_col(alias)
                if alias_n in df.columns:
                    df = df.rename(columns={alias_n: canonical})
                    break
    return df

def profile(df):
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(map(str, df.columns)),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_count": {c: int(v) for c, v in df.isna().sum().items()},
        "missing_pct": {c: round(float(v), 4) for c, v in (df.isna().mean()*100).items()},
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
