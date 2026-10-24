import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from .common import read_table, normalize_schema, profile, save_json

def clean(df):
    before=len(df)
    df=normalize_schema(df)
    audit={"rows_input": int(before), "columns_input": list(df.columns)}
    # Strip whitespace from text/object fields; standardize empty placeholders.
    for c in df.select_dtypes(include=["object"]).columns:
        df[c]=df[c].astype("string").str.strip()
        df[c]=df[c].replace({"":pd.NA, "NA":pd.NA, "N/A":pd.NA, "null":pd.NA, "None":pd.NA})
    # Numeric conversions are attempted only for canonical known fields.
    if "rating" in df:
        df["rating"]=pd.to_numeric(df["rating"], errors="coerce")
        valid=df["rating"].dropna()
        # Do not assume a scale. Flag values outside a commonly used 1-5 range for review.
        audit["rating_values_outside_1_to_5"] = int(((valid<1)|(valid>5)).sum())
    if "price" in df:
        df["price"]=pd.to_numeric(df["price"].astype("string").str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
        audit["nonpositive_price_count"] = int((df["price"]<=0).sum())
    if "review_date" in df:
        df["review_date"]=pd.to_datetime(df["review_date"], errors="coerce", utc=True)
        audit["unparsed_review_date_count"]=int(df["review_date"].isna().sum())
    if "recommended" in df:
        # Map common binary encodings while preserving unknowns.
        def map_rec(v):
            if pd.isna(v): return np.nan
            s=str(v).strip().lower()
            if s in {"1","true","yes","y","recommended","positive"}: return 1
            if s in {"0","false","no","n","not recommended","negative"}: return 0
            return np.nan
        mapped=df["recommended"].map(map_rec)
        # Only replace if at least one recognized value exists; unrecognized values become missing.
        if mapped.notna().any():
            df["recommended"]=mapped
    if "review_text" in df:
        df["review_text"]=df["review_text"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
        df["review_length_chars"]=df["review_text"].str.len()
        df["review_length_words"]=df["review_text"].str.split().str.len()
    if "category" in df:
        df["category"]=df["category"].astype("string").str.strip().str.lower()
    if "brand" in df:
        df["brand"]=df["brand"].astype("string").str.strip()
    duplicate_count=int(df.duplicated().sum())
    df=df.drop_duplicates().copy()
    audit.update({"exact_duplicate_rows_removed":duplicate_count, "rows_output":int(len(df)),
                  "rows_removed_total":int(before-len(df)), "columns_output":list(df.columns),
                  "profile_after":profile(df)})
    return df,audit

def main():
    ap=argparse.ArgumentParser(description="Clean review data and write an audit log.")
    ap.add_argument("--input",required=True); ap.add_argument("--output",default="outputs/cleaned_reviews.csv")
    args=ap.parse_args()
    df,audit=clean(read_table(args.input))
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(out,index=False)
    save_json(audit,out.parent/"cleaning_audit.json")
    print("Cleaned file:",out,"Rows:",len(df))
    print("Audit log:",out.parent/"cleaning_audit.json")
if __name__=="__main__": main()
