import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .common import read_table, normalize_schema, profile, save_json

def main():
    ap=argparse.ArgumentParser(description="Create descriptive EDA tables and charts.")
    ap.add_argument("--input",required=True); ap.add_argument("--output",default="outputs")
    args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    df=normalize_schema(read_table(args.input))
    save_json(profile(df),out/"eda_profile.json")
    df.describe(include="all").transpose().to_csv(out/"descriptive_statistics.csv")
    df.isna().mean().mul(100).sort_values(ascending=False).rename("missing_percent").to_csv(out/"missingness.csv")
    sns.set_theme()
    if "rating" in df:
        x=pd.to_numeric(df["rating"],errors="coerce").dropna()
        if len(x):
            plt.figure(figsize=(7,4)); sns.histplot(x,bins=20)
            plt.title("Review rating distribution"); plt.xlabel("Rating"); plt.ylabel("Number of records")
            plt.tight_layout(); plt.savefig(out/"eda_rating_distribution.png",dpi=160); plt.close()
    if "recommended" in df:
        x=df["recommended"].dropna().astype(str).value_counts()
        if len(x):
            plt.figure(figsize=(7,4)); x.plot(kind="bar")
            plt.title("Recommendation label counts"); plt.xlabel("Label"); plt.ylabel("Number of records")
            plt.tight_layout(); plt.savefig(out/"eda_recommendation_counts.png",dpi=160); plt.close()
    if "category" in df:
        x=df["category"].fillna("unknown").value_counts().head(20)
        if len(x):
            plt.figure(figsize=(9,5)); x.sort_values().plot(kind="barh")
            plt.title("Top categories by record count"); plt.xlabel("Records"); plt.ylabel("Category")
            plt.tight_layout(); plt.savefig(out/"eda_category_counts.png",dpi=160); plt.close()
    if "review_date" in df:
        dates=pd.to_datetime(df["review_date"],errors="coerce",utc=True).dropna()
        if len(dates):
            counts=dates.dt.to_period("M").value_counts().sort_index()
            plt.figure(figsize=(10,4)); counts.plot()
            plt.title("Review volume by month"); plt.xlabel("Month"); plt.ylabel("Records")
            plt.tight_layout(); plt.savefig(out/"eda_reviews_over_time.png",dpi=160); plt.close()
    print("EDA outputs saved under",out)
if __name__=="__main__": main()
