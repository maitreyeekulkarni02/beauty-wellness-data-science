import argparse
from pathlib import Path
from .common import read_table, normalize_schema, profile, save_json

def main():
    ap=argparse.ArgumentParser(description="Inspect raw review dataset schema and quality.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="outputs")
    args=ap.parse_args()
    df=normalize_schema(read_table(args.input))
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    p=profile(df); save_json(p, out/"data_profile.json")
    df.isna().mean().mul(100).rename("missing_percent").to_csv(out/"missingness.csv")
    print("Profile saved to", out/"data_profile.json")
    print("Rows:", p["rows"], "| columns:", p["columns"])
    print("Detected columns:", ", ".join(df.columns))
    print("Missing %:", p["missing_pct"])
if __name__=="__main__": main()
