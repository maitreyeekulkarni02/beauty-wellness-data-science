from pathlib import Path
import subprocess, sys
def main():
    raw=Path("data/raw")
    files=[p for p in raw.iterdir() if p.suffix.lower() in {".csv",".xlsx",".xls"}]
    if not files:
        raise SystemExit("No raw dataset found. Download a permitted review CSV into data/raw/; see README.md.")
    input_path=files[0]
    print("Using first raw file:",input_path)
    subprocess.run([sys.executable,"-m","src.inspect_data","--input",str(input_path)],check=True)
    subprocess.run([sys.executable,"-m","src.clean_data","--input",str(input_path),"--output","outputs/cleaned_reviews.csv"],check=True)
    subprocess.run([sys.executable,"-m","src.run_eda","--input","outputs/cleaned_reviews.csv"],check=True)
    subprocess.run([sys.executable,"-m","src.train_model","--input","outputs/cleaned_reviews.csv"],check=True)
if __name__=="__main__": main()
