# Beauty Product Recommendation and Customer Preference Analysis

A six-week, reproducible data science internship project focused on skincare reviews and product attributes.

## Business question
Which observable review/product characteristics are associated with a positive recommendation label, and how well can a simple model predict that label without leakage? If the chosen dataset does not contain a trustworthy recommendation label, the pipeline can instead evaluate rating prediction, clearly identified as a proxy task.

## Repository structure
```text
beauty_wellness_data_science_project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw/                 # Put source CSV files here locally; do not commit them
├── reports/                 # Six weekly DOCX reports
├── src/
│   ├── common.py
│   ├── inspect_data.py
│   ├── clean_data.py
│   ├── run_eda.py
│   ├── train_model.py
│   └── run_all.py
└── outputs/                 # Generated tables, charts, metrics, model artifacts
```

## Dataset sourcing
Core candidate: [Sephora Skincare Reviews & Sentiment documentation](https://huggingface.co/datasets/haoxianc/natashamessier_sephora-skincare-reviews-and-sentiment). Its page describes 40,234 reviews across 2,012 skincare products. Check the current files, schema, source attribution, and license before using. Download the permitted CSV file(s) and place one review-level CSV in `data/raw/`. If product information is in a separate CSV, place it there too; the starter pipeline reads one CSV at a time and can be adapted for a documented join.

The project does not redistribute third-party datasets. The raw data folder is git-ignored. Never commit reviewer identifiers, raw personal text, credentials, or any data whose license does not permit redistribution.

Other references:
- [Related skincare review classification project](https://github.com/meiw02/skincare-review-classification) — methodological reference, not a source of this project's results.
- [openFDA cosmetic adverse events API](https://open.fda.gov/apis/cosmetic/event/) — regulatory reporting context only; reports are not incidence estimates or proof of causation.
- [UCI Online Retail](https://uci-ics-mlr-prod.aws.uci.edu/dataset/352/online%2Bretail) — general retail practice dataset, not beauty-specific; do not use it to make skincare-specific claims.

## Setup
Python 3.10+ recommended.
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
Place a CSV in `data/raw/`, then:
```bash
python -m src.inspect_data --input data/raw/your_reviews.csv
python -m src.clean_data --input data/raw/your_reviews.csv --output outputs/cleaned_reviews.csv
python -m src.run_eda --input outputs/cleaned_reviews.csv
python -m src.train_model --input outputs/cleaned_reviews.csv
# Or run the workflow (uses the first CSV in data/raw):
python -m src.run_all
```
Scripts print the detected columns and will clearly stop if a target cannot be identified. Use `--target recommended` to force a documented binary target, or `--target rating` for rating regression. Do not choose a target merely because it gives better scores; justify it by the business question.

## Column handling
Common aliases are mapped (e.g. `rating`/`score`, `review_text`/`content`, `recommended`/`recommendation`, `product_id`/`productid`, `price`/`value_price`). Check the generated schema report. If your dataset uses other names, edit `COLUMN_ALIASES` in `src/common.py` and document the mapping.

## Method and validation
- A dummy baseline is compared with logistic regression and random forest.
- Preprocessing is placed inside scikit-learn pipelines to reduce leakage.
- Classification metrics: precision, recall, F1, ROC-AUC when valid, and confusion matrix.
- Regression metrics: MAE and RMSE.
- Random splitting is a starting point only. If multiple reviews belong to each product, also evaluate group-aware or time-based validation before claiming generalization.
- Text features are not included by default; this starter focuses on available structured fields. Add TF-IDF only if the review text is available and the pipeline fits it inside cross-validation.

## Outputs
Generated files may include `outputs/data_profile.json`, `outputs/missingness.csv`, `outputs/eda_*.png`, `outputs/model_metrics.csv`, `outputs/confusion_matrix.png`, `outputs/feature_importance.csv`, and `outputs/best_model.joblib`.

## Important interpretation limits
This is an educational prototype. Public reviews are self-selected and may not represent all shoppers. Predictions are not causal claims, medical advice, proof of efficacy, or a measure of product safety. Do not infer individual skin suitability or diagnose a condition. Do not claim performance numbers until the code has been run on a documented real dataset; include the dataset version, target definition, split strategy, and output metrics in the final submission.
