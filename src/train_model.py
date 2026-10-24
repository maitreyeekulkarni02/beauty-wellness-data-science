import argparse
from pathlib import Path
import warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, classification_report)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
import joblib
from .common import read_table, normalize_schema, save_json

DROP_FEATURES={"review_id","product_id","product_name","customer_id","user_id","author_id","id"}

def main():
    ap=argparse.ArgumentParser(description="Train baseline and candidate models; metrics are measured on supplied data.")
    ap.add_argument("--input",required=True)
    ap.add_argument("--target",choices=["auto","recommended","rating"],default="auto")
    ap.add_argument("--test-size",type=float,default=0.2)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    out=Path("outputs"); out.mkdir(exist_ok=True)
    df=normalize_schema(read_table(args.input))
    target=args.target
    if target=="auto":
        target="recommended" if "recommended" in df and df["recommended"].nunique(dropna=True)==2 else "rating"
    if target not in df:
        raise SystemExit(f"Target '{target}' not found. Available columns: {list(df.columns)}. Check schema and use --target.")
    df=df[df[target].notna()].copy()
    if target=="recommended":
        y=pd.to_numeric(df[target],errors="coerce")
        ok=y.isin([0,1]); df=df.loc[ok].copy(); y=y.loc[ok].astype(int)
        task="classification"
        if y.nunique()<2: raise SystemExit("Binary target has fewer than two classes after cleaning.")
    else:
        y=pd.to_numeric(df[target],errors="coerce")
        ok=y.notna(); df=df.loc[ok].copy(); y=y.loc[ok].astype(float)
        task="regression"
        if y.nunique()<2: raise SystemExit("Rating target has fewer than two distinct values.")
    # Remove target, direct identifiers, raw free text (starter models use structured features).
    drop={target}|DROP_FEATURES
    X=df.drop(columns=[c for c in drop if c in df.columns],errors="ignore")
    # Avoid datetime objects in generic sklearn transformers; derive year/month first.
    for c in list(X.columns):
        if c.endswith("date") or c in {"review_date","created_at"}:
            dt=pd.to_datetime(X[c],errors="coerce",utc=True)
            X[c+"_year"]=dt.dt.year
            X[c+"_month"]=dt.dt.month
            X=X.drop(columns=[c])
    # Raw text is excluded from this starter; review length features can remain.
    X=X.drop(columns=[c for c in X.columns if c in {"review_text","text","review","content"}],errors="ignore")
    # Remove columns with all missing values.
    X=X.dropna(axis=1,how="all")
    if X.shape[1]==0: raise SystemExit("No usable predictors remain after excluding target, identifiers, and raw text.")
    num_cols=X.select_dtypes(include=["number","bool"]).columns.tolist()
    cat_cols=[c for c in X.columns if c not in num_cols]
    pre=ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),num_cols),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat_cols)
    ],remainder="drop")
    strat=y if task=="classification" and y.value_counts().min()>=2 else None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=args.test_size,random_state=args.seed,stratify=strat)
    if task=="classification":
        models=[
            ("Dummy_majority",DummyClassifier(strategy="most_frequent")),
            ("Logistic_Regression",LogisticRegression(max_iter=1500,class_weight="balanced")),
            ("Random_Forest",RandomForestClassifier(n_estimators=250,min_samples_leaf=3,class_weight="balanced_subsample",random_state=args.seed,n_jobs=-1))
        ]
    else:
        models=[
            ("Dummy_median",DummyRegressor(strategy="median")),
            ("Ridge_Regression",Ridge(alpha=1.0)),
            ("Random_Forest",RandomForestRegressor(n_estimators=250,min_samples_leaf=3,random_state=args.seed,n_jobs=-1))
        ]
    rows=[]; fitted={}
    for name,est in models:
        pipe=Pipeline([("preprocess",pre),("model",est)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Xtr,ytr)
            pred=pipe.predict(Xte)
        if task=="classification":
            row={"model":name,"task":task,"n_train":len(Xtr),"n_test":len(Xte),
                 "accuracy":accuracy_score(yte,pred),
                 "precision":precision_score(yte,pred,zero_division=0),
                 "recall":recall_score(yte,pred,zero_division=0),
                 "f1":f1_score(yte,pred,zero_division=0)}
            try:
                probs=pipe.predict_proba(Xte)[:,1]
                row["roc_auc"]=roc_auc_score(yte,probs) if len(np.unique(yte))==2 else np.nan
            except Exception: row["roc_auc"]=np.nan
            if name!="Dummy_majority":
                fitted[name]=pipe
        else:
            rmse=float(np.sqrt(mean_squared_error(yte,pred)))
            row={"model":name,"task":task,"n_train":len(Xtr),"n_test":len(Xte),
                 "mae":mean_absolute_error(yte,pred),"rmse":rmse}
            if name!="Dummy_median": fitted[name]=pipe
        rows.append(row)
    metrics=pd.DataFrame(rows)
    metrics.to_csv(out/"model_metrics.csv",index=False)
    (out/"classification_report.txt").write_text(
        classification_report(yte, fitted[list(fitted)[0]].predict(Xte), zero_division=0)
        if task=="classification" and fitted else "Not applicable for regression or no fitted candidate.\n",encoding="utf-8")
    # Choose best non-dummy by F1 or lowest RMSE.
    candidates=metrics[~metrics["model"].str.startswith("Dummy")]
    if task=="classification":
        bestrow=candidates.sort_values("f1",ascending=False).iloc[0]
        best_name=bestrow["model"]; best=fitted[best_name]
        pred=best.predict(Xte)
        cm=confusion_matrix(yte,pred,labels=[0,1])
        fig,ax=plt.subplots(figsize=(5,4)); ax.imshow(cm)
        ax.set_title(f"Confusion matrix — {best_name}"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticks([0,1],labels=["0","1"]); ax.set_yticks([0,1],labels=["0","1"])
        for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center")
        fig.tight_layout(); fig.savefig(out/"confusion_matrix.png",dpi=160); plt.close(fig)
    else:
        bestrow=candidates.sort_values("rmse",ascending=True).iloc[0]
        best_name=bestrow["model"]; best=fitted[best_name]
        pred=best.predict(Xte)
        fig,ax=plt.subplots(figsize=(5,4)); ax.scatter(yte,pred,alpha=.35)
        ax.set_title(f"Predicted vs actual — {best_name}"); ax.set_xlabel("Actual rating"); ax.set_ylabel("Predicted rating")
        fig.tight_layout(); fig.savefig(out/"predicted_vs_actual.png",dpi=160); plt.close(fig)
    joblib.dump(best,out/"best_model.joblib")
    save_json({"task":task,"target":target,"best_model":best_name,"seed":args.seed,
               "test_size":args.test_size,"predictor_columns":list(X.columns),
               "n_rows_used":int(len(df)),"warning":"Random split is a baseline only; consider product-grouped or time-based validation."},
              out/"model_run_metadata.json")
    print("Task:",task,"Target:",target)
    print(metrics.to_string(index=False))
    print("Best candidate by",("F1" if task=="classification" else "RMSE"),":",best_name)
    print("Outputs written to outputs/. These metrics are valid only for the supplied dataset and this split strategy.")
if __name__=="__main__": main()
