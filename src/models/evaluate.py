#Shared evaluation for all models (grouped cross-validation on the training set)
import pandas as pd
from sklearn.model_selection import PredefinedSplit, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, precision_score, f1_score
from src import config

def evaluate_model(model, features):
  df = pd.read_csv(config.ML_READY_CSV)
  df = df[df["split"] == "train"]
  X, y = df[features], df[config.TARGET_COLUMN]
  #use patient grouped folds
  cv = PredefinedSplit(df["cv_fold"])
  probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
  preds = (probs >= 0.5).astype(int)

  return {"roc_auc": roc_auc_score(y, probs),
        "pr_auc": average_precision_score(y, probs),
        "recall": recall_score(y, preds, zero_division=0),
        "precision": precision_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
         "accuracy": accuracy_score(y, preds), }
  
