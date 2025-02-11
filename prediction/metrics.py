from sklearn.metrics import (
    roc_curve,
    auc,
    recall_score,
    precision_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
import pandas as pd


def compute_metrics(df_metrics):
    target_labels = [
        int(col.replace("target_", ""))
        for col in df_metrics.columns
        if "target_" in col
    ]

    metrics = {}
    for label in target_labels:
        preds = df_metrics[f"class_{label}"].values
        targets = df_metrics[f"target_{label}"].values

        roc_auc = roc_auc_score(targets, preds)

        preds = (preds > 0.5).astype(int)
        targets = targets.astype(int)

        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        precision = precision_score(targets, preds)
        recall = recall_score(targets, preds)
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds)

        metrics[label] = [f1, acc, recall, precision, roc_auc, tpr, fpr]

    columns = ["f1", "acc", "recall", "precision", "roc_auc", "tpr", "fpr"]
    df = pd.DataFrame.from_dict(metrics, columns=columns, orient="index")
    return df
