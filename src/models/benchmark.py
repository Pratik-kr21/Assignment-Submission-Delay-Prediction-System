"""Multi-model benchmarking and evaluation visualization script."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.data.make_dataset import load_or_create_dataset
from src.features.build_features import create_preprocessor, prepare_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def run_benchmarks(output_dir: str = "docs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    df = load_or_create_dataset()
    X_train, X_test, y_train, y_test = prepare_features(df)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=7, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42, eval_metric="logloss")
    }
    
    results = {}
    fitted_pipelines = {}
    
    for name, clf in models.items():
        preprocessor = create_preprocessor()
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        results[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "y_pred": y_pred.tolist(),
            "y_proba": y_proba.tolist()
        }
        print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
    # 1. ROC Curves Plot
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {res['ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curve Comparison - Assignment Delay Models", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    # 2. Confusion Matrices Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["On-Time (0)", "Late (1)"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
    plt.suptitle("Confusion Matrices across Evaluated Models", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # 3. Model Comparison Bar Chart
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    df_compare = pd.DataFrame({m: [results[model][m] for model in models.keys()] for m in metrics_names}, index=list(models.keys()))
    
    ax = df_compare.plot(kind="bar", figsize=(10, 5), colormap="viridis", width=0.75)
    plt.title("Model Performance Benchmark Comparison", fontsize=13, fontweight="bold")
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11)
    plt.ylim([0, 1.1])
    plt.xticks(rotation=0, fontsize=11)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.grid(axis="y", alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
    plt.tight_layout()
    comp_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    
    # 4. Feature Importance for XGBoost
    xgb_pipe = fitted_pipelines["XGBoost"]
    xgb_clf = xgb_pipe.named_steps["classifier"]
    
    # Extract feature names from preprocessor
    cat_cols = xgb_pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = NUMERIC_FEATURES + list(cat_cols)
    importances = xgb_clf.feature_importances_
    
    feat_df = pd.DataFrame({"Feature": all_feature_names, "Importance": importances}).sort_values(by="Importance", ascending=True)
    
    plt.figure(figsize=(9, 6))
    plt.barh(feat_df["Feature"], feat_df["Importance"], color="#2563eb", edgecolor="#1e40af")
    plt.title("XGBoost Feature Importance (Gini / Gain)", fontsize=13, fontweight="bold")
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fi_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(fi_path, dpi=300)
    plt.close()
    
    # Clean up predictions from JSON summary
    summary_metrics = {name: {k: v for k, v in res.items() if k not in ["y_pred", "y_proba"]} for name, res in results.items()}
    summary_path = os.path.join("docs", "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_metrics, f, indent=4)
        
    print(f"\nAll benchmark plots saved to: {output_dir}")
    print(f"Summary metrics written to: {summary_path}")
    return summary_metrics


if __name__ == "__main__":
    run_benchmarks()
