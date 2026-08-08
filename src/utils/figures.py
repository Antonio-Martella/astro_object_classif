from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

from configs.paths import DataPathConfig

path_config = DataPathConfig()


def confusion_matrix_imag(y_true: pd.Series, y_pred: pd.Series, save_path: Path):
    with open(path_config.target_le, "rb") as f:
        le = joblib.load(f)

    if any(isinstance(x, int) for x in y_true) and any(isinstance(x, int) for x in y_pred):
        y_true = le.inverse_transform(y_true)
        y_pred = le.inverse_transform(y_pred)
    elif (
        any(isinstance(x, int) for x in y_true)
        and any(not isinstance(x, int) for x in y_pred)
        or any(not isinstance(x, int) for x in y_true)
        and any(isinstance(x, int) for x in y_pred)
    ):
        raise ValueError("Attenzione il formato di y_true e y_pred non coincidono!")

    cm = confusion_matrix(y_true, y_pred, labels=le.classes_)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="coolwarm", xticklabels=le.classes_, yticklabels=le.classes_, fmt="d")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)


def plot_target_distribution(y: pd.Series, save_path: Path, dataset_name: str):
    with open(path_config.target_le, "rb") as f:
        le = joblib.load(f)

    if any(isinstance(x, int) for x in y):
        y = le.inverse_transform(y)

    plt.figure(figsize=(6, 4))

    ax = sns.countplot(x=y)

    for p in ax.patches:
        ax.annotate(
            text=str(int(p.get_height())),
            xy=(p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.title(f"Target {dataset_name.upper()} Distribution")
    plt.tight_layout()
    plt.savefig(save_path)


def log_feature_importance(pipeline, X, y, feature_names, save_path: Path):
    result = permutation_importance(pipeline, X, y, n_repeats=10, random_state=42, n_jobs=-1)

    # Crea il DataFrame
    df_importance = pd.DataFrame({"feature": feature_names, "importance": result.importances_mean}).sort_values(
        by="importance", ascending=False
    )

    plt.figure(figsize=(10, 6))
    plt.barh(df_importance["feature"][:15], df_importance["importance"][:15])
    plt.gca().invert_yaxis()
    plt.savefig(save_path)
    plt.close()


def corr_matrix(X, save_path: Path):
    corr = X.corr(method="spearman")

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True)
    plt.savefig(save_path)
