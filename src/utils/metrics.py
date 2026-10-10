import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    log_loss,
    classification_report,
    roc_auc_score,
    matthews_corrcoef,
)

from src.utils.validate_type import validate_type



def evaluate_classification_metrics(
    X: pd.DataFrame, 
    y_true: pd.Series, 
    y_pred: pd.Series | None = None,
    model: ImbPipeline | None = None, 
    label_encoder: list[str] | LabelEncoder | dict | None = None,
) -> dict:
    validate_type(
        X=(X, pd.DataFrame),
        y_true=(y_true, pd.Series),
        y_pred=(y_pred, (pd.Series, type(None))),
        model=(model, (ImbPipeline, type(None))),
        label_encoder=(label_encoder, (list, LabelEncoder, dict, type(None))),
    )
    
    if isinstance(label_encoder, LabelEncoder):
        class_names = label_encoder.classes_.tolist()
    elif isinstance(label_encoder, dict):
        class_names = label_encoder.get("classes", [])
    elif isinstance(label_encoder, list):
        class_names = label_encoder
        print("ciaoooooooooo")
    else:
        class_names = None

    scores = {}

    if y_pred is None and model is not None:
        y_pred = model.predict(X)
        #y_prob = model.predict_proba(X)
    elif y_pred is None and model is None:
        raise ValueError(
            "Warning: no model (pipeline) and y_pred neance has been provided, it is not possible to calculate "
            "the metrics"
        )
    
    report = classification_report(y_true, y_pred, output_dict=True)

    # Global Metrics
    scores["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
    scores["balance_accuracy"] = round(balanced_accuracy_score(y_true, y_pred), 4)
    scores["f1_weighted"] = round(report["weighted avg"]["f1-score"], 4)
    scores["f1_macro"] = round(report["macro avg"]["f1-score"], 4)
    scores["precision_weighted"] = round(report["weighted avg"]["precision"], 4)
    scores["precision_macro"] = round(report["macro avg"]["precision"], 4)
    scores["recall_weighted"] = round(report["weighted avg"]["recall"], 4)
    scores["recall_macro"] = round(report["macro avg"]["recall"], 4)

    # Metrics for class
    if class_names is not None: 
        for i, name in enumerate(class_names): 
            if str(i) in report: 
                scores[f"f1_{name}"] = round(report[str(i)]["f1-score"], 4) 
                scores[f"precision_{name}"] = round(report[str(i)]["precision"], 4) 
                scores[f"recall_{name}"] = round(report[str(i)]["recall"], 4) 
            elif name in report:
                scores[f"f1_{name}"] = round(report[name]["f1-score"], 4) 
                scores[f"precision_{name}"] = round(report[name]["precision"], 4) 
                scores[f"recall_{name}"] = round(report[name]["recall"], 4) 
            else: 
                for cls_key in report: 
                    if cls_key not in ["accuracy", "macro avg", "weighted avg"]: 
                        scores[f"f1_class_{cls_key}"] = round(report[cls_key]["f1-score"], 4)


    # other important scores
    #scores["log_loss"] = round(log_loss(y_true, y_prob), 4)
    #scores["roc_auc_weighted"] = round(
    #    roc_auc_score(
    #        y_true,
    #        y_prob,
    #        average="weighted",
    #        multi_class="ovo",
    #    ),
    #    4,
    #)
    scores["mcc"] = round(matthews_corrcoef(y_true, y_pred), 4)

    
    return scores
