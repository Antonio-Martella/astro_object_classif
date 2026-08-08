from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classification_metrics(model, X_val, y_val):
    # inizializzo il dizionario degli scores
    scores = {}
    # dato il modello calcolo la y predetta e la y probabilità
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)

    # calcolo delle metriche scelte
    scores["log_loss"] = round(log_loss(y_val, y_prob), 4)
    scores["accuracy"] = round(accuracy_score(y_val, y_pred), 4)
    scores["f1"] = round(f1_score(y_val, y_pred, average="weighted"), 4)
    scores["precision"] = round(precision_score(y_val, y_pred, average="weighted"), 4)
    scores["recall"] = round(recall_score(y_val, y_pred, average="weighted"), 4)
    scores["roc-auc"] = round(roc_auc_score(y_val, y_prob, average="weighted", multi_class="ovo"), 4)

    return scores
