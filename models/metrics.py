from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, roc_auc_score

def aggregate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "accuracy": acc,
        "precision": pr,
        "recall": rc,
        "f1": f1,
    }


def test_model(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    perf = aggregate_metrics(y_test, y_pred)
    perf["auc"] = roc_auc_score_aux(estimator, X_test, y_test)
    # perf["classification_report"] = classification_report(y_test, y_pred, zero_division=0)
    return perf

def roc_auc_score_aux(estimator, X, y_true):

    y_proba = estimator.predict_proba(X)[:, 1]

    y_pred = (y_proba >= 0.5).astype(int)

    return roc_auc_score(y_true, y_proba)
    print("\n PERFORMANCE REPORT")
    print(f"Training AUC: {roc_auc_score(y_true, y_proba):.4f} | ACC: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Validation AUC: {roc_auc_score(y_valid, valid_proba):.4f} | ACC: {accuracy_score(y_valid, valid_pred):.4f}")
