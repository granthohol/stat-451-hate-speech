"""
Shared evaluation so every model is reported the same way.

We report per class precision, recall, and F1, macro F1, and a confusion matrix.
We do not report accuracy: predicting "neutral" for everything already gets
about 82%.

Usage:
    from evaluate import evaluate
    results = evaluate(pipe, validation["text_clean"], validation["label"], name="Logistic regression")
    results["macro_f1"]           # float
    results["per_class"]          # DataFrame: precision, recall, f1, support per class
    results["confusion_matrix"]   # DataFrame: rows are true labels, columns are predictions
    results["y_pred"]             # predictions, handy for looking at mistakes
"""

import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_recall_fscore_support

from load_cad import CLASSES


def evaluate(model, X, y, name: str = None, verbose: bool = True, plot: bool = False) -> dict:
    """Evaluate a fitted model (or pipeline) on X, y.

    Prints the results unless verbose=False, draws the confusion matrix if
    plot=True, and returns a dict with macro_f1, per_class, confusion_matrix,
    and y_pred.
    """
    y_true = np.asarray(y)
    y_pred = np.asarray(model.predict(X))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    per_class = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support}, index=CLASSES
    )
    macro_f1 = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=CLASSES),
        index=pd.Index(CLASSES, name="true"),
        columns=pd.Index(CLASSES, name="predicted"),
    )

    if verbose:
        if name:
            print(f"== {name} ==")
        print(per_class.round(3).to_string())
        print(f"\nmacro F1: {macro_f1:.3f}\n")
        print(cm.to_string())
        print()

    if plot:
        disp = ConfusionMatrixDisplay(cm.to_numpy(), display_labels=CLASSES)
        disp.plot(cmap="Blues", colorbar=False)
        disp.ax_.set_title(name or "Confusion matrix")

    return {"macro_f1": macro_f1, "per_class": per_class, "confusion_matrix": cm, "y_pred": y_pred}
