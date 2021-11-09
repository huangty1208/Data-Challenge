
from sklearn.metrics import confusion_matrix

# Generate predictions
y_pred = pipeline.predict(X_test)

# Create the confusion matrix
confusion_matrix(y_test, y_pred)


# plot confusion metrics

import seaborn as sns
from sklearn.metrics import plot_confusion_matrix

fig, ax = plt.subplots(figsize=(8, 5))
plot_confusion_matrix(
    pipeline,  # Fitted estimator (it must have been fit)
    X_test,
    y_test,  # test sets
    ax=ax,
);

# or 

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

def custom_confusion_matrix(y_true, y_pred, display_labels=None):
    """
    A function to plot a custom confusion matrix with
    positive class as the first row and the first column.
    """
    # Create a flipped matrix
    cm = np.flip(confusion_matrix(y_true, y_pred))
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    cmp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    cmp.plot(ax=ax)

custom_confusion_matrix(
  y_test, y_pred, 
  display_labels=["Approved", "Rejected"]
)

# get tn, fp, fn, tp
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# get F1, precision, recall
from sklearn.metrics import f1_score, precision_score, recall_score

# Compute precision for the two pipelines
print("Precision of RandomForests: {}".format(precision_score(y_test, y_pred)))
print(
    "Precision of LogisticRegression: {}".format(
        precision_score(y_test, y_pred_logistic)
    )
)

recall_score(y_test, y_pred)
f1_score(y_test, y_pred)


from sklearn.metrics import classification_report

# Classification report for LogReg
print(classification_report(y_test, y_pred_logistic))

