
# sklearn that support multi-class classification natively

naive_bayes.BernoulliNB
tree.DecisionTreeClassifier
tree.ExtraTreeClassifier
ensemble.ExtraTreesClassifier
naive_bayes.GaussianNB
neighbors.KNeighborsClassifier
svm.LinearSVC (setting multi_class=”crammer_singer”)`
linear_model.LogisticRegression (setting multi_class=”multinomial”)
linear_model.LogisticRegressionCV (setting multi_class=”multinomial”)


# example

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split


# Train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1121218
)

# Fit/predict
etc = ExtraTreesClassifier()
_ = etc.fit(X_train, y_train)
y_pred = etc.predict(X_test)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 5))
cmp = ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred),
    display_labels=["class_1", "class_2", "class_3", "class_4"],
)

cmp.plot(ax=ax)
plt.show();


# Binary classifiers with One-vs-One (OVO) strategy

svm.NuSVC
svm.SVC
gaussian_process.GaussianProcessClassifier (setting multi_class = “one_vs_one”)

# example 

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC

# Don't have to set `multi_class` argument if used with OVOClassifier
ovo = OneVsOneClassifier(estimator=GaussianProcessClassifier())

ovo.fit(X_train, y_train)

### not recommended since each pair of classes require a separate binary classifier, targets with high cardinality may take too long to train



# Binary classifiers with One-vs-Rest (OVR) strategy

ensemble.GradientBoostingClassifier
gaussian_process.GaussianProcessClassifier (setting multi_class = “one_vs_rest”)
svm.LinearSVC (setting multi_class=”ovr”)
linear_model.LogisticRegression (setting multi_class=”ovr”)
linear_model.LogisticRegressionCV (setting multi_class=”ovr”)
linear_model.SGDClassifier
linear_model.Perceptron

### however, this is usually a typical imbalanced classification


# GENERATE ROC_AUC SCORE FOR just once clasee

# Find the index of Ideal class diamonds
idx = np.where(pipeline.classes_ == "class1")[0][0]

# Don't have to set multiclass and average params
roc_auc_score(y_test == "Ideal", y_pred_probs[:, idx])


# score for all classes

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# weight F1
from sklearn.metrics import f1_score

# Weighed F1 across all classes
f1_score(y_test, y_pred, average="weighted")

# F1 score for specific labels with weighted average
f1_score(
...    y_test, y_pred, labels=["c1", "c2"], average="weighted"
...  )


# custom scoring function

from sklearn.metrics import make_scorer

custom_f1 = make_scorer(
    f1_score, greater_is_better=True, average="weighted", labels=["c1", "c2"]
  )

make_scorer(f1_score, average=weighted)

