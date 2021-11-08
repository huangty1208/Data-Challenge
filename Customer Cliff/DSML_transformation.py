
# solving situations where one feature has a much larger variance than others is using scaling 
# StandardScaler() transforms numerical features in the dataset to have a mean of 0 and a variance of 1.

dd.var().round(2)

from sklearn.preprocessing import StandardScaler

# Transform
ss = StandardScaler()
ss.fit(X_train)
X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)


# Init, fit
ss = StandardScaler()
_ = ss.fit(dd[to_scale])

# Transform
dd[to_scale] = pd.DataFrame(ss.transform(dd[to_scale]), columns=to_scale)


# non-linear distribution (lognormal especially)

from sklearn.preprocessing import PowerTransformer

# Init
pt = PowerTransformer()

dd[["co1", "co2"]] = pd.DataFrame(
    pt.fit_transform(dd[["co1", "co2"]]), columns=["co1", "co2"]
)


dd[["co1", "co2"]].var()


# MinMaxScaler does not change the shape of the distribution but gives the distribution an absolute minimum and a maximum value, usually between 0 and 1

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

dd[["co1", "co2"]] = pd.DataFrame(
    mms.fit_transform(dd[["co1", "co2"]]), columns=["co1", "co2"]
)


# use pipeline

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

# Set up the colnames
to_scale = cs1
to_log = cs2
categorical = X.select_dtypes(include="category").columns


scale_pipe = make_pipeline(StandardScaler())
log_pipe = make_pipeline(PowerTransformer())
categorical_pipe = make_pipeline(OneHotEncoder(sparse=False, handle_unknown="ignore"))

transformer = ColumnTransformer(
    transformers=[
        ("scale", scale_pipe, to_scale),
        ("log_transform", log_pipe, to_log),
        ("oh_encode", categorical_pipe, categorical),
    ]
)

# plug this transformer into a Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

knn_pipe = Pipeline([("prep", transformer), ("logistic_reg", LogisticRegression())])




from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# Encode the target
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Fit/predict/score
_ = knn_pipe.fit(X_train, y_train)
preds = knn_pipe.predict_proba(X_test)

roc_auc_score(y_test, preds, multi_class="ovr")

### move to parameter tuning
