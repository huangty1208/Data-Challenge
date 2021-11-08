# get parameters first

forest.get_params()


# define parameter grid for RF

n_estimators = np.arange(100, 2000, step=100)
max_features = ["auto", "sqrt", "log2"]
max_depth = list(np.arange(10, 100, step=10)) + [None]
min_samples_split = np.arange(2, 10, step=2)
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

param_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
}


# n_iter parameter controls how many iterations of random picking of hyperparameter combinations allowed in the search

from sklearn.model_selection import RandomizedSearchCV

forest = RandomForestRegressor()

random_cv = RandomizedSearchCV(
    forest, param_grid, n_iter=100, cv=3, scoring="r2", n_jobs=-1
)

print(random_cv.best_params_)
print( random_cv.best_score_)

# use random RandomSearchCV to narrow down the value range for each hyperparameter so that you can provide a better parameter grid to GridSearchCV


new_params = {
    "n_estimators": [650, 700, 750, 800, 850, 900, 950, 1000],
    "max_features": ['sqrt'],
    "max_depth": [10, 15, 20, 25, 30],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2],
    "bootstrap": [False],
}


from sklearn.model_selection import GridSearchCV

forest = RandomForestRegressor()

grid_cv = GridSearchCV(forest, new_params, n_jobs=-1)

_ = grid_cv.fit(X, y)

print('Best params:\n')
print(grid_cv.best_params_, '\n')



# faster approach :HalvingGridSearch

import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Init the classifier
xgb_cl = xgb.XGBClassifier(objective="binary:logistic", verbose=None, seed=1121218)
%%time

# Create train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1121218, stratify=y
)

# Fit
_ = xgb_cl.fit(X_train, y_train)

# Predict
preds = xgb_cl.predict(X_test)

from sklearn.preprocessing import LabelEncoder

# Encode target and preds for roc_auc

def encode(target_array, predictions):
    le = LabelEncoder()

    y_true = le.fit_transform(target_array)
    preds = le.fit_transform(predictions)

    return y_true, preds

y_test_encoded, preds_encoded = encode(y_test, preds)
roc_auc_score(y_test_encoded, preds_encoded)

# get grid

param_grid = {
    "max_depth": [3, 4, 5, 7],
    "gamma": [0, 0.25, 1],
    "reg_lambda": [0, 1, 10],
    "scale_pos_weight": [1, 3, 5],
    "subsample": [0.8],  # Fix subsample
    "colsample_bytree": [0.5],  # Fix colsample_bytree
}


from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

halving_cv = HalvingGridSearchCV(
    xgb_cl, param_grid, scoring="roc_auc", n_jobs=-1, min_resources="exhaust", factor=3
)

_ = halving_cv.fit(X, y)


# deal with class imbalance

counts = pd.Series(y.flatten()).value_counts()

scale_pos_weight = counts["No"] / counts["Yes"]


param_grid_2 = {
    "max_depth": [3, 4, 5],
    "gamma": [5, 30, 50],
    "learning_rate": [0.01, 0.1, 0.3, 0.5],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [50, 100, 300],
    "scale_pos_weight": [scale_pos_weight],  # Fix scale_pos_weight
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}

from sklearn.model_selection import HalvingRandomSearchCV

halving_random_cv = HalvingRandomSearchCV(
    xgb_cl, param_grid_2, scoring="roc_auc", n_jobs=-1, n_candidates="exhaust", factor=4
)

_ = halving_random_cv.fit(X, y)


