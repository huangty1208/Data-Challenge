
from sklearn.model_selection import train_test_split

# Create features and target arrays
X = dd[['c1', 'c2', 'c3']]
y = dd.price

# Split data into train and test sets
# for classification, stratify=y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=X['c2'], random_state=1121218)



from sklearn.model_selection import cross_val_score

# Initiate a lin_reg
forest = RandomForestRegressor()

# Cross validate
scores = cross_val_score(forest, X, y, cv=7)  # pass scoring option ex. scoring='neg_mean_absolute_error' To see all the names of scorers you can pass to scoring, run sklearn.metrics.SCORERS.keys()

#The output contains scores of each iteration. To return models as wel, use cross_validate

from sklearn.model_selection import cross_validate

# List out desired scorers
scorers = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']

results = cross_validate(forest, X, y, cv=5, scoring=scorers, return_estimator=True)

results.keys()

best_forest = results['estimator'][2]


# stratified splits and do cross-validation at the same time
# useful when you want to perform custom operations in each split 
# shuffles the data before doing the splits 

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)

forest_clf = RandomForestClassifier(random_state=1121218)

# result is a generator object which contains train and test indices for each split.

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Fit a classifier
    forest_clf.fit(X_train, y_train)
    # Score accuracy
    print(forest_clf.score(X_test, y_test))

