
# get missing rate

dd.isnull().mean().sort_values(ascending=False)


import missingno as msno  

# Plot correlation heatmap of missingness
msno.matrix(dd)

msno.heatmap(dd)

# knniputer

from sklearn.impute import KNNImputer

# Init the transformer
knn_imp = KNNImputer(n_neighbors=3)

# Fit/transform
dd_imputed = knn_imp.fit_transform(dd)

# to find the best k for imputer, plot the graph to compare distribution
n_neighbors = [2, 3, 5, 7]

fig, ax = plt.subplots(figsize=(16, 8))
# Plot the original distribution
sns.kdeplot(dd.col1, label="Original Distribution")
for k in n_neighbors:
    knn_imp = KNNImputer(n_neighbors=k)
    dd_imputed = knn_imp.fit_transform(dd)
    sns.kdeplot(dd_imputed.col1, label=f"Imputed Dist with k={k}")

plt.legend();


# IterativeImputer takes an arbitrary Sklearn estimator and tries to impute missing values by modeling other features as a function of features with missing values.

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Init
ii_imp = IterativeImputer(
    estimator=ExtraTreesRegressor(), max_iter=10, random_state=1121218
)

# Tranform
dd_imputed = ii_imp.fit_transform(dd)

# IterativeImputer predicts not one but max_iter number of possible values for a single missing sample. This has the benefit of treating each missing data point as a random variable




