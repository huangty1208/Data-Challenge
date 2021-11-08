

### RFE Recursive Feature Elimination ( based on coefficient or feature importance )
###  RFE trains the given model on the full dataset every time it drops a feature 


from sklearn.feature_selection import RFE

# Init the transformer
rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=10)

# Fit to the training data
_ = rfe.fit(X_train_std, y_train)

# subset data
X_train.loc[:, rfe.support_]


# RFE provides step parameter that lets us drop an arbitrary number of features in each iteration instead of one

rfe = RFE(estimator=RandomForestRegressor(), 
          n_features_to_select=10, step=10)


# or use CV as well 
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression

# Init, fit
rfecv = RFECV(
    estimator=LinearRegression(),
    min_features_to_select=5,
    step=5,
    n_jobs=-1,
    scoring="r2",
    cv=5,
)

_ = rfecv.fit(X_train_std, y_train)
X_train.columns[rfecv.support_]

lr_mask = rfecv.support_

# ensemble feature selection

%%time

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

# Gradient Boosting Regressor
rfecv = RFECV(
    estimator=GradientBoostingRegressor(),
    cv=3,
    scoring="r2",
    n_jobs=-1,
    min_features_to_select=1,
)
_ = rfecv.fit(X_train, y_train)

gb_mask = rfecv.support_


votes = np.sum([gb_mask, lr_mask], axis=0)

final_mask = votes == 2



### Pairwise - feature correlation 


matrix = houses.corr()

plt.figure(figsize=(16,12))

_ = sns.heatmap(matrix)


plt.figure(figsize=(16,12))

# Create a custom diverging palette
cmap = sns.diverging_palette(250, 15, s=75, l=40,
                             n=9, center="light", as_cmap=True)

_ = sns.heatmap(matrix, center=0, annot=True, 
                fmt='.2f', square=True, cmap=cmap)



# Create a mask
plt.figure(figsize=(16,12))

mask = np.triu(np.ones_like(matrix, dtype=bool))

sns.heatmap(matrix, mask=mask, center=0, annot=True,
             fmt='.2f', square=True, cmap=cmap)

# correlation plot function

def correlation_matrix(df: pd.DataFrame):
    """
    A function to calculate and plot
    correlation matrix of a DataFrame.
    """
    # Create the matrix
    matrix = df.corr()
    
    # Create cmap
    cmap = sns.diverging_palette(250, 15, s=75, l=40,
                             n=9, center="light", as_cmap=True)
    # Create a mask
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    
    # Make figsize bigger
    fig, ax = plt.subplots(figsize=(16,12))
    
    # Plot the matrix
    _ = sns.heatmap(matrix, mask=mask, center=0, annot=True,
             fmt='.2f', square=True, cmap=cmap, ax=ax)

# selection based on correaltion    
    

def identify_correlated(df, threshold):
    """
    A function to identify highly correlated features.
    """
    # Compute correlation matrix with absolute values
    matrix = df.corr().abs()
    
    # Create a boolean mask
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    
    # Subset the matrix
    reduced_matrix = matrix.mask(mask)
    
    # Find cols that meet the threshold
    to_drop = [c for c in reduced_matrix.columns if \
              any(reduced_matrix[c] > threshold)]
    
    return to_drop
  
# Drop the cols
ansur_reduced = ansur.drop(to_drop, axis=1)


# Build feature/target arrays
X, y = ansur_reduced.drop('Weightlbs', axis=1), ansur_reduced['Weightlbs']

# Generate train/test sets
X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.3, random_state=1121218
                      )

%%time
# Init, fit, score
forest = RandomForestRegressor()
_ = forest.fit(X_train, y_train)

print(f"Training score: {forest.score(X_train, y_train)}")

print(f"Test score: {forest.score(X_test, y_test)}")




### VarianceThreshold Estimator

from sklearn.feature_selection import VarianceThreshold

# need to variance decide threshold first
vt = VarianceThreshold(threshold=0.005)   # or 0.003

# numeric data only
ansur_male_num = ansur_male.select_dtypes(include='number')

# need to normalizing all features by dividing them by their mean
normalized_df = ansur_male_num / ansur_male_num.mean()

# ensure the same variance
normalized_df.var()

_ = vt.fit(ansur_male_num)

mask = vt.get_support()
ansur_male_num = ansur_male_num.loc[:, mask]


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1121218)

# Init, fit, score
forest = RandomForestRegressor(random_state=1121218)

_ = forest.fit(X_train, y_train)

# Training Score
print(f"Training Score: {forest.score(X_train, y_train)}")


