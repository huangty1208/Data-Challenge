
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=.3, random_state=1121218)


from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error

# Initilize, fit, predict
lin_reg = LinearRegression()
_ = lin_reg.fit(X_train, y_train)
preds = lin_reg.predict(X_test)

print("Training score:", lin_reg.score(X_train, y_train))
print("Testing score:", lin_reg.score(X_test, y_test))
print("MAE of Linear Regression:", mean_absolute_error(y_test, preds), '\n')

# ridge
ridge = Ridge(alpha=0.5)
_ = ridge.fit(X_train, y_train)
preds = ridge.predict(X_test)

print("Training score:", ridge.score(X_train, y_train))
print("Testing score:", ridge.score(X_test, y_test))
print("MAE of Ridge Regression:", mean_absolute_error(y_test, preds), '\n')

# use ridgeCV for alpha search
from sklearn.linear_model import RidgeCV

ridge = RidgeCV(alphas=np.arange(1, 100, 5), scoring='r2', cv=10)
_ = ridge.fit(X, y)

# best alpha
ridge.alpha_



# Now Lasso

from sklearn.linear_model import Lasso, LassoCV

lasso = LassoCV(alphas=np.arange(0.000000001, 1, 0.05), cv=10)
_ = lasso.fit(X, y)

print('Best alpha:', lasso.alpha_)


# check coefficient value

fig, ax = plt.subplots(figsize=(12, 10))

ax.plot(X_train.columns, lasso.coef_, color='#111111')

plt.setp(ax.get_xticklabels(), rotation=90);


