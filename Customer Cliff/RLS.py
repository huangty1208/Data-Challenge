
# Generate 'random' data

np.random.seed(0)
X = 2.5 * np.random.randn(100) + 1.5   # Array of 100 values with mean = 1.5, stddev = 2.5
res = 0.5 * np.random.randn(100)       # Generate 100 residual terms
y = 2 + 0.3 * X + res                  # Actual values of Y

# Create pandas dataframe to store our X and y values
df = pd.DataFrame(
    {'X': X,
     'y': y}
)


# Calculate the mean of X and y
xmean = np.mean(X)
ymean = np.mean(y)

# Calculate the terms needed for the numator and denominator of beta
df['xycov'] = (df['X'] - xmean) * (df['y'] - ymean)
df['xvar'] = (df['X'] - xmean)**2

# Calculate beta and alpha
beta = df['xycov'].sum() / df['xvar'].sum()
alpha = ymean - (beta * xmean)
print(f'alpha = {alpha}')
print(f'beta = {beta}')

# use statsmodel for OLS
import statsmodels.formula.api as smf

# Initialise and fit linear regression model using `statsmodels`
model = smf.ols('y ~ X', data=df)
model = model.fit()

model.summary()

# Predict values
x_pred = model.predict()

# Plot regression against actual data
plt.figure(figsize=(12, 6))
plt.plot(X, y, 'o')           # scatter plot showing actual data
plt.plot(X, x_pred, 'r', linewidth=2)   # regression line
plt.xlabel('x')
plt.ylabel('y')
plt.title('xy')

plt.show()


from sklearn.linear_model import LinearRegression

# Build linear regression model using TV and Radio as predictors
# Split data into predictors X and output Y


# Initialise and fit model
lm = LinearRegression()
model = lm.fit(X.reshape(-1, 1), y)

print(f'alpha = {model.intercept_}')
print(f'betas = {model.coef_}')
