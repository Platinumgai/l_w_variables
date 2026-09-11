import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv('L_W_variables.csv')
cols = ['L', 'W', 'F', 'S11']

print("--- Linear Relationships between variables ---")
for target in cols:
    features = [c for c in cols if c != target]
    
    # Drop rows with NaN in any of the relevant columns just in case
    data = df[cols].dropna()
    
    X = data[features]
    y = data[target]
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get predictions and R-squared
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Build formula string
    intercept = model.intercept_
    coefs = model.coef_
    
    terms = [f"({coef:.4f} * {feat})" for coef, feat in zip(coefs, features)]
    formula = f"{target} = " + " + ".join(terms) + f" + ({intercept:.4f})"
    
    print(f"To find {target} when others are given:")
    print(f"Formula: {formula}")
    print(f"R-squared (Accuracy of fit): {r2:.4f}")
    print("-" * 50)
