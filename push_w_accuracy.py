import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Load data
df = pd.read_csv('L_W_variables.csv')
cols = ['L', 'W', 'F', 'S11']
df_clean = df[cols].dropna()

target = 'W'
features = ['L', 'F', 'S11']
X = df_clean[features]
y = df_clean[target]

print("--- Pushing Accuracy for W ---")

# Try higher degree polynomials
for degree in [6, 7, 8]:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    print(f"Polynomial Degree {degree} Accuracy: {r2*100:.2f}%")
    if r2 >= 0.95:
        break

# Try Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
y_pred_rf = rf.predict(X)
r2_rf = r2_score(y, y_pred_rf)
print(f"Random Forest Accuracy: {r2_rf*100:.2f}%")

# Try XGBoost
xg = xgb.XGBRegressor(n_estimators=100, random_state=42)
xg.fit(X, y)
y_pred_xg = xg.predict(X)
r2_xg = r2_score(y, y_pred_xg)
print(f"XGBoost Accuracy: {r2_xg*100:.2f}%")
