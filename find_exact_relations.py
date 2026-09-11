import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv('L_W_variables.csv')
cols = ['L', 'W', 'F', 'S11']

for target in cols:
    features = [c for c in cols if c != target]
    data = df[cols].dropna()
    X = data[features]
    y = data[target]
    
    # Try degree 2 and degree 3
    best_degree = 1
    best_r2 = -1
    best_model = None
    best_poly = None
    
    for degree in [2, 3]:
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        
        y_pred = model.predict(X_poly)
        r2 = r2_score(y, y_pred)
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_poly = poly
            best_degree = degree
            
    print(f"--- Formula for {target} (Polynomial Degree: {best_degree}) ---")
    print(f"R-squared: {best_r2:.4f}")
    
    coefs = best_model.coef_
    intercept = best_model.intercept_
    feature_names = best_poly.get_feature_names_out(features)
    
    terms = []
    for coef, name in zip(coefs, feature_names):
        if name == "1":
            continue
        if abs(coef) > 1e-4:
            # Format the name (e.g., "W^2" instead of "W^2")
            name_clean = name.replace(" ", " * ")
            terms.append(f"({coef:.4f} * {name_clean})")
            
    formula = f"{target} = " + " + ".join(terms) + f" + ({intercept:.4f})"
    print(f"Exact Formula:\n{formula}\n")

