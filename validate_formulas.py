import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load data
df = pd.read_csv('L_W_variables.csv')
cols = ['L', 'W', 'F', 'S11']
df_clean = df[cols].dropna()

# Create a DataFrame for output
output_df = df_clean.copy()
output_df.rename(columns={'L': 'L_actual', 'W': 'W_actual', 'F': 'F_actual', 'S11': 'S11_actual'}, inplace=True)

# Train and predict for each column using degree 5
for target in cols:
    features = [c for c in cols if c != target]
    
    X = df_clean[features]
    y = df_clean[target]
    
    # 5th-degree polynomial
    poly = PolynomialFeatures(degree=5)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict
    y_pred = model.predict(X_poly)
    
    # Save predicted values
    output_df[f'{target}_predicted'] = y_pred

# Reorder columns for easier comparison
final_cols = []
for c in cols:
    final_cols.extend([f'{c}_actual', f'{c}_predicted'])
    
output_df = output_df[final_cols]

# Save to output.csv
output_df.to_csv('output.csv', index=False)
print("Successfully generated output.csv with actual and predicted values.")
