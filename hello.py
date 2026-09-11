import nbformat as nbf
from pathlib import Path

nb_path = Path("S11_Mathematical_Distribution_Discovery.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

new_cells = []

new_cells.append(nbf.v4.new_markdown_cell(
"""# 11. Important structural check: Is F an independent variable?

Before deriving the final equation, we must check the experimental/design structure.

There is exactly one `F` and one `S11` value for each `(L, W)` pair.

Therefore, for a fixed geometry `(L,W)`, we do **not** have a full frequency sweep.

This matters because we cannot automatically interpret the dataset as:

\[
S_{11}(F)\quad\text{at fixed }L,W
\]

Instead, the dataset is primarily a parameterized surface:

\[
(L,W)\rightarrow(F,S_{11})
\]

We therefore first investigate whether `F` itself can be expressed mathematically in terms of `L` and `W`.
"""))

new_cells.append(nbf.v4.new_code_cell(
"""# Check the number of F values for each (L, W)
f_per_geometry = df.groupby(["L", "W"])["F"].nunique()

print("Maximum number of F values for one (L,W):", f_per_geometry.max())
print("Minimum number of F values for one (L,W):", f_per_geometry.min())

if f_per_geometry.max() == 1:
    print("\\nEach geometry has exactly one F value.")
    print("Therefore F is part of the geometry/design mapping, not a repeated frequency sweep.")
"""))

new_cells.append(nbf.v4.new_markdown_cell(
"""## 12. Discover the mathematical relationship for F

We test simple physically interpretable forms for the frequency:

\[
F\sim \frac{1}{L},\quad
F\sim \frac{1}{W},\quad
F\sim \frac{1}{L+W},\quad
F\sim \frac{1}{\sqrt{LW}}
\]

The purpose is not to use an ML algorithm. These are explicit mathematical hypotheses.
"""))

new_cells.append(nbf.v4.new_code_cell(
"""# Candidate geometric descriptors for F
geom = pd.DataFrame({
    "F": df["F"],
    "1/L": 1/df["L"],
    "1/W": 1/df["W"],
    "1/(L+W)": 1/(df["L"] + df["W"]),
    "1/sqrt(LW)": 1/np.sqrt(df["L"] * df["W"]),
    "L/W": df["L"]/df["W"],
    "W/L": df["W"]/df["L"],
    "1/(L*W)": 1/(df["L"]*df["W"])
})

print("Correlation with F:")
display(geom.corr()["F"].sort_values(key=np.abs, ascending=False).to_frame("Correlation"))
"""))

new_cells.append(nbf.v4.new_markdown_cell(
"""# 13. Nonlinear mathematical candidate families for S11

Now we test explicit mathematical forms.

We use **nonlinear least-squares curve fitting** only as a parameter-estimation method.

We are not using:

- Linear Regression
- Random Forest
- XGBoost
- SVR
- a neural network

The candidate families include:

### A. Additive power model

\[
S_{11}=a+bL^p+cW^q+dF^r
\]

### B. Multiplicative power model

\[
-S_{11}=A L^pW^qF^r
\]

### C. Dimensionless ratio model

\[
S_{11}=a+b(W/L)^p+cF^q
\]

The goal is to compare compact mathematical equations and their residual structure.
"""))

new_cells.append(nbf.v4.new_code_cell(
"""from scipy.optimize import least_squares

L = df["L"].to_numpy(dtype=float)
W = df["W"].to_numpy(dtype=float)
F = df["F"].to_numpy(dtype=float)
Y = df["S11"].to_numpy(dtype=float)

def metrics(y, yp):
    e = y - yp
    ss_res = np.sum(e**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return {
        "R2": 1 - ss_res/ss_tot,
        "RMSE": np.sqrt(np.mean(e**2)),
        "MAE": np.mean(np.abs(e)),
        "MaxAbsError": np.max(np.abs(e))
    }

# A. Additive power model
def model_additive(theta):
    a, b, p, c, q, d, r = theta
    return a + b*L**p + c*W**q + d*F**r

def residual_additive(theta):
    return model_additive(theta) - Y

initial_additive = np.array([-10, -1, 1, -1, 1, -1, 1], dtype=float)
fit_add = least_squares(residual_additive, initial_additive, max_nfev=50000)

pred_add = model_additive(fit_add.x)
m_add = metrics(Y, pred_add)

print("Additive power model:")
print("S11 = a + b L^p + c W^q + d F^r")
print("\\nParameters:")
for name, value in zip(["a","b","p","c","q","d","r"], fit_add.x):
    print(f"{name} = {value:.8g}")

print("\\nMetrics:")
print(m_add)
"""))

new_cells.append(nbf.v4.new_code_cell(
"""# B. Multiplicative power model on -S11
M = -Y

def model_power(theta):
    A, p, q, r = theta
    return A * L**p * W**q * F**r

def residual_power(theta):
    return model_power(theta) - M

# Estimate starting values from log-transformed relation
Z = np.column_stack([np.ones(len(df)), np.log(L), np.log(W), np.log(F)])
coef_log, *_ = np.linalg.lstsq(Z, np.log(M), rcond=None)

initial_power = np.array([np.exp(coef_log[0]), coef_log[1], coef_log[2], coef_log[3]])

fit_power = least_squares(
    residual_power,
    initial_power,
    bounds=([1e-12, -20, -20, -20], [np.inf, 20, 20, 20]),
    max_nfev=50000
)

pred_power = -model_power(fit_power.x)
m_power = metrics(Y, pred_power)

print("Multiplicative power model:")
print("-S11 = A L^p W^q F^r")
print("\\nParameters:")
for name, value in zip(["A","p","q","r"], fit_power.x):
    print(f"{name} = {value:.8g}")

print("\\nMetrics:")
print(m_power)
"""))

new_cells.append(nbf.v4.new_code_cell(
"""# C. Dimensionless ratio model
R = W / L

def model_ratio(theta):
    a, b, p, c, q = theta
    return a + b*R**p + c*F**q

def residual_ratio(theta):
    return model_ratio(theta) - Y

initial_ratio = np.array([-15, -5, 1, -2, 1], dtype=float)

fit_ratio = least_squares(
    residual_ratio,
    initial_ratio,
    max_nfev=50000
)

pred_ratio = model_ratio(fit_ratio.x)
m_ratio = metrics(Y, pred_ratio)

print("Dimensionless ratio model:")
print("S11 = a + b (W/L)^p + c F^q")
print("\\nParameters:")
for name, value in zip(["a","b","p","c","q"], fit_ratio.x):
    print(f"{name} = {value:.8g}")

print("\\nMetrics:")
print(m_ratio)
"""))

new_cells.append(nbf.v4.new_code_cell(
"""comparison = pd.DataFrame([
    {"Model": "Additive power", **m_add},
    {"Model": "Multiplicative power", **m_power},
    {"Model": "Dimensionless ratio", **m_ratio},
])

display(comparison.sort_values("RMSE"))
"""))

new_cells.append(nbf.v4.new_markdown_cell(
"""# 14. Residual-structure test

A compact mathematical equation should not only have a high goodness-of-fit.

We check whether the errors show systematic patterns against:

- L
- W
- F
- predicted S11

If the residuals form a curve or a systematic trend, the equation is missing an important mathematical term.
"""))

new_cells.append(nbf.v4.new_code_cell(
"""model_predictions = {
    "Additive power": pred_add,
    "Multiplicative power": pred_power,
    "Dimensionless ratio": pred_ratio
}

for name, prediction in model_predictions.items():
    residual = Y - prediction

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    axes[0].scatter(L, residual, s=15)
    axes[0].axhline(0, linewidth=1)
    axes[0].set_xlabel("L")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{name}: residual vs L")

    axes[1].scatter(W, residual, s=15)
    axes[1].axhline(0, linewidth=1)
    axes[1].set_xlabel("W")
    axes[1].set_ylabel("Residual")
    axes[1].set_title(f"{name}: residual vs W")

    axes[2].scatter(F, residual, s=15)
    axes[2].axhline(0, linewidth=1)
    axes[2].set_xlabel("F")
    axes[2].set_ylabel("Residual")
    axes[2].set_title(f"{name}: residual vs F")

    plt.tight_layout()
    plt.show()
"""))

new_cells.append(nbf.v4.new_markdown_cell(
"""# 15. Current conclusion

At this stage we are **not selecting the final equation yet**.

The important findings to carry forward are:

1. The dataset contains a complete 26×26 `(L,W)` parameter grid.
2. Each `(L,W)` geometry has only one `F` value.
3. Therefore, this is not a conventional fixed-geometry frequency sweep.
4. We need to study the mathematical mapping:
   \[
   (L,W)\rightarrow F
   \]
   and then:
   \[
   (L,W,F)\rightarrow S_{11}
   \]
5. Power-law and dimensionless mathematical forms are now being tested explicitly.
6. Residual patterns will tell us what mathematical structure is still missing.

### Next stage

The next notebook section will investigate **separable and normalized mathematical forms**, including whether S11 can be represented using a small number of dimensionless groups.

That is the direction most likely to produce a genuinely useful mathematical equation rather than an arbitrary high-degree polynomial.
"""))

nb["cells"].extend(new_cells)

with open(nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Updated notebook:", nb_path)
print("Total cells:", len(nb["cells"]))
