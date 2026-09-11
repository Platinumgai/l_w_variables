import pandas as pd
import numpy as np
import scipy.stats as st
import warnings

# Suppress warnings from scipy.stats
warnings.filterwarnings("ignore")

def make_pdf(dist, params, size=10000):
    """Generate distribution's PDF formula."""
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    
    if dist.name == 'norm':
        return f"f(x) = (1 / ({scale:.4f} * sqrt(2 * pi))) * exp(-0.5 * ((x - {loc:.4f}) / {scale:.4f})^2)"
    elif dist.name == 'uniform':
        return f"f(x) = 1 / ({scale:.4f}) for {loc:.4f} <= x <= {loc+scale:.4f}, 0 otherwise"
    elif dist.name == 'expon':
        return f"f(x) = (1 / {scale:.4f}) * exp(-(x - {loc:.4f}) / {scale:.4f})"
    elif dist.name == 'gamma':
        a = arg[0]
        return f"f(x) = ((x - {loc:.4f}) / {scale:.4f})^({a:.4f}-1) * exp(-(x - {loc:.4f}) / {scale:.4f}) / ({scale:.4f} * Gamma({a:.4f}))"
    elif dist.name == 'lognorm':
        s = arg[0]
        return f"f(x) = (1 / ((x - {loc:.4f}) * {s:.4f} * sqrt(2*pi))) * exp(- (ln(x - {loc:.4f}) / {scale:.4f})^2 / (2 * {s:.4f}^2))"
    else:
        return f"{dist.name} PDF with loc={loc:.4f}, scale={scale:.4f}, args={arg}"

def best_fit_distribution(data, bins=200):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [        
        st.norm, st.uniform, st.gamma, st.lognorm, st.expon
    ]

    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    for distribution in DISTRIBUTIONS:
        try:
            # Fit distribution
            params = distribution.fit(data)

            # Separate parts of parameters
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            # Calculate fitted PDF and error with fit in distribution
            pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
            sse = np.sum(np.power(y - pdf, 2.0))

            if best_sse > sse > 0:
                best_distribution = distribution
                best_params = params
                best_sse = sse
        except Exception:
            pass

    return best_distribution, best_params

# Load data
df = pd.read_csv('L_W_variables.csv')
cols_of_interest = ['L', 'W', 'F', 'S11']

for col in cols_of_interest:
    data = df[col].dropna().values
    best_dist, best_params = best_fit_distribution(data)
    formula = make_pdf(best_dist, best_params)
    
    # Verification: calculate PDF for a few sample values
    sample_values = np.percentile(data, [25, 50, 75])
    pdf_values = best_dist.pdf(sample_values, *best_params[:-2], loc=best_params[-2], scale=best_params[-1])
    
    print(f"--- Column: {col} ---")
    print(f"Best fit distribution: {best_dist.name}")
    print(f"Formula: {formula}")
    print("Verification (Sample Values -> PDF Value):")
    for val, pdf_val in zip(sample_values, pdf_values):
        print(f"  x = {val:.4f} -> f(x) = {pdf_val:.4f}")
    print()
