import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path

filename_without_ext = Path(__file__).stem

def plot_drift_bm():
    np.random.seed(42)
    M = 10000
    T = 1.0
    N = 100
    dt = T / N
    t = np.linspace(0.0, T, N + 1)
    
    # Drift and volatility parameters
    mu = 2.0
    sigma = 1.0
    
    dW = np.random.normal(0.0, np.sqrt(dt), (M, N))
    W = np.zeros((M, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)
    
    # Calculate Brownian motion with drift
    X = mu * t + sigma * W
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    for i in range(50):
        axs[0].plot(t, X[i, :], color="teal", alpha=0.3, linewidth=0.5)
        
    # Plot the deterministic drift component
    axs[0].plot(t, mu * t, color="red", linestyle="--", label="Drift (mu * t)")
    axs[0].set_title("Brownian Motion with Drift Paths")
    axs[0].set_xlabel("Time (t)")
    axs[0].set_ylabel("X(t)")
    axs[0].legend()
    
    final_X = X[:, -1]
    # Set bins centered around the expected value at T
    min_val = mu * T - 4 * sigma * np.sqrt(T)
    max_val = mu * T + 4 * sigma * np.sqrt(T)
    bins_X = np.linspace(min_val, max_val, 60)
    
    axs[1].hist(final_X, bins=bins_X, density=True, alpha=0.6, color="teal", label="Simulated")
    
    x_val_X = np.linspace(min_val, max_val, 200)
    pdf_X = stats.norm.pdf(x_val_X, loc=mu * T, scale=sigma * np.sqrt(T))
    axs[1].plot(x_val_X, pdf_X, "r-", lw=2, label="Analytical PDF")
    axs[1].set_title("Distribution at t=T")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"./pictures/{filename_without_ext}.png")

if __name__ == "__main__":
    plot_drift_bm()