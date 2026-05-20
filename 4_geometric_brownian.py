import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path

filename_without_ext = Path(__file__).stem

def plot_geometric_bm():
    np.random.seed(42)
    M = 10000
    T = 1.0
    N = 100
    dt = T / N
    t = np.linspace(0.0, T, N + 1)
    
    dW = np.random.normal(0.0, np.sqrt(dt), (M, N))
    W = np.zeros((M, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)
    
    mu = 1.0
    sigma = 0.5
    S0 = 1.0
    W_gbm = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    for i in range(50):
        axs[0].plot(t, W_gbm[i, :], color="purple", alpha=0.3, linewidth=0.5)
    axs[0].set_title("Geometric BM Paths")
    axs[0].set_xlabel("Time (t)")
    axs[0].set_ylabel("S(t)")
    
    final_gbm = W_gbm[:, -1]
    bins_gbm = np.linspace(0, 8, 60)
    axs[1].hist(final_gbm, bins=bins_gbm, density=True, alpha=0.6, color="purple", label="Simulated")
    
    x_val_gbm = np.linspace(0.01, 8, 200)
    pdf_gbm = stats.lognorm.pdf(x_val_gbm, s=sigma * np.sqrt(T), scale=np.exp(np.log(S0) + (mu - 0.5 * sigma**2) * T))
    axs[1].plot(x_val_gbm, pdf_gbm, "r-", lw=2, label="Analytical PDF")
    axs[1].set_title("Geometric BM Distribution at t=T")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"./pictures/{filename_without_ext}.png")

if __name__ == "__main__":
    plot_geometric_bm()