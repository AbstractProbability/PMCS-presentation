import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path

filename_without_ext = Path(__file__).stem

def plot_standard_bm():
    np.random.seed(42)
    M = 10000
    T = 1.0
    N = 100
    dt = T / N
    t = np.linspace(0.0, T, N + 1)
    
    dW = np.random.normal(0.0, np.sqrt(dt), (M, N))
    W = np.zeros((M, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    for i in range(50):
        axs[0].plot(t, W[i, :], color="gray", alpha=0.3, linewidth=0.5)
    axs[0].set_title("Standard BM Paths")
    axs[0].set_xlabel("Time (t)")
    axs[0].set_ylabel("W(t)")
    
    final_standard = W[:, -1]
    bins_std = np.linspace(-3.5, 3.5, 60)
    axs[1].hist(final_standard, bins=bins_std, density=True, alpha=0.6, color="gray", label="Simulated")
    
    x_val_std = np.linspace(-3.5, 3.5, 200)
    pdf_standard = stats.norm.pdf(x_val_std, loc=0.0, scale=np.sqrt(T))
    axs[1].plot(x_val_std, pdf_standard, "r-", lw=2, label="Analytical PDF")
    axs[1].set_title("Standard BM Distribution at t=T")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"./pictures/{filename_without_ext}.png")

if __name__ == "__main__":
    plot_standard_bm()