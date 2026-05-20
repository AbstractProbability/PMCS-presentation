import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path

filename_without_ext = Path(__file__).stem

def plot_integrated_bm():
    np.random.seed(42)
    M = 10000
    T = 1.0
    N = 100
    dt = T / N
    t = np.linspace(0.0, T, N + 1)
    
    dW = np.random.normal(0.0, np.sqrt(dt), (M, N))
    W = np.zeros((M, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)
    
    W_integrated = np.zeros((M, N + 1))
    W_integrated[:, 1:] = np.cumsum(W[:, :-1], axis=1) * dt
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    for i in range(50):
        axs[0].plot(t, W_integrated[i, :], color="orange", alpha=0.3, linewidth=0.5)
    axs[0].set_title("Integrated BM Paths")
    axs[0].set_xlabel("Time (t)")
    axs[0].set_ylabel("Y(t)")
    
    final_integrated = W_integrated[:, -1]
    bins_int = np.linspace(-2, 2, 60)
    axs[1].hist(final_integrated, bins_int, density=True, alpha=0.6, color="orange", label="Simulated")
    
    var_integrated = (T**3) / 3.0
    x_val_int = np.linspace(-2, 2, 200)
    pdf_integrated = stats.norm.pdf(x_val_int, loc=0.0, scale=np.sqrt(var_integrated))
    axs[1].plot(x_val_int, pdf_integrated, "r-", lw=2, label="Analytical PDF")
    axs[1].set_title("Integrated BM Distribution at t=T")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"./pictures/{filename_without_ext}.png")

if __name__ == "__main__":
    plot_integrated_bm()