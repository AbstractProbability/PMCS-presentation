import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

filename_without_ext = Path(__file__).stem

def plot_reflected_bm():
    np.random.seed(42)
    M = 10000
    T = 1.0
    N = 100
    dt = T / N
    t = np.linspace(0.0, T, N + 1)
    
    dW = np.random.normal(0.0, np.sqrt(dt), (M, N))
    W = np.zeros((M, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)
    
    W_reflected = np.abs(W)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    for i in range(50):
        axs[0].plot(t, W_reflected[i, :], color="green", alpha=0.3, linewidth=0.5)
    axs[0].axhline(0, color="black", linewidth=0.8)
    axs[0].set_title("Reflected BM Paths")
    axs[0].set_xlabel("Time (t)")
    axs[0].set_ylabel("|W(t)|")
    
    final_reflected = W_reflected[:, -1]
    bins_ref = np.linspace(0, 3.5, 60)
    axs[1].hist(final_reflected, bins=bins_ref, density=True, alpha=0.6, color="green", label="Simulated")
    
    x_val_ref = np.linspace(0, 3.5, 200)
    pdf_reflected = np.sqrt(2 / (np.pi * T)) * np.exp(-x_val_ref**2 / (2 * T))
    axs[1].plot(x_val_ref, pdf_reflected, "r-", lw=2, label="Analytical PDF")
    axs[1].set_title("Reflected BM Distribution at t=T")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"./pictures/{filename_without_ext}.png")

if __name__ == "__main__":
    plot_reflected_bm()