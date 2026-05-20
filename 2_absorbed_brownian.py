import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

filename_without_ext = Path(__file__).stem

def plot_absorbed_bm():
    np.random.seed(42)
    M = 10000
    T = 1.0
    N = 100
    dt = T / N
    t = np.linspace(0.0, T, N + 1)
    
    dW = np.random.normal(0.0, np.sqrt(dt), (M, N))
    W = np.zeros((M, N + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)
    
    a = 0.5
    W_absorbed = np.copy(W)
    for i in range(M):
        idx = np.where(W_absorbed[i, :] >= a)[0]
        if len(idx) > 0:
            first_hit = idx[0]
            W_absorbed[i, first_hit:] = a
            
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    for i in range(50):
        axs[0].plot(t, W_absorbed[i, :], color="blue", alpha=0.3, linewidth=0.5)
    axs[0].axhline(a, color="red", linestyle="--", label="Barrier a=0.5")
    axs[0].set_title("Absorbed BM Paths")
    axs[0].set_xlabel("Time (t)")
    axs[0].set_ylabel("X(t)")
    axs[0].legend()
    
    final_absorbed = W_absorbed[:, -1]
    bins_abs = np.linspace(-3, a, 60)
    axs[1].hist(final_absorbed, bins=bins_abs, density=True, alpha=0.6, color="blue", label="Simulated (Hist)")
    
    x_val_abs = np.linspace(-3, a - 0.01, 200)
    pdf_absorbed = (1.0 / np.sqrt(2 * np.pi * T)) * (np.exp(-x_val_abs**2 / (2 * T)) - np.exp(-(x_val_abs - 2*a)**2 / (2 * T)))
    axs[1].plot(x_val_abs, pdf_absorbed, "r-", lw=2, label="Analytical PDF (x < a)")
    axs[1].set_title("Absorbed BM Distribution at t=T")
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"./pictures/{filename_without_ext}.png")

if __name__ == "__main__":
    plot_absorbed_bm()