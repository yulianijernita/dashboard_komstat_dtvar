"""Visualisasi hasil optimasi."""
import matplotlib.pyplot as plt


def plot_convergence(results, curves):
    """Satu panel per algoritma, satu garis per percobaan (sumbu-y log)."""
    algos = list(results["Algoritma"].unique())
    fig, axes = plt.subplots(1, len(algos), figsize=(5 * len(algos), 3.5), squeeze=False)

    for ax, name in zip(axes[0], algos):
        for (n, run), conv in curves.items():
            if n == name:
                ax.plot(conv, label=f"Percobaan {run}")
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("Iterasi")
        ax.set_ylabel("J")
        ax.legend(fontsize=7)

    fig.tight_layout()
    return fig