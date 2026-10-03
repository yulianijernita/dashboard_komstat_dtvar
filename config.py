"""Konfigurasi default aplikasi DTVaR-Optimization."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# Parameter distribusi (Weibull + FGM) dan level risiko
DEFAULT_DIST = {
    "k": 1.0563099182562832,
    "lamb": 0.02241097789937711,
    "theta": 0.7389,
    "alpha": 0.90,
    "delta": 0.60,
}

# Batas variabel optimasi [alpha1, delta1, mu, kappa]
# alpha1 in [alpha, 1] dan delta1 in [delta, 1] mengikuti alpha & delta
DEFAULT_BOUNDS = {
    "mu": (0.0, 1.0),
    "kappa": (10.0, 10000.0),
}

# Percobaan
DEFAULT_N_RUNS = 3
DEFAULT_SEED = 0            # 0 = acak
EXPECTED_N_BACKTEST = None  # mis. 639 untuk memvalidasi jumlah data

# Parameter algoritma (nama kunci = argumen konstruktor di optimizers/)
ALGORITHMS = ["PSO", "DE", "Hybrid PSO-DE"]

DEFAULT_ALGO_PARAMS = {
    "PSO": {"pop_size": 50, "max_iter": 500,
            "c1": 2.0, "c2": 2.0, "w": 1.0, "rho": 0.99},
    "DE": {"pop_size": 500, "max_iter": 500, "F": 0.5, "CR": 0.9},
    "Hybrid PSO-DE": {"pop_size": 500, "max_iter": 500, "F": 0.5, "CR": 0.9,
                      "c1": 2.0, "c2": 2.0, "w": 1.0, "rho": 0.99},
}

PARAM_LABELS = {
    "pop_size": "Populasi", "max_iter": "Iterasi",
    "c1": "c1", "c2": "c2", "w": "w (inertia)", "rho": "ρ (peluruhan w)",
    "F": "F", "CR": "CR",
}