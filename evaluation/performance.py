"""Evaluasi kinerja: menjalankan percobaan berulang, ringkasan terbaik, dan backtesting DTVaR."""
import time

import numpy as np
import pandas as pd

from copulas.fgm import cop_fgm
from optimizers.pso import PSO
from optimizers.de import DE
from optimizers.hybrid_pso_de import Hybrid_PSO_DE
from evaluation.statistical_test import kupiec_pof

OPTIMIZERS = {"PSO": PSO, "DE": DE, "Hybrid PSO-DE": Hybrid_PSO_DE}


# ----------------------------------------------------------------------
# Percobaan berulang
# ----------------------------------------------------------------------
def build_optimizer(name, bench, params):
    bounds = (bench.lower_bound, bench.upper_bound)
    return OPTIMIZERS[name](bench.objective_function, bench.objective_batch,
                            bounds, **params)


def run_experiments(bench, algo_params, n_runs=3, seed=0, progress_cb=None):
    """
    algo_params: {"PSO": {...}, "DE": {...}, ...} (hanya algoritma yang dipilih).
    Return: (DataFrame semua percobaan, dict kurva konvergensi {(algo, run): [...]}).
    """
    rows, curves = [], {}
    total, done = len(algo_params) * n_runs, 0

    for name, params in algo_params.items():
        for run in range(1, n_runs + 1):
            if seed:
                np.random.seed(int(seed) + run)
            t0 = time.perf_counter()
            x_best, score, conv, _ = build_optimizer(name, bench, params).optimize()
            elapsed = time.perf_counter() - t0

            rows.append({"Algoritma": name, "Percobaan": run,
                         "J (fitness)": float(score),
                         **bench.calculate_metrics(x_best),
                         "Waktu (s)": elapsed})
            curves[(name, run)] = conv

            done += 1
            if progress_cb:
                progress_cb(done / total)

    return pd.DataFrame(rows), curves


def best_per_algorithm(results):
    """Baris terbaik (J terkecil) untuk setiap algoritma."""
    return results.loc[results.groupby("Algoritma")["J (fitness)"].idxmin()]


# ----------------------------------------------------------------------
# Backtesting
# ----------------------------------------------------------------------
def backtesting_dtvar(alpha, delta, alpha1, delta1, dtvar, theta, returns,
                      expected_n=None):
    """
    expected_n : jika diisi (mis. 639), jumlah data return divalidasi.
                 Jika None, seluruh data yang diberikan digunakan.
    """
    returns = np.asarray(returns, dtype=float)
    jumlah_data = len(returns)

    if expected_n is not None and jumlah_data != expected_n:
        raise ValueError(
            f"Jumlah data return = {jumlah_data}, seharusnya {expected_n}.")

    C_alpha1_delta1 = cop_fgm(alpha1, delta1, theta)
    C_alpha_delta1 = cop_fgm(alpha, delta1, theta)
    C_alpha1_delta = cop_fgm(alpha1, delta, theta)
    C_alpha_delta = cop_fgm(alpha, delta, theta)

    # Joint significance
    join_sig = (C_alpha1_delta1 - C_alpha_delta1
                - C_alpha1_delta + C_alpha_delta)

    # Violation: return > DTVaR
    jumlah_violation = np.sum(returns > dtvar)
    proporsi_error = jumlah_violation / jumlah_data
    selisih = abs(join_sig - proporsi_error)

    return {
        'C_alpha1_delta1': C_alpha1_delta1,
        'C_alpha_delta1': C_alpha_delta1,
        'C_alpha1_delta': C_alpha1_delta,
        'C_alpha_delta': C_alpha_delta,
        'join_sig': join_sig,
        'jumlah_violation': int(jumlah_violation),
        'jumlah_data': int(jumlah_data),
        'proporsi_error': proporsi_error,
        'selisih': selisih,
    }


def backtest_all_runs(results, alpha, delta, theta, returns, expected_n=None):
    """Backtesting untuk setiap (algoritma, percobaan) + uji Kupiec (POF)."""
    rows = []
    for _, r in results.iterrows():
        bt = backtesting_dtvar(alpha, delta, r["alpha1_opt"], r["delta1_opt"],
                               r["dtvar"], theta, returns, expected_n=expected_n)
        lr, p_val = kupiec_pof(bt["jumlah_violation"], bt["jumlah_data"], bt["join_sig"])
        rows.append({"Algoritma": r["Algoritma"], "Percobaan": r["Percobaan"],
                     "DTVaR": r["dtvar"], **bt,
                     "LR_kupiec": lr, "p_value_kupiec": p_val})
    return pd.DataFrame(rows)