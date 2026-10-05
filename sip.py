"""
DTVaR-Optimization (Weibull + FGM Copula) dengan PSO / DE / Hybrid PSO-DE.
Jalankan:  streamlit run app.py
"""
from datetime import datetime

import pandas as pd
import streamlit as st

import config
from data.loader import load_table, numeric_columns, all_columns, guess_date_column
from data.preprocessing import preprocess, LOGRET_FORMULAS
from distributions.fitting import estimate_all
from dtvar.objective import benchFunctions
from evaluation.performance import run_experiments, best_per_algorithm, backtest_all_runs
from visualization.plots import plot_convergence

st.set_page_config(page_title="Optimasi DTVaR", layout="wide")
st.title("OPTIMISASI UKURAN RISIKO")

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
for key, val in config.DEFAULT_DIST.items():
    st.session_state.setdefault(key, val)
st.session_state.setdefault("show_full", False)


def use_estimation():
    est = st.session_state.get("est")
    if est is None:
        st.session_state["est_warning"] = True
        return
    st.session_state["est_warning"] = False
    st.session_state["k"] = est["weibull"]["k"]
    st.session_state["lamb"] = est["weibull"]["lamb"]
    st.session_state["theta"] = est["fgm"]["theta"]


def toggle_full():
    st.session_state["show_full"] = not st.session_state["show_full"]


def default_index(cols, name, fallback):
    return cols.index(name) if name in cols else min(fallback, len(cols) - 1)


# ======================================================================
# 1. DATA & ESTIMASI
# ======================================================================
st.header("1. Data & Estimasi Parameter")
uploaded = st.file_uploader("Upload data harga mentah (CSV / XLSX)", type=["csv", "xlsx"])

processed = None
if uploaded is not None:
    df = load_table(uploaded)
    num_cols, cols_all = numeric_columns(df), all_columns(df)

    c1, c2 = st.columns(2)
    col_x = c1.selectbox("Kolom harga X (Propana) — marginal Weibull", num_cols,
                         index=default_index(num_cols, "Propana", 0))
    col_y = c2.selectbox("Kolom harga Y (Brent) — marginal Student-t", num_cols,
                         index=default_index(num_cols, "Brent_Crude_Oil", 1))

    NO_DATE = "(tanpa kolom tanggal — data sudah urut lama → baru)"
    date_opts = [NO_DATE] + cols_all
    guess = guess_date_column(cols_all)
    c3, c4, c5 = st.columns(3)
    date_choice = c3.selectbox("Kolom tanggal", date_opts,
                               index=date_opts.index(guess) if guess else 0)
    dayfirst = c4.checkbox("Tanggal berformat dd/mm/yyyy (hari dulu)")
    formula = c5.selectbox("Rumus log return", list(LOGRET_FORMULAS),
                           format_func=lambda key: LOGRET_FORMULAS[key])
    if formula == "literal":
        st.caption("⚠ Rumus ini menghasilkan NaN pada hari harga turun/tetap, "
                   "sehingga banyak baris terbuang.")

    try:
        processed, info = preprocess(df, col_x, col_y,
                                     None if date_choice == NO_DATE else date_choice,
                                     formula, dayfirst)
        st.info(f"Data harga: {info['n_raw']} baris → log return: {info['n_return']} baris → "
                f"X positif (dipakai untuk estimasi & backtesting): **{info['n_final']} baris**")
        with st.expander("Lihat data hasil pemrosesan (X = −log return, X > 0; Y mengikuti)"):
            st.dataframe(processed)
    except Exception as e:
        st.error(f"Pemrosesan data gagal: {e}")

    if processed is not None and st.button("Hitung estimasi parameter"):
        try:
            st.session_state["est"] = estimate_all(processed["X"].to_numpy(),
                                                   processed["Y"].to_numpy())
        except Exception as e:
            st.error(f"Estimasi gagal: {e}")

if st.checkbox("Tampilkan hasil estimasi parameter marginal & FGM"):
    est = st.session_state.get("est")
    if est is None:
        st.info("Belum ada hasil estimasi. Upload data harga lalu klik 'Hitung estimasi parameter'.")
    else:
        w, t, f = est["weibull"], est["student_t"], est["fgm"]
        st.caption(f"Jumlah data berpasangan: {est['n']}")

        m1, m2 = st.columns(2)
        with m1:
            st.markdown("**Marginal X: Weibull**")
            st.dataframe(pd.DataFrame({
                "Parameter": ["Shape (k)", "Loc", "Scale (λ)", "KS statistic", "KS p-value"],
                "Nilai": [w["k"], w["loc"], w["lamb"], w["ks_stat"], w["ks_p"]]}),
                hide_index=True)
        with m2:
            st.markdown("**Marginal Y: Student-t**")
            st.dataframe(pd.DataFrame({
                "Parameter": ["df", "Location (μ)", "Scale (σ)", "KS statistic", "KS p-value"],
                "Nilai": [t["df"], t["loc"], t["scale"], t["ks_stat"], t["ks_p"]]}),
                hide_index=True)

        m3, m4 = st.columns(2)
        with m3:
            st.markdown("**Copula FGM**")
            st.dataframe(pd.DataFrame({
                "Parameter": ["Theta (θ)", "Log-likelihood"],
                "Nilai": [f["theta"], f["loglik"]]}), hide_index=True)
        with m4:
            st.markdown("**Dependensi**")
            st.dataframe(pd.DataFrame({
                "Ukuran": ["Spearman sample", "Kendall sample", "Spearman FGM", "Kendall FGM"],
                "Nilai": [f["spearman_sample"], f["kendall_sample"],
                          f["spearman_fgm"], f["kendall_fgm"]]}), hide_index=True)

# ======================================================================
# 2. PARAMETER DISTRIBUSI
# ======================================================================
st.header("2. Parameter Distribusi")
st.button("Pakai hasil estimasi", on_click=use_estimation)
if st.session_state.get("est_warning"):
    st.warning("Belum ada hasil estimasi. Hitung dulu di bagian 1.")

p1, p2, p3 = st.columns(3)
k = p1.number_input("k (shape Weibull)", key="k", format="%.10f", step=0.01)
lamb = p2.number_input("λ (scale Weibull)", key="lamb", format="%.10f", step=0.001)
theta = p3.number_input("θ (FGM)", key="theta", min_value=-1.0, max_value=1.0,
                        format="%.6f", step=0.01)
p4, p5 = st.columns(2)
alpha = p4.number_input("α", key="alpha", min_value=0.0, max_value=0.9999, format="%.4f", step=0.01)
delta = p5.number_input("δ", key="delta", min_value=0.0, max_value=0.9999, format="%.4f", step=0.01)

# ======================================================================
# 3. PARAMETER ALGORITMA
# ======================================================================
st.header("3. Parameter Algoritma Optimasi")
selected = st.multiselect("Algoritma", config.ALGORITHMS, default=config.ALGORITHMS)

g1, g2 = st.columns(2)
n_runs = int(g1.number_input("Jumlah percobaan", 1, 50, config.DEFAULT_N_RUNS))
seed = int(g2.number_input("Seed dasar (0 = acak)", 0, 10**6, config.DEFAULT_SEED))

algo_params = {}
for name in selected:
    with st.expander(f"Parameter {name}"):
        defaults = config.DEFAULT_ALGO_PARAMS[name]
        cols_ = st.columns(len(defaults))
        algo_params[name] = {}
        for col, (pname, dval) in zip(cols_, defaults.items()):
            label = config.PARAM_LABELS.get(pname, pname)
            if isinstance(dval, int):
                algo_params[name][pname] = int(col.number_input(
                    label, min_value=1, value=dval, step=1, key=f"{name}_{pname}"))
            else:
                algo_params[name][pname] = float(col.number_input(
                    label, min_value=0.0, value=float(dval), format="%.4f",
                    key=f"{name}_{pname}"))

mu_lo, mu_hi = config.DEFAULT_BOUNDS["mu"]
ka_lo, ka_hi = config.DEFAULT_BOUNDS["kappa"]
st.caption(f"Batas variabel otomatis: α1 ∈ [α, 1], δ1 ∈ [δ, 1], "
           f"μ ∈ [{mu_lo}, {mu_hi}], κ ∈ [{ka_lo}, {ka_hi}] (ubah di config.py).")

# ======================================================================
# 4. JALANKAN
# ======================================================================
st.header("4. Jalankan Optimasi")
if st.button("Jalankan", type="primary"):
    if not algo_params:
        st.error("Pilih minimal satu algoritma.")
    elif not (0 < alpha < 1 and 0 < delta < 1):
        st.error("α dan δ harus di antara 0 dan 1.")
    else:
        bench = benchFunctions(alpha, delta, k, lamb, theta)
        prog = st.progress(0.0)
        results, curves = run_experiments(bench, algo_params, n_runs, seed,
                                          progress_cb=prog.progress)
        st.session_state["results"] = results
        st.session_state["curves"] = curves
        st.session_state["run_params"] = dict(alpha=alpha, delta=delta, theta=theta)
        st.session_state["show_full"] = False
        st.session_state.pop("bt_df", None)

# ======================================================================
# 5. BEST DTVaR
# ======================================================================
results = st.session_state.get("results")
if results is not None:
    best = best_per_algorithm(results).reset_index(drop=True)

    st.header("5. Best DTVaR")
    best_cols = ["Algoritma", "Percobaan", "J (fitness)", "dtvar", "alpha1_opt", "delta1_opt",
                 "m_opt", "c_opt", "a_tail", "d_tail", "Waktu (s)"]
    st.dataframe(best[best_cols], hide_index=True)
    top = best.loc[best["J (fitness)"].idxmin()]
    st.success(f"J terkecil: **{top['Algoritma']}** (percobaan #{int(top['Percobaan'])}) — "
               f"J = {top['J (fitness)']:.4e}, DTVaR = {top['dtvar']:.6f}")

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------
    st.subheader("Backtesting DTVaR terbaik")
    st.caption("Violation = return > DTVaR. Uji Kupiec memakai joint significance sebagai p0.")
    st.caption("Data backtesting = data X hasil pemrosesan (X = −log return, hanya X > 0).")
    exp_n = int(st.number_input("Validasi jumlah data (0 = tidak divalidasi, mis. 639)",
                                0, 10**7, config.EXPECTED_N_BACKTEST or 0))
    returns = processed["X"].to_numpy() if processed is not None else None

    rp = st.session_state["run_params"]
    if returns is None:
        st.info("Upload data harga terlebih dahulu untuk backtesting.")
    else:
        try:
            bt_best = backtest_all_runs(best, rp["alpha"], rp["delta"], rp["theta"],
                                        returns, expected_n=exp_n or None)
            st.session_state["bt_df"] = bt_best
            st.dataframe(bt_best, hide_index=True)
        except ValueError as e:
            st.error(str(e))
            returns = None

    # ------------------------------------------------------------------
    # Hasil optimasi lengkap (toggle)
    # ------------------------------------------------------------------
    st.button("Sembunyikan hasil optimasi lengkap" if st.session_state["show_full"]
              else "Hasil optimasi lengkap", on_click=toggle_full)

    if st.session_state["show_full"]:
        st.header("Hasil Optimasi Lengkap")

        st.subheader("Hasil setiap percobaan")
        for name in results["Algoritma"].unique():
            sub = results[results["Algoritma"] == name].reset_index(drop=True)
            best_run = int(sub.loc[sub["J (fitness)"].idxmin(), "Percobaan"])
            st.markdown(f"**{name}** — percobaan terbaik: #{best_run}")
            st.dataframe(sub.drop(columns="Algoritma"), hide_index=True)

        st.subheader("Kurva konvergensi (semua percobaan)")
        st.pyplot(plot_convergence(results, st.session_state["curves"]))

        if returns is not None:
            st.subheader("Backtesting semua percobaan")
            try:
                bt_all = backtest_all_runs(results, rp["alpha"], rp["delta"], rp["theta"],
                                           returns, expected_n=exp_n or None)
                st.dataframe(bt_all, hide_index=True)
                bb = bt_all.loc[bt_all["selisih"].idxmin()]
                st.info(f"Selisih backtesting terkecil: {bb['Algoritma']} percobaan "
                        f"#{int(bb['Percobaan'])} (selisih = {bb['selisih']:.6f})")
            except ValueError as e:
                st.error(str(e))

    # ------------------------------------------------------------------
    # Simpan ke results/
    # ------------------------------------------------------------------
    if st.button("Simpan hasil ke folder results/"):
        config.RESULTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results.to_csv(config.RESULTS_DIR / f"optimasi_{stamp}.csv", index=False)
        if "bt_df" in st.session_state:
            st.session_state["bt_df"].to_csv(
                config.RESULTS_DIR / f"backtesting_best_{stamp}.csv", index=False)
        st.success(f"Tersimpan di {config.RESULTS_DIR} (stamp {stamp})")