"""
DTVaR Risk Analytics Dashboard (Optimized Visual Contrast for Breaches)
"""
from datetime import datetime, date
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from data.loader import load_table, numeric_columns, all_columns, guess_date_column
from data.preprocessing import preprocess, LOGRET_FORMULAS
from distributions.fitting import estimate_all
from dtvar.objective import benchFunctions
from evaluation.performance import run_experiments, best_per_algorithm, backtesting_dtvar

st.set_page_config(page_title="Dashboard Analisis Risiko Portofolio", layout="wide")
# ==============================================================
# SIDEBAR NAVIGASI
# ==============================================================

with st.sidebar:
    st.title("🛡 DTVaR Dashboard")
    st.markdown("---")

    menu = st.radio(
        "Menu",
        [
            "🏠 Home",
            "Optimization Story"
        ]
    )

    st.markdown("---")
    st.caption("Dashboard Analisis Risiko DTVaR")
    
st.markdown("""
<style>
    .status-warning { background-color: #fff3cd; color: #856404; padding: 6px 10px; border-radius: 6px; font-weight: bold; text-align: center; font-size: 12px; margin-bottom: 6px;}
    .block-container { padding-top: 0.6rem; padding-bottom: 0.6rem; }
</style>
""", unsafe_allow_html=True)


if menu == "🏠 Home":
    st.title("🛡 Dashboard Analisis Risiko DTVaR")

    # Inisialisasi Session State
    for key, val in config.DEFAULT_DIST.items():
        st.session_state.setdefault(key, val)
    st.session_state.setdefault("custom_algo_params", {"Hybrid PSO-DE": config.DEFAULT_ALGO_PARAMS["Hybrid PSO-DE"]})
    st.session_state.setdefault("custom_n_runs", config.DEFAULT_N_RUNS)

    @st.dialog("⚙ Pengaturan Teknis Sistem")
    def advanced_settings_dialog():
        st.caption("Konfigurasi sistem otomatis.")
        n_runs = st.number_input("Jumlah Pengujian Iterasi", 1, 20, st.session_state["custom_n_runs"])
        if st.button("Simpan Perubahan", type="primary"):
            st.session_state["custom_n_runs"] = n_runs
            st.success("Pengaturan diperbarui!")
            st.rerun()

    # ======================================================================
    # BARIS KONTROL ATAS & FILTER RENTANG WAKTU
    # ======================================================================
    c_up, c_btn_set, c_btn_run = st.columns([4, 1, 1])
    with c_up:
        uploaded = st.file_uploader("Upload file data harga (CSV / Excel)", type=["csv", "xlsx"], label_visibility="collapsed")
    with c_btn_set:
        if st.button("Setting", use_container_width=True):
            advanced_settings_dialog()
    with c_btn_run:
        run_clicked = st.button("OPTIMASI", type="primary", use_container_width=True)

    raw_processed = None
    if uploaded is not None:
        try:
            df = load_table(uploaded)
            num_cols, cols_all = numeric_columns(df), all_columns(df)
            col_x = num_cols[cols_all.index("Propana") if "Propana" in cols_all else 0]
            col_y = num_cols[cols_all.index("Brent_Crude_Oil") if "Brent_Crude_Oil" in cols_all else min(1, len(num_cols)-1)]
            guess = guess_date_column(cols_all)
            raw_processed, info = preprocess(df, col_x, col_y, guess, list(LOGRET_FORMULAS)[0], True)
            
            # Ambil rentang tanggal minimum dan maksimum dari data
            date_col = pd.to_datetime(raw_processed["Tanggal"] if "Tanggal" in raw_processed.columns else raw_processed.index)
            min_date, max_date = date_col.min().date(), date_col.max().date()
            
            # Widget Filter Rentang Waktu (Dari - Sampai)
            st.markdown("**Setting Date Time**")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                start_date = st.date_input("Mulai Tanggal", value=min_date, min_value=min_date, max_value=max_date)
            with col_d2:
                end_date = st.date_input("Sampai Tanggal", value=max_date, min_value=min_date, max_value=max_date)
                
            # Filter dataframe berdasarkan tanggal yang dipilih user
            mask = (date_col.dt.date >= start_date) & (date_col.dt.date <= end_date)
            processed = raw_processed.loc[mask].copy()
            st.session_state["processed"] = processed
            
            if "est" not in st.session_state or run_clicked:
                st.session_state["est"] = estimate_all(processed["X"].to_numpy(), processed["Y"].to_numpy())
                w = st.session_state["est"]["weibull"]
                st.session_state["k"] = w["k"]
                st.session_state["lamb"] = w["lamb"]
                st.session_state["theta"] = st.session_state["est"]["fgm"]["theta"]
                
        except Exception as e:
            st.error(f"Gagal memproses data: {e}")

    if run_clicked:
        processed_data = st.session_state.get("processed")

        if processed_data is not None:
            scenarios = [
                (0.90, 0.60),
                (0.90, 0.80),
                (0.95, 0.60),
                (0.95, 0.80)
            ]

            all_results = []
            all_curves = {}

            with st.spinner("Menganalisis 4 skenario..."):
                for alpha, delta in scenarios:
                    bench = benchFunctions(
                        alpha,
                        delta,
                        st.session_state["k"],
                        st.session_state["lamb"],
                        st.session_state["theta"]
                    )

                    results_s, curves_s = run_experiments(
                        bench,
                        st.session_state["custom_algo_params"],
                        st.session_state["custom_n_runs"],
                        0
                    )

                    results_s["alpha_scenario"] = alpha
                    results_s["delta_scenario"] = delta
                    results_s["Skenario"] = f"({alpha:.2f}, {delta:.2f})"

                    all_results.append(results_s)
                    all_curves[(alpha, delta)] = curves_s

            results = pd.concat(all_results, ignore_index=True)

            st.session_state["results"] = results
            st.session_state["curves"] = all_curves

            st.session_state["run_params"] = {
                "theta": st.session_state["theta"]
            }
        else:
            st.error("Silakan unggah file data terlebih dahulu.")

    # ======================================================================
    # STRUKTUR 4 GRID UTAMA (2 x 2 LAYOUT — ANTI SCROLL)
    # ======================================================================
    results = st.session_state.get("results")
    processed = st.session_state.get("processed")

    if results is not None and processed is not None:
        # best = best_per_algorithm(results).reset_index(drop=True)
        # top = best.loc[best["J (fitness)"].idxmin()]
        
        # val_dtvar = top['dtvar']
        # val_var = top.get('var')
        # val_tvar = top.get('tvar')
        # val_ctvar = top.get('ctvar')
        # ==============================================================
    # PILIH BEST HASIL OPTIMASI UNTUK SETIAP SKENARIO
    # ==============================================================

        best_scenarios = (
            results.loc[
                results.groupby("Skenario")["J (fitness)"].idxmin()
            ]
            .reset_index(drop=True)
        )

        # ==============================================================
        # BACKTESTING HANYA UNTUK BEST TIAP SKENARIO
        # ==============================================================

        all_backtest = []

        for _, row in best_scenarios.iterrows():

            bt = backtesting_dtvar(
                alpha=row["alpha_scenario"],
                delta=row["delta_scenario"],
                alpha1=row["alpha1_opt"],
                delta1=row["delta1_opt"],
                dtvar=row["dtvar"],
                theta=st.session_state["theta"],
                returns=processed["X"].to_numpy(),
                expected_n=len(processed)
            )

            all_backtest.append({
                "Skenario": row["Skenario"],
                "Algoritma": row["Algoritma"],
                "Percobaan": row["Percobaan"],
                "alpha1_opt": row["alpha1_opt"],
                "delta1_opt": row["delta1_opt"],
                "J (fitness)": row["J (fitness)"],
                "var": row["var"],
                "tvar": row["tvar"],
                "ctvar": row["ctvar"],
                "dtvar": row["dtvar"],
                "join_sig": bt["join_sig"],
                "jumlah_violation": bt["jumlah_violation"],
                "jumlah_data": bt["jumlah_data"],
                "proporsi_error": bt["proporsi_error"],
                "selisih": bt["selisih"]
            })

        backtest_results = pd.DataFrame(all_backtest)

        # ==============================================================
        # PILIH SKENARIO TERBAIK BERDASARKAN BACKTESTING
        # ==============================================================

        top = backtest_results.loc[
            backtest_results["selisih"].idxmin()
        ]

        val_dtvar = top["dtvar"]
        val_var = top["var"]
        val_tvar = top["tvar"]
        val_ctvar = top["ctvar"]

        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        # --- GRID 1 (KIRI-ATAS): Dua Grafik Bersusun (Return Target & Asosiasi) ---
        with row1_col1:
            st.markdown("📈 **Pergerakan Return (Atas: Target, Bawah: Asosiasi)**")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12)
            x_axis = processed["Tanggal"] if "Tanggal" in processed.columns else processed.index
            
            fig.add_trace(go.Scatter(
                x=x_axis, y=processed["X"].to_numpy(), mode='lines', name='Target',
                line=dict(color='royalblue', width=1)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=x_axis, y=processed["Y"].to_numpy(), mode='lines', name='Asosiasi',
                line=dict(color='darkorange', width=1)
            ), row=2, col=1)
            
            fig.update_layout(
                margin=dict(l=5, r=5, t=10, b=5),
                height=200,
                xaxis=dict(showticklabels=False),
                xaxis2=dict(showticklabels=False),
                yaxis=dict(title="Target"),
                yaxis2=dict(title="Asosiasi"),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True, key="grid1_chart")

        # --- GRID 2 (KANAN-ATAS): Status & Kartu Metrik Utama ---
        with row1_col2:
            st.markdown("""
                <div class="status-warning">
                    ⚠️ STATUS KESEHATAN PORTOFOLIO: WASPADA
                </div>
            """, unsafe_allow_html=True)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("DTVaR", f"{val_dtvar:.3f}")
            m2.metric("VaR", f"{val_var:.3f}")
            m3.metric("TVaR", f"{val_tvar:.3f}")
            m4.metric("CTVaR", f"{val_ctvar:.3f}")
            
            st.markdown("<div style='font-size: 11px; color: gray; margin-top: 10px;'>*Catatan: Nilai dihitung berdasarkan batas toleransi risiko optimal sistem.</div>", unsafe_allow_html=True)

        # --- GRID 3 (KIRI-BAWAH): Hasil Backtesting DTVaR 4 Skenario ---
        with row2_col1:
            st.markdown("📋 **Backtesting DTVaR (Hasil Optimasi Algoritma)**")

            best_scenarios = (
                results.loc[
                    results.groupby("Skenario")["J (fitness)"].idxmin()
                ]
                .reset_index(drop=True)
            )

            global_best_scenario = top["Skenario"]

            scenarios_data = []

            for _, row in backtest_results.iterrows():
                scenarios_data.append({
                    "Skenario": row["Skenario"],
                    "α₁": round(row["alpha1_opt"], 4),
                    "δ₁": round(row["delta1_opt"], 4),
                    "DTVaR": round(row["dtvar"], 4),
                    "Joint Sig.": round(row["join_sig"], 4),
                    "Proporsi Error": round(row["proporsi_error"], 4),
                    "Diff": round(row["selisih"], 4)
                })

            df_scenarios = pd.DataFrame(scenarios_data)
            def highlight_best_row(s):
                is_best = s["Skenario"] == global_best_scenario
                return ['background-color: #fff3cd; font-weight: bold;' if is_best else '' for _ in s]

            df_styled = df_scenarios.style.apply(highlight_best_row, axis=1)

            st.dataframe(
                df_scenarios,
                hide_index=True,
                use_container_width=True,
                height=170
            )
            st.caption(f"🏆 Skenario optimal global terpilih: **{global_best_scenario}** (Nilai Fitness J Terendah)")
            
        # --- GRID 4 (KANAN-BAWAH): Grafik Neg Log-Return dengan Kontras Pelanggaran Jelas ---
        with row2_col2:
            sub_c1, sub_c2 = st.columns([2, 2])
            sub_c1.markdown("📉 **Neg Log-Return & Pelanggaran**")

            chosen_metric = sub_c2.selectbox(
                "Pilih Batas",
                ["VaR", "TVaR", "CTVaR", "DTVaR"],
                label_visibility="collapsed",
                key="metric_sel_grid4"
            )

            threshold_dict = {
                "VaR": val_var,
                "TVaR": val_tvar,
                "CTVaR": val_ctvar,
                "DTVaR": val_dtvar
            }

            active_lim = threshold_dict[chosen_metric]

            x_date = pd.to_datetime(processed["Tanggal"] if "Tanggal" in processed.columns else processed.index)
            y_neg_ret = processed["X"].to_numpy()

            fig_breach = go.Figure()

            fig_breach.add_trace(go.Scatter(
                x=x_date,
                y=y_neg_ret,
                mode="lines",
                name="Normal Return",
                line=dict(
                    color="rgba(120, 120, 120, 0.4)",
                    width=1
                )
            ))

            fig_breach.add_hline(
                y=active_lim,
                line_dash="dash",
                line_color="darkorange",
                annotation_text=f"Batas {chosen_metric}",
                annotation_position="top left",
                annotation_font_color="darkorange"
            )

            breach_mask = y_neg_ret >= active_lim

            if np.any(breach_mask):
                fig_breach.add_trace(go.Scatter(
                    x=x_date[breach_mask],
                    y=y_neg_ret[breach_mask],
                    mode="markers",
                    name="Pelanggaran Kritis",
                    marker=dict(
                        color="red",
                        size=9,
                        symbol="x",
                        line=dict(
                            width=2,
                            color="darkred"
                        )
                    )
                ))

            fig_breach.update_layout(
                margin=dict(l=5, r=5, t=10, b=5),
                height=150,
                xaxis=dict(
                    title="Tanggal",
                    type="date",
                    tickformat="%d %b %Y",
                    tickangle=-45,
                    nticks=6
                ),
                yaxis=dict(
                    title=""
                ),
                showlegend=False
            )

            st.plotly_chart(
                fig_breach,
                use_container_width=True,
                key="grid4_breach_chart"
            )
    else:
        st.info("ℹ️ Silakan unggah file data di atas lalu klik tombol **HITUNG** untuk menampilkan analisis risiko.")

elif menu == "Optimization Story":

    st.title("📊 Optimization Story")

    st.markdown(
        """
        <div style="
            background-color: #f8f9fa;
            padding: 18px 22px;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            margin-bottom: 20px;
        ">
            <h3 style="margin: 0 0 8px 0;">Ringkasan Penelitian</h3>
            <p style="margin: 0; font-size: 15px;">
                Analisis risiko Propana Mont Belvieu dilakukan menggunakan
                Dependent Tail Value at Risk (DTVaR) dengan mempertimbangkan
                dependensi terhadap Brent Crude Oil. Optimasi DTVaR dilakukan
                menggunakan PSO, DE, dan Hybrid PSO-DE.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # 1. DATA PENELITIAN
    # ============================================================

    st.subheader("1. Data Penelitian")

    st.markdown(
        """
        Penelitian menggunakan data harga harian **Propana Mont Belvieu**
        sebagai variabel target risiko dan **Brent Crude Oil** sebagai
        variabel asosiasi. Data mencakup periode Januari 2020 hingga
        Desember 2025.
        """
    )

    # Kartu informasi
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Periode", "2020–2025")
    c2.metric("Target Risiko", "Propana")
    c3.metric("Variabel Asosiasi", "Brent Crude Oil")
    c4.metric("Observasi", "639")

    st.markdown("---")

    # ============================================================
    # TRANSFORMASI DATA
    # ============================================================

    st.markdown("### Transformasi Data")

    t1, t2, t3 = st.columns(3)

    with t1:
        st.markdown(
            """
            **1. Harga → Log Return**

            Harga harian ditransformasikan menjadi log return untuk
            memperoleh perubahan relatif harga antarperiode.
            """
        )

    with t2:
        st.markdown(
            """
            **2. Transformasi Risiko**

            Return Propana digunakan sebagai target risiko, sedangkan
            return Brent digunakan sebagai variabel asosiasi.
            """
        )

    with t3:
        st.markdown(
            """
            **3. Penyaringan Data**

            Observasi yang memenuhi kondisi risiko digunakan untuk
            tahap estimasi distribusi marginal dan dependensi.
            """
        )

    # ============================================================
    # GRAFIK
    # ============================================================

# ============================================================
# GRAFIK DATA PENELITIAN
# ============================================================

    st.markdown("### Pergerakan Harga")

    # Membaca data dari assets/EIA
    data_target = pd.read_excel("assets/EIA/crude_oil.xls")
    data_asosiasi = pd.read_excel("assets/EIA/propana.xlsx")

    # Memastikan tipe data tanggal
    data_target["Tanggal"] = pd.to_datetime(data_target["Date"])
    data_asosiasi["Tanggal"] = pd.to_datetime(data_asosiasi["Date"])

    # Sinkronisasi berdasarkan tanggal
    data_gabungan = pd.merge(
        data_target,
        data_asosiasi,
        on="Tanggal",
        how="inner"
    )

    # Periode penelitian
    data_gabungan = data_gabungan[
        (data_gabungan["Tanggal"] >= "2020-01-01") &
        (data_gabungan["Tanggal"] <= "2025-12-31")
    ].reset_index(drop=True)

    # ------------------------------------------------------------
    # Grafik harga Propana
    # ------------------------------------------------------------

    fig_propana = go.Figure()

    fig_propana.add_trace(
        go.Scatter(
            x=data_gabungan["Tanggal"],
            y=data_gabungan["propana"],
            mode="lines",
            name="Propana Mont Belvieu"
        )
    )

    fig_propana.update_layout(
        title="Perkembangan Harga Propana Mont Belvieu Periode 2020–2025",
        xaxis_title="Tahun",
        yaxis_title="USD per galon",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(
        fig_propana,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Grafik harga Brent
    # ------------------------------------------------------------

    fig_brent = go.Figure()

    fig_brent.add_trace(
        go.Scatter(
            x=data_gabungan["Tanggal"],
            y=data_gabungan["crudeoil"],
            mode="lines",
            name="Brent Crude Oil"
        )
    )

    fig_brent.update_layout(
        title="Perkembangan Harga Brent Crude Oil Periode 2020–2025",
        xaxis_title="Tahun",
        yaxis_title="USD/bbl",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(
        fig_brent,
        use_container_width=True
    )

    # ============================================================
    # TRANSFORMASI LOG RETURN
    # ============================================================

    data_gabungan["return_target"] = np.log(
        data_gabungan["propana"] /
        data_gabungan["propana"].shift(1)
    )

    data_gabungan["return_asosiasi"] = np.log(
        data_gabungan["crudeoil"] /
        data_gabungan["crudeoil"].shift(1)
    )

    data_gabungan = data_gabungan.dropna(
        subset=["return_target", "return_asosiasi"]
    ).reset_index(drop=True)

    # Return Propana dikalikan -1 sebagai return risiko
    data_gabungan["return_target"] = -data_gabungan["return_target"]

    # Hanya return risiko Propana yang positif
    data_gabungan = data_gabungan[
        data_gabungan["return_target"] > 0
    ].reset_index(drop=True)

    # ============================================================
    # GRAFIK RETURN
    # ============================================================

    st.markdown("### Pergerakan Return")

    # ------------------------------------------------------------
    # Return risiko Propana
    # ------------------------------------------------------------

    fig_return_propana = go.Figure()

    fig_return_propana.add_trace(
        go.Scatter(
            x=data_gabungan["Tanggal"],
            y=data_gabungan["return_target"],
            mode="lines",
            name="Return Risiko Propana"
        )
    )

    fig_return_propana.update_layout(
        title="Perkembangan Return Risiko Propana Mont Belvieu",
        xaxis_title="Tahun",
        yaxis_title="Log Return",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(
        fig_return_propana,
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Return Brent
    # ------------------------------------------------------------

    fig_return_brent = go.Figure()

    fig_return_brent.add_trace(
        go.Scatter(
            x=data_gabungan["Tanggal"],
            y=data_gabungan["return_asosiasi"],
            mode="lines",
            name="Return Brent Crude Oil"
        )
    )

    fig_return_brent.update_layout(
        title="Perkembangan Return Brent Crude Oil",
        xaxis_title="Tahun",
        yaxis_title="Log Return",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(
        fig_return_brent,
        use_container_width=True
    )

    # Update jumlah observasi hasil preprocessing
    st.metric(
        "Observasi setelah preprocessing",
        len(data_gabungan)
    )
    # ============================================================
    # CATATAN
    # ============================================================

    st.caption(
        "Sumber: Data harga harian EIA, periode Januari 2020–Desember 2025."
    )
