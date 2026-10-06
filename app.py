"""
DTVaR Risk Analytics Dashboard
"""

from datetime import datetime, date
from pathlib import Path

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


# ==============================================================
# CONFIGURATION
# ==============================================================

st.set_page_config(
    page_title="Dashboard Analisis Risiko Portofolio",
    layout="wide"
)

OPTIMIZATION_FILE = Path("assets/EIA/Hasil_Optimasi_DTVaR_Rill.xlsx")

# Skenario yang digunakan untuk grafik konvergensi
CONVERGENCE_SCENARIO = "0.90, 0.8"


# ==============================================================
# HELPER — LOAD HASIL OPTIMASI
# ==============================================================

@st.cache_data
def load_optimization_file():

    if not OPTIMIZATION_FILE.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {OPTIMIZATION_FILE}"
        )

    return pd.read_excel(
        OPTIMIZATION_FILE,
        sheet_name=None
    )


def find_sheet(sheet_data, aliases):

    """
    Mencari sheet berdasarkan nama.
    Tidak sensitif terhadap huruf besar/kecil.
    """

    normalized = {
        str(k).strip().lower(): k
        for k in sheet_data.keys()
    }

    for alias in aliases:

        alias = alias.lower()

        if alias in normalized:
            return sheet_data[normalized[alias]].copy()

    # pencarian sebagian nama
    for name, original_name in normalized.items():

        for alias in aliases:

            if alias.lower() in name:
                return sheet_data[original_name].copy()

    return None


def find_column(df, aliases):

    if df is None:
        return None

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    # 1. Prioritaskan kecocokan persis
    for alias in aliases:

        alias_norm = str(alias).strip().lower()

        if alias_norm in normalized:
            return normalized[alias_norm]

    # 2. Pencarian sebagian hanya jika benar-benar diperlukan
    for col, original_col in normalized.items():

        for alias in aliases:

            alias_norm = str(alias).strip().lower()

            # Hindari VaR terbaca sebagai bagian dari DTVaR/CTVaR/TVaR
            if alias_norm in {"var", "tvar", "ctvar", "dtvar"}:
                continue

            if alias_norm in col:
                return original_col

    return None


def format_scenario(value):

    """
    Menyamakan format skenario.
    """

    if pd.isna(value):
        return value

    if isinstance(value, str):
        return value

    try:
        return f"{float(value):.2f}"
    except:
        return str(value)


# ==============================================================
# SIDEBAR NAVIGASI
# ==============================================================

with st.sidebar:
    st.markdown("---")

    menu = st.radio(
        "Menu",
        [
            "Main",
            "DTVaR Optimization Story"
        ]
    )

    st.markdown("---")
    st.caption("Dashboard Analisis Risiko DTVaR")


# ==============================================================
# STYLE
# ==============================================================

st.markdown(
    """
    <style>

    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 6px 10px;
        border-radius: 6px;
        font-weight: bold;
        text-align: center;
        font-size: 12px;
        margin-bottom: 6px;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 0.6rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ################################################################
# ################################################################
# MAIN
# ################################################################
# ################################################################

if menu == "Main":

    st.title("Dashboard Optimasi Ukuran Risiko")

    # ============================================================
    # SESSION STATE
    # ============================================================

    for key, val in config.DEFAULT_DIST.items():
        st.session_state.setdefault(key, val)

    st.session_state.setdefault(
        "custom_algo_params",
        {
            "Hybrid PSO-DE":
                config.DEFAULT_ALGO_PARAMS["Hybrid PSO-DE"]
        }
    )

    st.session_state.setdefault(
        "custom_n_runs",
        config.DEFAULT_N_RUNS
    )

    # ============================================================
    # SETTINGS
    # ============================================================

    @st.dialog("⚙ Pengaturan Teknis Sistem")
    def advanced_settings_dialog():

        st.caption("Konfigurasi sistem otomatis.")

        n_runs = st.number_input(
            "Jumlah Pengujian Iterasi",
            1,
            20,
            st.session_state["custom_n_runs"]
        )

        if st.button(
            "Simpan Perubahan",
            type="primary"
        ):

            st.session_state["custom_n_runs"] = n_runs

            st.success(
                "Pengaturan diperbarui!"
            )

            st.rerun()

    # ============================================================
    # UPLOAD DATA
    # ============================================================

    c_up, c_btn_set, c_btn_run = st.columns(
        [4, 1, 1]
    )

    with c_up:

        uploaded = st.file_uploader(
            "Upload file data harga (CSV / Excel)",
            type=["csv", "xlsx"],
            label_visibility="collapsed"
        )

    with c_btn_run:

        run_clicked = st.button(
            "OPTIMASI",
            type="primary",
            use_container_width=True
        )

    with c_btn_set:

        if st.button(
            "Setting",
            use_container_width=True
        ):

            advanced_settings_dialog()

    # ============================================================
    # PREPROCESSING
    # ============================================================

    raw_processed = None

    if uploaded is not None:

        try:

            df = load_table(uploaded)

            num_cols = numeric_columns(df)
            cols_all = all_columns(df)

            col_x = num_cols[
                cols_all.index("Propana")
                if "Propana" in cols_all
                else 0
            ]

            col_y = num_cols[
                cols_all.index("Brent_Crude_Oil")
                if "Brent_Crude_Oil" in cols_all
                else min(
                    1,
                    len(num_cols) - 1
                )
            ]

            guess = guess_date_column(
                cols_all
            )

            raw_processed, info = preprocess(
                df,
                col_x,
                col_y,
                guess,
                list(LOGRET_FORMULAS)[0],
                True
            )

            # ====================================================
            # DATE FILTER
            # ====================================================

            date_col = pd.to_datetime(
                raw_processed["Tanggal"]
                if "Tanggal" in raw_processed.columns
                else raw_processed.index
            )

            min_date = date_col.min().date()
            max_date = date_col.max().date()

            st.markdown(
                "**Setting Date Time**"
            )

            col_d1, col_d2 = st.columns(2)

            with col_d1:

                start_date = st.date_input(
                    "Mulai Tanggal",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )

            with col_d2:

                end_date = st.date_input(
                    "Sampai Tanggal",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )

            mask = (
                (date_col.dt.date >= start_date)
                &
                (date_col.dt.date <= end_date)
            )

            processed = raw_processed.loc[
                mask
            ].copy()

            st.session_state[
                "processed"
            ] = processed

            # ====================================================
            # ESTIMATION
            # ====================================================

            if (
                "est" not in st.session_state
                or run_clicked
            ):

                st.session_state["est"] = estimate_all(
                    processed["X"].to_numpy(),
                    processed["Y"].to_numpy()
                )

                w = st.session_state[
                    "est"
                ]["weibull"]

                st.session_state["k"] = w["k"]
                st.session_state["lamb"] = w["lamb"]

                st.session_state["theta"] = (
                    st.session_state["est"]
                    ["fgm"]["theta"]
                )

        except Exception as e:

            st.error(
                f"Gagal memproses data: {e}"
            )

    # ============================================================
    # RUN OPTIMIZATION
    # ============================================================

    if run_clicked:

        processed_data = st.session_state.get(
            "processed"
        )

        if processed_data is not None:

            scenarios = [
                (0.90, 0.60),
                (0.90, 0.80),
                (0.95, 0.60),
                (0.95, 0.80)
            ]

            all_results = []
            all_curves = {}

            with st.spinner(
                "Menganalisis 4 skenario..."
            ):

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
                        st.session_state[
                            "custom_algo_params"
                        ],
                        st.session_state[
                            "custom_n_runs"
                        ],
                        0
                    )

                    results_s[
                        "alpha_scenario"
                    ] = alpha

                    results_s[
                        "delta_scenario"
                    ] = delta

                    results_s[
                        "Skenario"
                    ] = (
                        f"({alpha:.2f}, {delta:.2f})"
                    )

                    all_results.append(
                        results_s
                    )

                    all_curves[
                        (alpha, delta)
                    ] = curves_s

            results = pd.concat(
                all_results,
                ignore_index=True
            )

            st.session_state[
                "results"
            ] = results

            st.session_state[
                "curves"
            ] = all_curves

            st.session_state[
                "run_params"
            ] = {
                "theta":
                    st.session_state["theta"]
            }

        else:

            st.error(
                "Silakan unggah file data terlebih dahulu."
            )

    # ============================================================
    # MAIN DASHBOARD RESULT
    # ============================================================

    results = st.session_state.get(
        "results"
    )

    processed = st.session_state.get(
        "processed"
    )

    if results is not None and processed is not None:

        # ========================================================
        # BEST RESULT PER SCENARIO
        # ========================================================

        best_scenarios = (
            results.loc[
                results.groupby(
                    "Skenario"
                )["J (fitness)"].idxmin()
            ]
            .reset_index(drop=True)
        )

        # ========================================================
        # BACKTESTING BEST RESULT
        # ========================================================

        all_backtest = []

        for _, row in best_scenarios.iterrows():

            bt = backtesting_dtvar(
                alpha=row[
                    "alpha_scenario"
                ],
                delta=row[
                    "delta_scenario"
                ],
                alpha1=row[
                    "alpha1_opt"
                ],
                delta1=row[
                    "delta1_opt"
                ],
                dtvar=row["dtvar"],
                theta=st.session_state[
                    "theta"
                ],
                returns=processed[
                    "X"
                ].to_numpy(),
                expected_n=len(processed)
            )

            all_backtest.append({

                "Skenario":
                    row["Skenario"],

                "Algoritma":
                    row["Algoritma"],

                "Percobaan":
                    row["Percobaan"],

                "alpha1_opt":
                    row["alpha1_opt"],

                "delta1_opt":
                    row["delta1_opt"],

                "J (fitness)":
                    row["J (fitness)"],

                "var":
                    row["var"],

                "tvar":
                    row["tvar"],

                "ctvar":
                    row["ctvar"],

                "dtvar":
                    row["dtvar"],

                "join_sig":
                    bt["join_sig"],

                "jumlah_violation":
                    bt["jumlah_violation"],

                "jumlah_data":
                    bt["jumlah_data"],

                "proporsi_error":
                    bt["proporsi_error"],

                "selisih":
                    bt["selisih"]
            })

        backtest_results = pd.DataFrame(
            all_backtest
        )

        # ========================================================
        # GLOBAL BEST
        # ========================================================

        top = backtest_results.loc[
            backtest_results[
                "selisih"
            ].idxmin()
        ]

        val_dtvar = top["dtvar"]
        val_var = top["var"]
        val_tvar = top["tvar"]
        val_ctvar = top["ctvar"]

        # ========================================================
        # GRID
        # ========================================================

        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        # ========================================================
        # GRID 1
        # ========================================================

        with row1_col1:

            st.markdown(
                "📈 **Pergerakan Return "
                "(Atas: Target, Bawah: Asosiasi)**"
            )

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.12
            )

            x_axis = (
                processed["Tanggal"]
                if "Tanggal" in processed.columns
                else processed.index
            )

            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=processed["X"].to_numpy(),
                    mode="lines",
                    name="Target",
                    line=dict(
                        color="royalblue",
                        width=1
                    )
                ),
                row=1,
                col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=processed["Y"].to_numpy(),
                    mode="lines",
                    name="Asosiasi",
                    line=dict(
                        color="darkorange",
                        width=1
                    )
                ),
                row=2,
                col=1
            )

            fig.update_layout(
                margin=dict(
                    l=5,
                    r=5,
                    t=10,
                    b=5
                ),
                height=200,
                xaxis=dict(
                    showticklabels=False
                ),
                xaxis2=dict(
                    showticklabels=False
                ),
                yaxis=dict(
                    title="Target"
                ),
                yaxis2=dict(
                    title="Asosiasi"
                ),
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="grid1_chart"
            )

        # ========================================================
        # GRID 2
        # ========================================================

        with row1_col2:

            st.markdown(
                """
                <div class="status-warning">
                    ⚠️ STATUS KESEHATAN
                </div>
                """,
                unsafe_allow_html=True
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "DTVaR",
                f"{val_dtvar:.3f}"
            )

            m2.metric(
                "VaR",
                f"{val_var:.3f}"
            )

            m3.metric(
                "TVaR",
                f"{val_tvar:.3f}"
            )

            m4.metric(
                "CTVaR",
                f"{val_ctvar:.3f}"
            )

            st.markdown(
                """
                <div style="
                    font-size: 11px;
                    color: gray;
                    margin-top: 10px;
                ">
                *Catatan: Nilai dihitung berdasarkan
                batas toleransi risiko optimal sistem.
                </div>
                """,
                unsafe_allow_html=True
            )

        # ========================================================
        # GRID 3
        # ========================================================

        with row2_col1:

            st.markdown(
                "📋 **Backtesting DTVaR "
                "(Hasil Optimasi Algoritma)**"
            )

            global_best_scenario = top[
                "Skenario"
            ]

            scenarios_data = []

            for _, row in backtest_results.iterrows():

                scenarios_data.append({

                    "Skenario":
                        row["Skenario"],

                    "α₁":
                        round(
                            row["alpha1_opt"],
                            4
                        ),

                    "δ₁":
                        round(
                            row["delta1_opt"],
                            4
                        ),

                    "DTVaR":
                        round(
                            row["dtvar"],
                            4
                        ),

                    "Joint Sig.":
                        round(
                            row["join_sig"],
                            4
                        ),

                    "Proporsi Error":
                        round(
                            row["proporsi_error"],
                            4
                        ),

                    "Diff":
                        round(
                            row["selisih"],
                            4
                        )
                })

            df_scenarios = pd.DataFrame(
                scenarios_data
            )

            st.dataframe(
                df_scenarios,
                hide_index=True,
                use_container_width=True,
                height=170
            )

            st.caption(
                f"🏆 Skenario optimal global terpilih: "
                f"**{global_best_scenario}** "
                f"(Nilai Fitness J Terendah)"
            )

        # ========================================================
        # GRID 4
        # ========================================================

        with row2_col2:

            sub_c1, sub_c2 = st.columns(
                [2, 2]
            )

            sub_c1.markdown(
                "📉 **Neg Log-Return & Pelanggaran**"
            )

            chosen_metric = sub_c2.selectbox(
                "Pilih Batas",
                [
                    "VaR",
                    "TVaR",
                    "CTVaR",
                    "DTVaR"
                ],
                label_visibility="collapsed",
                key="metric_sel_grid4"
            )

            threshold_dict = {

                "VaR":
                    val_var,

                "TVaR":
                    val_tvar,

                "CTVaR":
                    val_ctvar,

                "DTVaR":
                    val_dtvar
            }

            active_lim = threshold_dict[
                chosen_metric
            ]

            x_date = pd.to_datetime(
                processed["Tanggal"]
                if "Tanggal" in processed.columns
                else processed.index
            )

            y_neg_ret = processed[
                "X"
            ].to_numpy()

            fig_breach = go.Figure()

            fig_breach.add_trace(
                go.Scatter(
                    x=x_date,
                    y=y_neg_ret,
                    mode="lines",
                    name="Normal Return",
                    line=dict(
                        color="rgba(120,120,120,0.4)",
                        width=1
                    )
                )
            )

            fig_breach.add_hline(
                y=active_lim,
                line_dash="dash",
                line_color="darkorange",
                annotation_text=(
                    f"Batas {chosen_metric}"
                ),
                annotation_position="top left",
                annotation_font_color="darkorange"
            )

            breach_mask = (
                y_neg_ret >= active_lim
            )

            if np.any(breach_mask):

                fig_breach.add_trace(
                    go.Scatter(
                        x=x_date[
                            breach_mask
                        ],
                        y=y_neg_ret[
                            breach_mask
                        ],
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
                    )
                )

            fig_breach.update_layout(
                margin=dict(
                    l=5,
                    r=5,
                    t=10,
                    b=5
                ),
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

        st.info(
            "ℹ️ Silakan unggah file data di atas lalu "
            "klik tombol **OPTIMASI** untuk menampilkan "
            "analisis risiko."
        )


# ################################################################
# ################################################################
# DTVaR OPTIMIZATION STORY
# ################################################################
# ################################################################

elif menu == "DTVaR Optimization Story":

    st.title(
        "DTVaR Optimization Story"
    )

    # ============================================================
    # RINGKASAN PENELITIAN
    # ============================================================

    with st.container(border=True):

        st.markdown(
            "### 📌 Ringkasan Penelitian"
        )

        st.write(
            "Analisis risiko Propana Mont Belvieu dilakukan "
            "menggunakan **Dependent Tail Value at Risk (DTVaR)** "
            "dengan mempertimbangkan dependensi terhadap Brent "
            "Crude Oil. Optimasi DTVaR dilakukan menggunakan "
            "**PSO, DE, dan Hybrid PSO-DE**."
        )

    # ============================================================
    # TABS
    # ============================================================

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📌 Ringkasan",
        "📋 Data & Transformasi",
        "📈 Pergerakan Harga",
        "📉 Pergerakan Return",
        "⚙️ Hasil Optimasi",
        "📊 Evaluasi"
    ])

    # ============================================================
    # TAB 1 — RINGKASAN
    # ============================================================

    with tab1:

        st.markdown(
            "### Gambaran Penelitian"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Periode",
            "2020–2025"
        )

        c2.metric(
            "Target Risiko",
            "Propana"
        )

        c3.metric(
            "Variabel Asosiasi",
            "Brent Crude Oil"
        )

        c4.metric(
            "Observasi",
            "639"
        )

        st.markdown(
            "### Alur Analisis"
        )

        a1, a2, a3, a4 = st.columns(4)

        with a1:

            st.markdown(
                """
                **1. Data Harga**

                Data harga harian Propana Mont
                Belvieu dan Brent Crude Oil.
                """
            )

        with a2:

            st.markdown(
                """
                **2. Log Return**

                Harga ditransformasikan menjadi
                log return untuk memperoleh
                perubahan relatif.
                """
            )

        with a3:

            st.markdown(
                """
                **3. Pemodelan Risiko**

                Return Propana digunakan sebagai
                target risiko dan Brent sebagai
                variabel asosiasi.
                """
            )

        with a4:

            st.markdown(
                """
                **4. Optimasi DTVaR**

                Optimasi dilakukan menggunakan
                PSO, DE, dan Hybrid PSO-DE.
                """
            )

    # ============================================================
    # TAB 2 — DATA & TRANSFORMASI
    # ============================================================

    with tab2:

        st.markdown(
            "### Data Penelitian"
        )

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                """
                **Target Risiko**

                Propana Mont Belvieu
                """
            )

        with c2:

            st.markdown(
                """
                **Variabel Asosiasi**

                Brent Crude Oil
                """
            )

        st.markdown(
            "### Transformasi Data"
        )

        t1, t2, t3 = st.columns(3)

        with t1:

            st.markdown(
                """
                **① Harga → Log Return**

                Harga harian ditransformasikan
                menjadi log return untuk memperoleh
                perubahan relatif harga antarperiode.
                """
            )

        with t2:

            st.markdown(
                """
                **② Transformasi Risiko**

                Return Propana dikonversi menjadi
                return risiko, sedangkan return Brent
                digunakan sebagai variabel asosiasi.
                """
            )

        with t3:

            st.markdown(
                """
                **③ Penyaringan Data**

                Observasi yang memenuhi kondisi risiko
                digunakan untuk tahap estimasi distribusi
                marginal dan dependensi.
                """
            )

        st.info(
            "Periode penelitian: Januari 2020–Desember 2025."
        )

    # ============================================================
    # LOAD DATA HARGA
    # ============================================================

    @st.cache_data
    def load_story_data():

        data_target = pd.read_excel(
            "assets/EIA/crude_oil.xls"
        )

        data_asosiasi = pd.read_excel(
            "assets/EIA/propana.xlsx"
        )

        data_target["Tanggal"] = pd.to_datetime(
            data_target["Date"]
        )

        data_asosiasi["Tanggal"] = pd.to_datetime(
            data_asosiasi["Date"]
        )

        data_gabungan = pd.merge(
            data_target,
            data_asosiasi,
            on="Tanggal",
            how="inner"
        )

        data_gabungan = data_gabungan[
            (data_gabungan["Tanggal"] >= "2020-01-01")
            &
            (data_gabungan["Tanggal"] <= "2025-12-31")
        ].reset_index(drop=True)

        return data_gabungan

    data_gabungan = load_story_data()

    # ============================================================
    # TAB 3 — PERGERAKAN HARGA
    # ============================================================

    with tab3:

        st.markdown(
            "### Pergerakan Harga"
        )

        col1, col2 = st.columns(2)

        # --------------------------------------------------------
        # PROPANA
        # --------------------------------------------------------

        with col1:

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
                title="Propana Mont Belvieu",
                xaxis_title="Tahun",
                yaxis_title="USD/galon",
                height=300,
                margin=dict(
                    l=10,
                    r=10,
                    t=45,
                    b=10
                ),
                showlegend=False
            )

            st.plotly_chart(
                fig_propana,
                use_container_width=True,
                key="story_price_propana"
            )

        # --------------------------------------------------------
        # BRENT
        # --------------------------------------------------------

        with col2:

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
                title="Brent Crude Oil",
                xaxis_title="Tahun",
                yaxis_title="USD/bbl",
                height=300,
                margin=dict(
                    l=10,
                    r=10,
                    t=45,
                    b=10
                ),
                showlegend=False
            )

            st.plotly_chart(
                fig_brent,
                use_container_width=True,
                key="story_price_brent"
            )

    # ============================================================
    # TRANSFORMASI LOG RETURN
    # ============================================================

    data_gabungan["return_target"] = np.log(
        data_gabungan["propana"]
        /
        data_gabungan["propana"].shift(1)
    )

    data_gabungan["return_asosiasi"] = np.log(
        data_gabungan["crudeoil"]
        /
        data_gabungan["crudeoil"].shift(1)
    )

    data_gabungan = data_gabungan.dropna(
        subset=[
            "return_target",
            "return_asosiasi"
        ]
    ).reset_index(drop=True)

    # Return Propana menjadi return risiko
    data_gabungan["return_target"] = (
        -data_gabungan["return_target"]
    )

    # ============================================================
    # TAB 4 — PERGERAKAN RETURN
    # ============================================================

    with tab4:

        st.markdown(
            "### Pergerakan Return"
        )

        col1, col2 = st.columns(2)

        # --------------------------------------------------------
        # RETURN PROPANA
        # --------------------------------------------------------

        with col1:

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
                title="Return Risiko Propana",
                xaxis_title="Tahun",
                yaxis_title="Log Return",
                height=300,
                margin=dict(
                    l=10,
                    r=10,
                    t=45,
                    b=10
                ),
                showlegend=False
            )

            st.plotly_chart(
                fig_return_propana,
                use_container_width=True,
                key="story_return_propana"
            )

        # --------------------------------------------------------
        # RETURN BRENT
        # --------------------------------------------------------

        with col2:

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
                title="Return Brent Crude Oil",
                xaxis_title="Tahun",
                yaxis_title="Log Return",
                height=300,
                margin=dict(
                    l=10,
                    r=10,
                    t=45,
                    b=10
                ),
                showlegend=False
            )

            st.plotly_chart(
                fig_return_brent,
                use_container_width=True,
                key="story_return_brent"
            )

        # --------------------------------------------------------
        # OBSERVASI
        # --------------------------------------------------------

        st.metric(
            "Observasi setelah preprocessing",
            len(data_gabungan)
        )

        st.caption(
            "Sumber: Data harga harian EIA, "
            "periode Januari 2020–Desember 2025."
        )

    # ============================================================
    # TAB 5 — HASIL OPTIMASI
    # ============================================================

    with tab5:
        
       
        try:

            optimization_data = (
                load_optimization_file()
            )

        except Exception as e:

            st.error(
                f"Gagal membaca hasil optimasi: {e}"
            )

            st.stop()

        st.markdown(
            "### ⚙️ Hasil Optimasi DTVaR"
        )

        st.caption(
            "Hasil eksperimen PSO, DE, dan Hybrid PSO-DE "
            "berdasarkan file hasil optimasi penelitian."
        )

        # ========================================================
        # LOAD EXCEL
        # ========================================================

        try:

            optimization_data = (
                load_optimization_file()
            )

        except Exception as e:

            st.error(
                f"Gagal membaca hasil optimasi: {e}"
            )

            st.stop()

        # ========================================================
        # CARI SHEET
        # ========================================================

        hasil = find_sheet(
            optimization_data,
            [
                "Summary Lengkap",
                "Summary Lengkap",
                "optimasi",
                "results",
                "hasil"
            ]
        )

        independent = find_sheet(
            optimization_data,
            [
                "Summary Lengkap",
                "independent_run",
                "runs",
                "50 runs",
                "hasil runs"
            ]
        )

        convergence = find_sheet(
            optimization_data,
            [
                "convergence",
                "konvergensi",
                "kurva konvergensi"
            ]
        )

        # ========================================================
        # VALIDASI SHEET
        # ========================================================

        if hasil is None:

            st.error(
                "Sheet hasil optimasi tidak ditemukan "
                "di hasil_optimasi.xls."
            )

            st.write(
                "Sheet yang tersedia:",
                list(optimization_data.keys())
            )

            st.stop()

        # ========================================================
        # COPY DATA
        # ========================================================

        hasil = hasil.copy()

        # ========================================================
        # DETEKSI KOLOM
        # ========================================================

        scenario_col = find_column(
            hasil,
            [
                "Skenario",
                "scenario"
            ]
        )

        algorithm_col = find_column(
            hasil,
            [
                "Algoritma",
                "algorithm",
                "Algorithm"
            ]
        )

        fitness_col = find_column(
            hasil,
            [
                "J (fitness)",
                "fitness",
                "J_fitness",
                "J"
            ]
        )

        dtvar_col = find_column(
            hasil,
            [
                "dtvar",
                "DTVaR"
            ]
        )

        alpha1_col = find_column(
            hasil,
            [
                "alpha1_opt",
                "alpha1",
                "α₁"
            ]
        )

        delta1_col = find_column(
            hasil,
            [
                "delta1_opt",
                "delta1",
                "δ₁"
            ]
        )

        mu_col = find_column(
            hasil,
            [
                "mu_opt",
                "mu",
                "μ"
            ]
        )

        kappa_col = find_column(
            hasil,
            [
                "kappa_opt",
                "kappa",
                "κ"
            ]
        )

        runtime_col = find_column(
            hasil,
            [
                "runtime",
                "Runtime",
                "waktu"
            ]
        )

        # ========================================================
        # FILTER
        # ========================================================

        if scenario_col is not None:

            scenarios_available = (
                hasil[scenario_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        else:

            scenarios_available = [
                "(0.90, 0.60)",
                "(0.90, 0.80)",
                "(0.95, 0.60)",
                "(0.95, 0.80)"
            ]

        c1, c2 = st.columns(2)

        with c1:

            selected_scenario = st.selectbox(
                "Skenario",
                scenarios_available,
                key="story_opt_scenario"
            )

        if algorithm_col is not None:

            algorithms_available = (
                hasil[algorithm_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        else:

            algorithms_available = [
                "PSO",
                "DE",
                "Hybrid PSO-DE"
            ]

        with c2:

            selected_algorithm = st.selectbox(
                "Algoritma",
                ["Semua"] + algorithms_available,
                key="story_opt_algorithm"
            )

        # ========================================================
        # FILTER DATA
        # ========================================================

        filtered = hasil.copy()

        if scenario_col is not None:

            filtered = filtered[
                filtered[
                    scenario_col
                ].astype(str)
                ==
                str(selected_scenario)
            ]

        if (
            selected_algorithm != "Semua"
            and algorithm_col is not None
        ):

            filtered = filtered[
                filtered[
                    algorithm_col
                ].astype(str)
                ==
                selected_algorithm
            ]

        # ========================================================
        # BEST RESULT
        # ========================================================

        if (
            fitness_col is not None
            and not filtered.empty
        ):

            best_idx = filtered[
                fitness_col
            ].idxmin()

            best_row = filtered.loc[
                best_idx
            ]

            # ====================================================
            # KPI
            # ====================================================

            st.markdown(
                "### Solusi Optimal"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Fitness",
                f"{best_row[fitness_col]:.3e}"
            )

            if dtvar_col is not None:

                c2.metric(
                    "DTVaR",
                    f"{best_row[dtvar_col]:.4f}"
                )

            if alpha1_col is not None:

                c3.metric(
                    "α₁",
                    f"{best_row[alpha1_col]:.4f}"
                )

            if delta1_col is not None:

                c4.metric(
                    "δ₁",
                    f"{best_row[delta1_col]:.4f}"
                )

            c1, c2, c3, c4 = st.columns(4)

            if mu_col is not None:

                c1.metric(
                    "μ",
                    f"{best_row[mu_col]:.4f}"
                )

            if kappa_col is not None:

                c2.metric(
                    "κ",
                    f"{best_row[kappa_col]:.4f}"
                )

            if runtime_col is not None:

                c3.metric(
                    "Runtime",
                    f"{best_row[runtime_col]:.2f} s"
                )

            # ====================================================
            # TABEL HASIL
            # ====================================================

            st.markdown(
                "### Perbandingan Hasil Optimasi"
            )

            display_cols = [
                col for col in [
                    scenario_col,
                    algorithm_col,
                    fitness_col,
                    alpha1_col,
                    delta1_col,
                    mu_col,
                    kappa_col,
                    dtvar_col,
                    runtime_col
                ]
                if col is not None
            ]

            st.dataframe(
                filtered[display_cols],
                hide_index=True,
                use_container_width=True
            )

        else:

            st.warning(
                "Data hasil optimasi tidak memiliki "
                "kolom fitness yang dapat digunakan."
            )

        # ========================================================
        # DISTRIBUSI 50 INDEPENDENT RUNS
        # ========================================================

        if independent is not None:

            st.markdown(
                "### Distribusi Fitness Independent Runs"
            )

            run_scenario_col = find_column(
                independent,
                [
                    "Skenario",
                    "scenario"
                ]
            )

            run_algorithm_col = find_column(
                independent,
                [
                    "Algoritma",
                    "algorithm"
                ]
            )

            run_fitness_col = find_column(
                independent,
                [
                    "J (fitness)",
                    "fitness",
                    "J"
                ]
            )

            if (
                run_fitness_col is not None
                and run_algorithm_col is not None
            ):

                run_plot = independent.copy()

                if run_scenario_col is not None:

                    run_plot = run_plot[
                        run_plot[
                            run_scenario_col
                        ].astype(str)
                        ==
                        str(selected_scenario)
                    ]

                if (
                    selected_algorithm != "Semua"
                ):

                    run_plot = run_plot[
                        run_plot[
                            run_algorithm_col
                        ].astype(str)
                        ==
                        selected_algorithm
                    ]

                fig_box = go.Figure()

                for algo in (
                    run_plot[
                        run_algorithm_col
                    ]
                    .dropna()
                    .unique()
                ):

                    data_algo = run_plot[
                        run_plot[
                            run_algorithm_col
                        ]
                        ==
                        algo
                    ]

                    fig_box.add_trace(
                        go.Box(
                            y=data_algo[
                                run_fitness_col
                            ],
                            name=str(algo),
                            boxpoints="outliers"
                        )
                    )

                fig_box.update_layout(
                    title="Distribusi Fitness",
                    yaxis_title="J (Fitness)",
                    height=380,
                    hovermode="closest"
                )

                st.plotly_chart(
                    fig_box,
                    use_container_width=True,
                    key="story_fitness_boxplot"
                )


        # ============================================================
    # ============================================================
    # CONVERGENCE — SUMMARY ITERASI
    # ============================================================
    # ============================================================
    # CONVERGENCE — SUMMARY ITERASI
    # ============================================================
    st.markdown("### 📈 Konvergensi Fitness per Iterasi")

    summary_iterasi = find_sheet(
        optimization_data,
        ["Summary Iterasi"]
    )

    if summary_iterasi is None:
        st.warning(
            "Sheet **Summary Iterasi** tidak ditemukan "
            "pada file hasil optimasi."
        )

    else:
        conv = summary_iterasi.copy()

        # --------------------------------------------------------
        # Bersihkan nama kolom
        # --------------------------------------------------------
        conv.columns = [
            str(c).strip()
            for c in conv.columns
        ]

        # --------------------------------------------------------
        # Tampilkan nama kolom untuk pengecekan
        # --------------------------------------------------------
        # st.write("Kolom Summary Iterasi:", list(conv.columns))

        # --------------------------------------------------------
        # Fungsi pencarian kolom berdasarkan kata kunci
        # --------------------------------------------------------
        def find_conv_column(df, keywords):
            """
            Mencari satu kolom berdasarkan kata kunci.
            Exact match diprioritaskan.
            """
            columns = list(df.columns)

            # 1. Exact match
            for col in columns:
                col_norm = str(col).strip().lower()

                for key in keywords:
                    if col_norm == str(key).strip().lower():
                        return col

            # 2. Partial match
            for col in columns:
                col_norm = str(col).strip().lower()

                for key in keywords:
                    if str(key).strip().lower() in col_norm:
                        return col

            return None

        # --------------------------------------------------------
        # Cari kolom
        # --------------------------------------------------------
        iter_col = find_conv_column(
            conv,
            [
                "Iterasi",
                "Iteration"
            ]
        )

        hpsode_col = find_conv_column(
            conv,
            [
                "Mean HPSODE",
                "HPSODE",
                "Hybrid PSO-DE",
                "Hybrid PSO DE",
                "Hybrid"
            ]
        )

        pso_col = find_conv_column(
            conv,
            [
                "Mean PSO",
                "PSO"
            ]
        )

        de_col = find_conv_column(
            conv,
            [
                "Mean DE",
                "DE"
            ]
        )

        # --------------------------------------------------------
        # Jika kolom tidak ditemukan
        # --------------------------------------------------------
        missing_cols = []

        if iter_col is None:
            missing_cols.append("Iterasi")

        if hpsode_col is None:
            missing_cols.append("HPSODE")

        if pso_col is None:
            missing_cols.append("PSO")

        if de_col is None:
            missing_cols.append("DE")

        if missing_cols:

            st.error(
                "Kolom pada sheet **Summary Iterasi** "
                "tidak dapat dikenali."
            )

            st.write(
                "Kolom yang ditemukan pada Excel:"
            )

            st.code(
                "\n".join(
                    str(c)
                    for c in conv.columns
                )
            )

            st.write(
                "Kolom yang belum ditemukan:"
            )

            st.write(
                missing_cols
            )

        else:

            # ----------------------------------------------------
            # Ambil kolom menggunakan nama aktual Excel
            # ----------------------------------------------------
            conv_plot = pd.DataFrame({
                "Iterasi": conv[iter_col],
                "Mean HPSODE": conv[hpsode_col],
                "Mean PSO": conv[pso_col],
                "Mean DE": conv[de_col]
            })

            # ----------------------------------------------------
            # Konversi numerik
            # ----------------------------------------------------
            for col in [
                "Iterasi",
                "Mean HPSODE",
                "Mean PSO",
                "Mean DE"
            ]:

                conv_plot[col] = pd.to_numeric(
                    conv_plot[col],
                    errors="coerce"
                )

            # ----------------------------------------------------
            # Hapus baris kosong
            # ----------------------------------------------------
            conv_plot = conv_plot.dropna(
                subset=["Iterasi"]
            )

            conv_plot = conv_plot.sort_values(
                "Iterasi"
            )

            # ----------------------------------------------------
            # Cek data
            # ----------------------------------------------------
            if conv_plot.empty:

                st.warning(
                    "Tidak terdapat data iterasi yang valid "
                    "pada Summary Iterasi."
                )

            else:

                # ------------------------------------------------
                # Grafik konvergensi
                # ------------------------------------------------
                fig_conv = go.Figure()

                fig_conv.add_trace(
                    go.Scatter(
                        x=conv_plot["Iterasi"],
                        y=conv_plot["Mean HPSODE"],
                        mode="lines",
                        name="Hybrid PSO-DE",
                        line=dict(width=3),
                        hovertemplate=(
                            "Iterasi: %{x}<br>"
                            "Mean Fitness: %{y:.6e}"
                            "<extra>Hybrid PSO-DE</extra>"
                        )
                    )
                )

                fig_conv.add_trace(
                    go.Scatter(
                        x=conv_plot["Iterasi"],
                        y=conv_plot["Mean PSO"],
                        mode="lines",
                        name="PSO",
                        line=dict(width=2),
                        hovertemplate=(
                            "Iterasi: %{x}<br>"
                            "Mean Fitness: %{y:.6e}"
                            "<extra>PSO</extra>"
                        )
                    )
                )

                fig_conv.add_trace(
                    go.Scatter(
                        x=conv_plot["Iterasi"],
                        y=conv_plot["Mean DE"],
                        mode="lines",
                        name="DE",
                        line=dict(width=2),
                        hovertemplate=(
                            "Iterasi: %{x}<br>"
                            "Mean Fitness: %{y:.6e}"
                            "<extra>DE</extra>"
                        )
                    )
                )

                fig_conv.update_layout(
                    title=(
                        "Perbandingan Konvergensi Fitness "
                        f"({CONVERGENCE_SCENARIO})"
                    ),
                    xaxis_title="Iterasi",
                    yaxis_title="Mean Fitness",
                    hovermode="x unified",
                    height=500,
                    margin=dict(
                        l=20,
                        r=20,
                        t=60,
                        b=20
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                # ------------------------------------------------
                # Gunakan skala log jika semua fitness positif
                # ------------------------------------------------
                fitness_data = conv_plot[
                    [
                        "Mean HPSODE",
                        "Mean PSO",
                        "Mean DE"
                    ]
                ].to_numpy(
                    dtype=float
                )

                finite_values = fitness_data[
                    np.isfinite(fitness_data)
                ]

                if (
                    len(finite_values) > 0
                    and np.all(finite_values > 0)
                ):
                    fig_conv.update_yaxes(
                        type="log"
                    )

                # ------------------------------------------------
                # Tampilkan grafik
                # ------------------------------------------------
                st.plotly_chart(
                    fig_conv,
                    use_container_width=True
                )

                st.caption(
                    "Sumber: sheet **Summary Iterasi**. "
                    "Grafik menampilkan rata-rata fitness "
                    "pada setiap iterasi untuk PSO, DE, "
                    "dan Hybrid PSO-DE."
                )

                # ------------------------------------------------
                # Tabel data
                # ------------------------------------------------
                with st.expander(
                    "Lihat Data Summary Iterasi"
                ):

                    st.dataframe(
                        conv_plot,
                        use_container_width=True,
                        hide_index=True
                    )
    # ============================================================
    # TAB 6 — EVALUASI & BACKTESTING
    # ============================================================

    with tab6:

       
        try:

            optimization_data = (
                load_optimization_file()
            )

        except Exception as e:

            st.error(
                f"Gagal membaca hasil optimasi: {e}"
            )

            st.stop()

        backtesting = find_sheet(
            optimization_data,
            [
                "Ringkasan_Backtesting",
                "backtest",
                "hasil backtesting",
                "evaluasi"
            ]
        )

        if backtesting is None:

            st.warning(
                "Sheet backtesting tidak ditemukan "
                "dalam hasil_optimasi.xls."
            )

            st.stop()

        backtesting = backtesting.copy()

        # ========================================================
        # DETEKSI KOLOM
        # ========================================================

        bt_date_col = find_column(
            backtesting,
            [
                "Tanggal",
                "date",
                "Date"
            ]
        )

        bt_scenario_col = find_column(
            backtesting,
            [
                "Skenario",
                "scenario"
            ]
        )

        bt_algorithm_col = find_column(
            backtesting,
            [
                "Algoritma",
                "algorithm"
            ]
        )

        bt_actual_col = find_column(
            backtesting,
            [
                "Actual_Loss",
                "actual_loss",
                "Actual Loss",
                "loss",
                "Loss"
            ]
        )

        bt_dtvar_col = find_column(
            backtesting,
            [
                "DTVaR",
                "dtvar"
            ]
        )

        bt_var_col = find_column(
            backtesting,
            [
                "VaR",
                "var"
            ]
        )

        bt_tvar_col = find_column(
            backtesting,
            [
                "TVaR",
                "tvar"
            ]
        )

        bt_violation_col = find_column(
            backtesting,
            [
                "Selisih",
                "violation",
                "Pelanggaran"
            ]
        )

        # ========================================================
        # FILTER
        # ========================================================

        bt = backtesting.copy()

        if bt_scenario_col is not None:

            bt = bt[
                bt[
                    bt_scenario_col
                ].astype(str)
                ==
                str(selected_scenario)
            ]

        if (
            selected_algorithm != "Semua"
            and bt_algorithm_col is not None
        ):

            bt = bt[
                bt[
                    bt_algorithm_col
                ].astype(str)
                ==
                selected_algorithm
            ]

        # ========================================================
        # BACKTESTING KPI
        # ========================================================


        # ========================================================
        # GRAFIK BACKTESTING
        # ========================================================

        if (
            bt_date_col is not None
            and bt_actual_col is not None
        ):

            bt[bt_date_col] = pd.to_datetime(
                bt[bt_date_col],
                errors="coerce"
            )

            bt = bt.sort_values(
                bt_date_col
            )

            fig_bt = go.Figure()

            # Actual Loss
            fig_bt.add_trace(
                go.Scatter(
                    x=bt[bt_date_col],
                    y=bt[bt_actual_col],
                    mode="lines",
                    name="Actual Loss"
                )
            )

            # DTVaR
            if bt_dtvar_col is not None:

                fig_bt.add_trace(
                    go.Scatter(
                        x=bt[bt_date_col],
                        y=bt[bt_dtvar_col],
                        mode="lines",
                        name="DTVaR",
                        line=dict(
                            dash="dash"
                        )
                    )
                )

            # VaR
            if bt_var_col is not None:

                fig_bt.add_trace(
                    go.Scatter(
                        x=bt[bt_date_col],
                        y=bt[bt_var_col],
                        mode="lines",
                        name="VaR",
                        line=dict(
                            dash="dot"
                        )
                    )
                )

            # Violation
            if (
                bt_violation_col is not None
            ):

                violation_mask = (
                    pd.to_numeric(
                        bt[
                            bt_violation_col
                        ],
                        errors="coerce"
                    )
                    .fillna(0)
                    .astype(int)
                    == 1
                )

                violation_data = bt[
                    violation_mask
                ]

                if not violation_data.empty:

                    fig_bt.add_trace(
                        go.Scatter(
                            x=violation_data[
                                bt_date_col
                            ],
                            y=violation_data[
                                bt_actual_col
                            ],
                            mode="markers",
                            name="Violation",
                            marker=dict(
                                size=9,
                                symbol="x"
                            )
                        )
                    )

            fig_bt.update_layout(
                title="Backtesting DTVaR",
                xaxis_title="Tanggal",
                yaxis_title="Loss",
                height=430,
                hovermode="x unified"
            )

            st.plotly_chart(
                fig_bt,
                use_container_width=True,
                key="story_backtesting"
            )

        # ========================================================
        # TABEL DATA BACKTESTING
        # ========================================================

        st.markdown(
            "### Ringkasan Backtesting"
        )

        display_bt_cols = [
            col for col in [
                bt_date_col,
                bt_algorithm_col,
                bt_actual_col,
                bt_var_col,
                bt_tvar_col,
                bt_dtvar_col,
                bt_violation_col
            ]
            if col is not None
        ]

 

        if display_bt_cols:

            st.dataframe(
                bt[display_bt_cols],
                hide_index=True,
                use_container_width=True,
                height=250
            )

        st.caption(
            "Grafik bersifat interaktif. Gunakan hover, "
            "zoom, dan legenda untuk mengeksplorasi hasil."
        )