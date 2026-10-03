"""
Pemrosesan data harga mentah -> data siap estimasi & backtesting.

Alur:
  1. urutkan data dari waktu lama -> baru (jika ada kolom tanggal)
  2. hitung log return untuk X dan Y
  3. X dikalikan (-1), Y tetap
  4. ambil hanya baris dengan X > 0; Y mengikuti baris X yang terambil
"""
import numpy as np
import pandas as pd

LOGRET_FORMULAS = {
    "standard": "ln(P_t / P_{t-1})  — log return standar",
    "literal": "ln((P_t − P_{t-1}) / P_t)  — sesuai tulisan awal",
}


def log_return(prices, formula="standard"):
    """Log return dari deret harga urut lama -> baru. Panjang hasil = n - 1."""
    p = np.asarray(prices, dtype=float)
    prev, cur = p[:-1], p[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        if formula == "standard":
            r = np.log(cur / prev)
        elif formula == "literal":
            r = np.log((cur - prev) / cur)   # NaN jika harga tidak naik
        else:
            raise ValueError(f"Rumus log return tidak dikenal: {formula}")
    r[~np.isfinite(r)] = np.nan
    return r


def preprocess(df, col_x, col_y, date_col=None, formula="standard", dayfirst=False):
    """
    Return (processed_df, info).
    processed_df berkolom [Tanggal (jika ada), X, Y]; X sudah dikali (-1) dan > 0.
    """
    cols = [c for c in (date_col, col_x, col_y) if c]
    d = df[cols].copy()

    # 1. Urutkan lama -> baru
    if date_col:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce", dayfirst=dayfirst)
        d = d.dropna(subset=[date_col]).sort_values(date_col)
    d = d.dropna(subset=[col_x, col_y]).reset_index(drop=True)
    if len(d) < 3:
        raise ValueError("Data harga terlalu sedikit.")

    # 2. Log return (baris pertama hilang)
    out = pd.DataFrame({
        "X": -log_return(d[col_x], formula),   # 3. X dikali (-1)
        "Y": log_return(d[col_y], formula),    #    Y tetap
    })
    if date_col:
        out.insert(0, "Tanggal", d[date_col].iloc[1:].to_numpy())

    out = out.dropna(subset=["X", "Y"])
    n_return = len(out)

    # 4. Filter X positif (Y mengikuti baris yang sama)
    out = out[out["X"] > 0].reset_index(drop=True)

    info = {"n_raw": len(d), "n_return": n_return, "n_final": len(out)}
    if info["n_final"] < 10:
        raise ValueError(f"Data X positif hanya {info['n_final']} baris; tidak cukup untuk estimasi.")
    return out, info