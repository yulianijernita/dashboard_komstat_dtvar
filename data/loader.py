"""Pembacaan data dari file CSV / XLSX."""
import numpy as np
import pandas as pd

DATE_NAMES = ("date", "tanggal", "tgl", "waktu", "time")


def load_table(uploaded_file):
    """Baca file upload (CSV atau XLSX) menjadi DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def all_columns(df):
    return df.columns.tolist()


def guess_date_column(columns):
    """Tebak kolom tanggal dari namanya; None jika tidak ada."""
    for c in columns:
        if str(c).strip().lower() in DATE_NAMES:
            return c
    return None