# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:34:27 2026

@author: Leo
"""
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _clean_label(label) -> str | None:
    if label is None:
        return None
    s = str(label).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s if s else None


def _parse_time_label(value) -> float | None:
    """
    Convierte etiquetas tipo:
        t=0
        t=0.01d
        t = 1 d
    a float en días.
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    s = s.replace("t", "").replace("=", "").replace("d", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def read_domain_raw(excel_path: str | Path, sheet_name: str = "domain") -> pd.DataFrame:
    """
    Lee la hoja completa sin encabezados.
    """
    return pd.read_excel(excel_path, sheet_name=sheet_name, header=None)


def extract_domain_data(
    excel_path: str | Path,
    sheet_name: str = "domain"
) -> Tuple[np.ndarray, List[float], pd.DataFrame]:

    df = read_domain_raw(excel_path, sheet_name=sheet_name)

    # fila 1: coordenadas espaciales
    x = pd.to_numeric(df.iloc[1, 1:], errors="coerce").dropna().to_numpy(dtype=float)
    nx = len(x)

    # encontrar bloques aq species conc
    block_rows = []
    for i in range(len(df)):
        c0 = _clean_label(df.iloc[i, 0])
        c1 = df.iloc[i, 1] if df.shape[1] > 1 else None

        if c0 == "aq species conc":
            t = _parse_time_label(c1)
            if t is not None:
                block_rows.append((i, t))

    if not block_rows:
        raise ValueError("No se encontraron bloques 'aq species conc'.")

    records = []
    times = []

    for ib, (block_row, t) in enumerate(block_rows):
        times.append(t)

        # final del bloque: antes del siguiente aq species conc
        if ib < len(block_rows) - 1:
            next_block = block_rows[ib + 1][0]
        else:
            next_block = len(df)

        section = "species"

        for r in range(block_row + 1, next_block):
            name = _clean_label(df.iloc[r, 0])

            if name is None:
                continue

            # cambiar sección si aparecen encabezados
            if name == "vol frac":
                section = "volume_fraction"
                continue

            if name == "r_eq":
                section = "r_eq"
                continue

            if name == "rk":
                section = "rk"
                continue

            vals = pd.to_numeric(
                df.iloc[r, 1:1 + nx],
                errors="coerce"
            ).to_numpy(dtype=float)

            if np.all(np.isnan(vals)):
                continue

            for xi, vi in zip(x, vals):
                records.append({
                    "time": t,
                    "kind": section,
                    "name": name,
                    "x": float(xi),
                    "value": float(vi),
                })

    df_long = pd.DataFrame.from_records(records)
    times = sorted(set(times))

    return x, times, df_long


def list_available_names(df_long: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Regresa nombres disponibles para especies y componentes.
    """
    out = {}
    for kind in ["species", "component"]:
        names = sorted(df_long.loc[df_long["kind"] == kind, "name"].unique().tolist())
        out[kind] = names
    return out


def get_variable_matrix(
    df_long: pd.DataFrame,
    name: str,
    kind: str = "species"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrae una variable como matriz M(time, x).

    Parámetros
    ----------
    df_long : DataFrame largo
    name : str
        Nombre, por ejemplo:
            'ch2o(aq)', 'n2(aq)', 'ca+2', 'u1', 'u3'
    kind : str
        'species' o 'component'

    Retorna
    -------
    times : ndarray shape (nt,)
    x : ndarray shape (nx,)
    values : ndarray shape (nt, nx)
    """
    name = _clean_label(name)
    kind = _clean_label(kind)

    sub = df_long[(df_long["kind"] == kind) & (df_long["name"] == name)].copy()
    if sub.empty:
        raise ValueError(f"No se encontró '{name}' como '{kind}'.")

    pivot = sub.pivot_table(index="time", columns="x", values="value", aggfunc="first")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    times = pivot.index.to_numpy(dtype=float)
    x = pivot.columns.to_numpy(dtype=float)
    values = pivot.to_numpy(dtype=float)
    return times, x, values


def get_profile_at_time(
    df_long: pd.DataFrame,
    name: str,
    time_value: float,
    kind: str = "species"
) -> pd.DataFrame:
    """
    Extrae el perfil espacial de una variable para un tiempo dado.

    Retorna DataFrame con columnas ['x', 'value'].
    """
    name = _clean_label(name)
    kind = _clean_label(kind)

    sub = df_long[
        (df_long["kind"] == kind) &
        (df_long["name"] == name) &
        (df_long["time"] == time_value)
    ][["x", "value"]].copy()

    if sub.empty:
        raise ValueError(f"No se encontró '{name}' en t={time_value} como '{kind}'.")

    return sub.sort_values("x").reset_index(drop=True)


def get_snapshot(
    df_long: pd.DataFrame,
    time_value: float,
    kind: str = "species"
) -> pd.DataFrame:
    """
    Extrae todas las variables de un tiempo dado.

    Retorna DataFrame con columnas:
        ['name', 'x', 'value']
    """
    kind = _clean_label(kind)
    sub = df_long[
        (df_long["kind"] == kind) &
        (df_long["time"] == time_value)
    ][["name", "x", "value"]].copy()

    if sub.empty:
        raise ValueError(f"No se encontraron datos para t={time_value} y kind='{kind}'.")

    return sub.sort_values(["name", "x"]).reset_index(drop=True)


def get_time_series_at_x(
    df_long: pd.DataFrame,
    name: str,
    x_value: float,
    kind: str = "species",
    tol: float = 1e-12
) -> pd.DataFrame:
    """
    Extrae la serie temporal de una variable en una posición x dada.

    Retorna DataFrame con columnas ['time', 'value'].
    """
    name = _clean_label(name)
    kind = _clean_label(kind)

    sub = df_long[
        (df_long["kind"] == kind) &
        (df_long["name"] == name) &
        (np.abs(df_long["x"] - x_value) < tol)
    ][["time", "value"]].copy()

    if sub.empty:
        raise ValueError(f"No se encontró '{name}' en x={x_value} como '{kind}'.")

    return sub.sort_values("time").reset_index(drop=True)


if __name__ == "__main__":

    archivo = "denit_case1_v8.xlsx"

    # ===== 1) EXTRAER TODO =====
    x, times, df_long = extract_domain_data(archivo)

    print("\n=== Coordenadas x ===")
    print(x)

    print("\n=== Tiempos disponibles ===")
    print(times)

    # ===== 2) VER QUÉ VARIABLES EXISTEN =====
    names = list_available_names(df_long)

    print("\n=== Especies disponibles ===")
    print(names["species"])

    print("\n=== Componentes disponibles ===")
    print(names["component"])

    # ===== 3) EXTRAER MATRIZ COMPLETA (time, x) =====
    # ejemplo: nitrato
    times_no3, x_no3, NO3 = get_variable_matrix(
        df_long,
        name="no3-",
        kind="species"
    )

    print("\n=== Matriz NO3 shape ===")
    print(NO3.shape)

    # ===== 4) PERFIL ESPACIAL A UN TIEMPO =====
    perfil = get_profile_at_time(
        df_long,
        name="no3-",
        time_value=1.0,
        kind="species"
    )

    print("\n=== Perfil NO3 a t=1.0 ===")
    print(perfil.head())

    # ===== 5) SERIE TEMPORAL EN UN PUNTO =====
    serie = get_time_series_at_x(
        df_long,
        name="no3-",
        x_value=0.5,
        kind="species"
    )

    print("\n=== Serie temporal en x=0.5 ===")
    print(serie)

    # ===== 6) SNAPSHOT COMPLETO =====
    snapshot = get_snapshot(
        df_long,
        time_value=1.0,
        kind="species"
    )

    print("\n=== Snapshot completo t=1.0 ===")
    print(snapshot.head())